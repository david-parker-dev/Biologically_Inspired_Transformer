import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from agent.Rollout_Buffer import RolloutBuffer
from models.Model import Model


class Agent:
    def __init__(self, config, num_actions, env, eval_env, episode_max_length):

        # Passed Parameters
        self.env = env
        self.config = config
        self.num_actions = num_actions
        self.episode_max_length = episode_max_length
        self.eval_env = eval_env

        # Object Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Network = Model(num_actions, config=self.config).to(self.device)
        self.Optimiser = optim.AdamW(self.Network.parameters(), lr=self.config.LEARNING_RATE, amsgrad=False)
        self.writer = SummaryWriter(log_dir="runs/" + self.config.RUN_NAME)
        self.RolloutBuffer = RolloutBuffer(self.config, image_shape=(config.INPUT_GRID_SIZE, config.INPUT_GRID_SIZE, 3))
        self.memory = None

        # Counters
        self.episode_timestep = np.zeros(self.config.NUM_ENVS, dtype=np.int64)
        self.episode_return = np.zeros(self.config.NUM_ENVS, dtype=np.float64)
        self.global_step = 0

        # Starting Observation
        seeds = [self.config.SEED + i for i in range(self.config.NUM_ENVS)]
        self.observation, _ = self.env.reset(seed=seeds)

    def training(self):

        # PPO Iteration Loop
        for iteration in range(self.config.NUMBER_ITERATIONS):

            # Collect Samples
            bootstrap_value, last_done = self.fill_rollout()

            # Compute Training Numebers
            self.RolloutBuffer.compute_gae(bootstrap_value, last_done)

            # Update Model
            self.update()

            # Evaluate Current Model Performance
            if iteration % self.config.EVAL_FREQUENCY == 0:
                self.evaluate()

                if self.config.ENABLE_SPARSITY:
                    for layer_index, block in enumerate(self.Network.blocks):
                        alphas = 1.0 + F.softplus(block.an1.alpha)
                        for head_index, alpha in enumerate(alphas):
                            self.writer.add_scalar(f"alpha/layer_{layer_index}_head_{head_index}", alpha.item(), self.global_step)

    def select_action(self, observation, memory, eval=False):

        # Add batch and sequence dimensions
        image = torch.as_tensor(observation["image"], dtype=torch.float32, device=self.device)
        direction = torch.as_tensor(observation["direction"], dtype=torch.int64, device=self.device)
        image = image.unsqueeze(1)
        direction = direction.unsqueeze(1)

        with torch.no_grad():
            critic_value, actor_logits, new_memory = self.Network(image, direction, memory=memory)

            critic_value = critic_value[:, -1, :].squeeze(-1)
            actor_logits = actor_logits[:, -1, :]

            distribution = torch.distributions.Categorical(logits=actor_logits)

            if eval:
                action = torch.argmax(actor_logits, dim=-1)
            else:
                action = distribution.sample()

            log_probability = distribution.log_prob(action)

            return action, log_probability, critic_value, new_memory

    def fill_rollout(self):

        last_done = False

        for timestep in range(self.config.ROLLOUT_SIZE):

            action, log_probability, critic_value, self.memory = self.select_action(self.observation, self.memory)

            next_observation, reward, terminated, truncated, infos = self.env.step(action.cpu().numpy())

            logged_reward = reward.copy()

            truncation_only = truncated & (~terminated)

            if truncation_only.any():
                indices = np.where(truncation_only)[0]
                truncated_obs = {
                    "image": np.stack([infos["final_obs"][i]["image"] for i in indices]),
                    "direction": np.array([infos["final_obs"][i]["direction"] for i in indices]),
                }
                truncated_memory = [layer_memory[indices] for layer_memory in self.memory]

                with torch.no_grad():
                    _, _, truncation_value, _ = self.select_action(truncated_obs, truncated_memory)

                reward[indices] = reward[indices] + self.config.GAMMA * truncation_value.cpu().numpy()

            done = terminated | truncated
            last_done = done

            self.RolloutBuffer.store(timestep,
                                      self.observation["image"],
                                      self.observation["direction"],
                                      action, reward, critic_value, log_probability, done)

            if done.any():
                keep = torch.as_tensor(~done, dtype=torch.float32, device=self.device).view(-1, 1, 1)
                self.memory = [layer_memory * keep for layer_memory in self.memory]

            self.episode_timestep += 1
            self.episode_return += logged_reward
            self.global_step += self.config.NUM_ENVS

            for i in range(self.config.NUM_ENVS):
                if done[i]:
                    self.writer.add_scalar("rollout/episode_length_vs_steps", self.episode_timestep[i], self.global_step)
                    self.writer.add_scalar("rollout/episode_return_vs_steps", self.episode_return[i], self.global_step)
                    self.episode_timestep[i] = 0
                    self.episode_return[i] = 0.0

            self.observation = next_observation

        with torch.no_grad():
            _, _, raw_bootstrap_value, _ = self.select_action(self.observation, self.memory, eval=True)
        bootstrap_value = raw_bootstrap_value * torch.as_tensor(~last_done, dtype=torch.float32, device=self.device)

        return bootstrap_value, last_done

    def update(self):

        for epoch in range(self.config.EPOCHS):

            batches = self.RolloutBuffer.get_sequence_batches(sequence_length=self.config.SEQUENCE_LENGTH)

            for batch in batches:

                images, directions, actions, old_log_probs, advantages, returns, old_values, dones = batch

                batch_actions       = actions.reshape(-1)
                batch_old_log_probs = old_log_probs.reshape(-1)
                batch_old_values    = old_values.reshape(-1)
                batch_advantages    = advantages.reshape(-1)
                batch_returns       = returns.reshape(-1)

                new_values, actor_logits, _ = self.Network(images, directions, memory=None, dones=dones)
                actor_logits = actor_logits.reshape(-1, self.num_actions)
                new_values = new_values.reshape(-1)

                distribution = torch.distributions.Categorical(logits=actor_logits)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy_loss = distribution.entropy().mean()

                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                term1 = ratio * batch_advantages
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.CLIP_EPS, 1.0 + self.config.CLIP_EPS)
                term2 = clipped_ratio * batch_advantages
                policy_loss = -torch.min(term1, term2).mean()

                value_loss_unclipped = (new_values - batch_returns) ** 2
                value_clipped = batch_old_values + torch.clamp(new_values - batch_old_values, -self.config.CLIP_EPS, self.config.CLIP_EPS)
                value_loss_clipped = (value_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                total_loss = (  policy_loss
                                + self.config.VALUE_LOSS_COEFFICIENT * value_loss
                                - self.config.ENTROPY_COEFFICIENT * entropy_loss)

                self.Optimiser.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.Network.parameters(), self.config.MAX_GRAD_NORM)
                self.Optimiser.step()

                # Log training diagnostics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()

                self.writer.add_scalar("loss/policy_loss", policy_loss.item(), self.global_step)
                self.writer.add_scalar("loss/value_loss", value_loss.item(), self.global_step)
                self.writer.add_scalar("loss/entropy_loss", entropy_loss.item(), self.global_step)
                self.writer.add_scalar("loss/total_loss", total_loss.item(), self.global_step)
                self.writer.add_scalar("diagnostics/grad_norm", grad_norm.item(), self.global_step)
                self.writer.add_scalar("diagnostics/approx_kl", approx_kl.item(), self.global_step)

    def evaluate(self):
        eval_seeds = self.config.EVAL_SEEDS
        observation, _ = self.eval_env.reset(seed=eval_seeds)

        n = self.config.EVAL_EPISODES
        memory = None
        finished = np.zeros(n, dtype=bool)
        total_reward = np.zeros(n, dtype=np.float64)
        steps = np.zeros(n, dtype=np.int64)
        success = np.zeros(n, dtype=np.float64)

        for step in range(self.episode_max_length):
            if finished.all():
                break

            action, _, _, memory = self.select_action(observation, memory, eval=True)
            observation, reward, terminated, truncated, _ = self.eval_env.step(action.cpu().numpy())

            active = ~finished
            total_reward[active] += reward[active]
            steps[active] += 1

            newly_done = (terminated | truncated) & active
            success[newly_done] = (terminated[newly_done] & (total_reward[newly_done] > 0)).astype(np.float64)
            finished |= newly_done

        returns_tensor = torch.as_tensor(total_reward, dtype=torch.float32)
        average_return = returns_tensor.mean().item()
        success_rate = success.mean()
        average_length = steps.mean()

        self.writer.add_scalar("eval/mean_return", average_return, self.global_step)
        self.writer.add_scalar("eval/std_return", returns_tensor.std().item(), self.global_step)
        self.writer.add_scalar("eval/success_rate", success_rate, self.global_step)
        self.writer.add_scalar("eval/mean_episode_length", average_length, self.global_step)

        return average_return
