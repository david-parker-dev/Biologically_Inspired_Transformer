import gymnasium as gym
import torch
from agent.Rollout_Buffer import RolloutBuffer
from models.Model import Model
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class Agent:
    def __init__(self, config, num_actions, env, episode_max_length):

        # Passed Parameters
        self.env = env
        self.config = config
        self.num_actions = num_actions
        self.episode_max_length = episode_max_length

        # Object Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Network = Model(num_actions, config=self.config).to(self.device)
        self.Optimiser = optim.AdamW(self.Network.parameters(), lr=self.config.LEARNING_RATE, amsgrad=False)
        self.writer = SummaryWriter(log_dir="runs/" + self.config.RUN_NAME)
        self.RolloutBuffer = RolloutBuffer(self.config, image_shape=(config.INPUT_GRID_SIZE, config.INPUT_GRID_SIZE, 3))
        self.memory = None
        self.eval_env = gym.make(self.config.ENV_NAME)

        # Counters
        self.global_step = 0
        self.episode_timestep = 0
        self.episode_return = 0.0

        # Starting Observation
        self.observation, _ = self.env.reset(seed=self.config.SEED)

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

    def select_action(self, observation, eval=False):

        # Add batch and sequence dimensions: (H, W, C) -> (1, 1, H, W, C)
        image = torch.as_tensor(observation["image"], dtype=torch.float32, device=self.device)
        image = image.unsqueeze(0).unsqueeze(0)

        # Add batch and sequence dimensions: scalar -> (1, 1)
        direction = torch.as_tensor(observation["direction"], dtype=torch.int64, device=self.device)
        direction = direction.reshape(1, 1)

        with torch.no_grad():
            critic_value, actor_logits, new_memory = self.Network(image, direction, memory=self.memory)
            self.memory = new_memory

            critic_value = critic_value[:, -1, :].squeeze(-1)
            actor_logits = actor_logits[:, -1, :]

            distribution = torch.distributions.Categorical(logits=actor_logits)

            if eval:
                action = torch.argmax(actor_logits, dim=-1)
            else:
                action = distribution.sample()

            log_probability = distribution.log_prob(action)

            return action, log_probability, critic_value

    def fill_rollout(self):

        last_done = False

        for timestep in range(self.config.ROLLOUT_SIZE):

            action, log_probability, critic_value = self.select_action(self.observation)

            next_observation, reward, terminated, truncated, _ = self.env.step(action.item())

            done = terminated or truncated
            last_done = done

            self.RolloutBuffer.store(timestep,
                                      self.observation["image"],
                                      self.observation["direction"],
                                      action, reward, critic_value, log_probability, done)

            self.episode_timestep += 1
            self.episode_return += reward
            self.global_step += 1

            if done or self.episode_timestep >= self.episode_max_length:

                self.writer.add_scalar("rollout/episode_length_vs_steps", self.episode_timestep, self.global_step, )
                self.writer.add_scalar("rollout/episode_return_vs_steps", self.episode_return, self.global_step, )

                self.observation, _ = self.env.reset()
                self.episode_timestep = 0
                self.episode_return = 0.0
                self.memory = None

            else:
                self.observation = next_observation

        if last_done:
            bootstrap_value = 0.0
        else:
            with torch.no_grad():
                _, _, bootstrap_value = self.select_action(self.observation)

        return bootstrap_value, last_done

    def update(self):

        for epoch in range(self.config.EPOCHS):

            memory = None
            batches = self.RolloutBuffer.get_sequence_batches(sequence_length=self.config.SEQUENCE_LENGTH)

            for batch in batches:

                images, directions, actions, old_log_probs, advantages, returns, old_values, dones = batch

                batch_actions       = actions.reshape(-1)
                batch_old_log_probs = old_log_probs.reshape(-1)
                batch_old_values    = old_values.reshape(-1)
                batch_advantages    = advantages.reshape(-1)
                batch_returns       = returns.reshape(-1)

                new_values, actor_logits, memory = self.Network(images, directions, memory=memory)

                if dones.any():
                    memory = None

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

                total_loss = (policy_loss + self.config.VALUE_LOSS_COEFFICIENT * value_loss - self.config.ENTROPY_COEFFICIENT * entropy_loss)

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
        episode_returns = []
        episode_lengths = []
        episode_successes = []

        training_memory = self.memory

        for episode in range(self.config.EVAL_EPISODES):
            if episode == 0:
                observation, _ = self.eval_env.reset(seed=self.config.EVAL_SEED)
            else:
                observation, _ = self.eval_env.reset()

            done = False
            total_reward = 0.0
            steps = 0
            self.memory = None

            while not done and steps < self.episode_max_length:
                action, _, _ = self.select_action(observation, eval=True)
                observation, reward, terminated, truncated, _ = self.eval_env.step(action.item())
                done = terminated or truncated
                total_reward += reward
                steps += 1

            episode_returns.append(total_reward)
            episode_lengths.append(steps)
            episode_successes.append(1.0 if terminated else 0.0)

        self.memory = training_memory

        returns_tensor = torch.tensor(episode_returns)
        average_return = returns_tensor.mean().item()
        success_rate = sum(episode_successes) / len(episode_successes)
        average_length = sum(episode_lengths) / len(episode_lengths)

        self.writer.add_scalar("eval/mean_return", average_return, self.global_step)
        self.writer.add_scalar("eval/std_return", returns_tensor.std().item(), self.global_step)
        self.writer.add_scalar("eval/success_rate", success_rate, self.global_step)
        self.writer.add_scalar("eval/mean_episode_length", average_length, self.global_step)

        return average_return
