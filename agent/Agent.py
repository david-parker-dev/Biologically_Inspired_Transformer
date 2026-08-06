import math

import gymnasium as gym
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from agent.Rollout_Buffer import RolloutBuffer
from models.Model import Model


class Agent:
    def __init__(self, config, num_observations, num_actions, env, episode_max_length):

        self.env = env
        self.config = config
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.episode_max_length = episode_max_length

        self.flat_observation_size = math.prod(num_observations)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Network = Model(self.flat_observation_size,
                             self.config.MODEL_EMBED_SIZE,
                             num_actions,
                             num_layers=self.config.MODEL_LAYERS,
                             max_sequence_length=self.config.SEQUENCE_LENGTH,
                             max_memory_length=self.config.MAX_MEMORY_LENGTH,
                             ).to(self.device)

        self.Optimiser = optim.AdamW(self.Network.parameters(), lr=self.config.LEARNING_RATE, amsgrad=False)

        self.writer = SummaryWriter(log_dir="runs/" + self.config.RUN_NAME)
        self.RolloutBuffer = RolloutBuffer(self.config, self.num_observations)
        self.global_step = 0

        self.observation, _ = self.env.reset()
        self.episode_timestep = 0
        self.episode_return = 0.0
        self.memory = None

    def training(self):
        for iteration in range(self.config.NUMBER_ITERATIONS):

            # Collect a full rollout of experience
            bootstrap_value, last_done = self.fill_rollout()

            # Compute Advantages/Returns
            self.RolloutBuffer.compute_gae(bootstrap_value, last_done)

            # PPO Update
            self.update()

            if iteration % self.config.EVAL_FREQUENCY == 0:
                self.evaluate()

    def select_action(self, state, eval=False):
        state = torch.from_numpy(state).float().to(self.device)

        # Ensure state has a batch dimension
        if state.ndim == len(self.num_observations):
            state = state.unsqueeze(0)
        state = state.reshape(state.size(0), -1)
        state = state.unsqueeze(1)


        with torch.no_grad():
            critic_value, actor_logits, new_memory = self.Network(state, memory=self.memory)
            self.memory = new_memory

            # Last Timestep
            critic_value = critic_value[:, -1, :].squeeze(-1)
            actor_logits = actor_logits[:, -1, :]

            # Next create a categorical distribution to sample from
            distribution = torch.distributions.Categorical(logits=actor_logits)

            if eval:
                action = torch.argmax(actor_logits, dim=-1)
            else:
                action = distribution.sample()

            # Find the entropy of the current transition
            log_probability = distribution.log_prob(action)

            return action, log_probability, critic_value

    def fill_rollout(self):

        last_done = False

        for timestep in range(self.config.ROLLOUT_SIZE):

            # Extract Models Action Selection
            action, log_probability, critic_value = self.select_action(self.observation)

            # Step Environment Given Model Action Choice
            next_observation, reward, terminated, truncated, _ = self.env.step(action.item())

            # Combine terminated & truncated into single variable
            done = terminated or truncated
            last_done = done

            # Store Collected Data in Rollout Buffer
            self.RolloutBuffer.store(timestep, self.observation, action, reward, critic_value, log_probability, done)

            self.episode_timestep += 1
            self.episode_return += reward
            self.global_step += 1

            # Check if episode ended
            if done or self.episode_timestep >= self.episode_max_length:

                # Log performance to Tensorboard
                self.writer.add_scalar("rollout/episode_length_vs_steps", self.episode_timestep, self.global_step, )
                self.writer.add_scalar("rollout/episode_return_vs_steps", self.episode_return, self.global_step, )

                # Reset
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

                observation, actions, old_log_probs, advantages, returns, old_values, dones = batch

                batch_size, sequence_length = observation.shape[0], observation.shape[1]
                batch_observations  = observation.reshape(batch_size, sequence_length, self.flat_observation_size)
                batch_actions       = actions.reshape(-1)
                batch_old_log_probs = old_log_probs.reshape(-1)
                batch_old_values    = old_values.reshape(-1)
                batch_advantages    = advantages.reshape(-1)
                batch_returns       = returns.reshape(-1)

                 # Current Values and Policy
                new_values, actor_logits, memory = self.Network(batch_observations, memory=memory)

                if dones.any():
                    memory = None

                actor_logits = actor_logits.reshape(-1, self.num_actions)
                new_values = new_values.reshape(-1)

                # Categorical Distribution
                distribution = torch.distributions.Categorical(logits=actor_logits)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy_loss = distribution.entropy().mean()

                # Probability Ratio
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # Unclipped term
                term1 = ratio * batch_advantages

                # Clipped Term
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.CLIP_EPS, 1.0 + self.config.CLIP_EPS)
                term2 = clipped_ratio * batch_advantages

                # Clipped Policy Loss
                policy_loss = -torch.min(term1, term2).mean()

                # Value Function Loss
                value_loss_unclipped = (new_values - batch_returns) ** 2
                value_clipped = batch_old_values + torch.clamp(new_values - batch_old_values, -self.config.CLIP_EPS, self.config.CLIP_EPS)
                value_loss_clipped = (value_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Total Loss
                total_loss = (policy_loss + self.config.VALUE_LOSS_COEFFICIENT * value_loss - self.config.ENTROPY_COEFFICIENT * entropy_loss)

                # Gradients / Backprop
                self.Optimiser.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Network.parameters(), self.config.MAX_GRAD_NORM)
                self.Optimiser.step()

    def evaluate(self, num_episodes=10):
        episode_returns = []
        eval_env = gym.make("CartPole-v1")

        training_memory = self.memory

        for _ in range(num_episodes):
            observation, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            self.memory = None

            while not done and steps < self.episode_max_length:
                action, _, _ = self.select_action(observation, eval=True)
                observation, reward, terminated, truncated, _ = eval_env.step(action.item())
                done = terminated or truncated
                total_reward += reward
                steps += 1

            episode_returns.append(total_reward)

        self.memory = training_memory

        average_return = sum(episode_returns) / len(episode_returns)
        self.writer.add_scalar("eval/mean_return", average_return, self.global_step)
        return average_return
