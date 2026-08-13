import torch


class RolloutBuffer:
    # Collects State Transitions for PPO Training
    # Returns a rollout of state transitions

    def __init__(self, config, image_shape):

        # Defines Device to be used - GPU(Cuda) or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_shape = image_shape
        self.num_envs = config.NUM_ENVS

        # Hyperparameters
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.rollout_size = config.ROLLOUT_SIZE

        self.clear()


    def store(self, step, image, direction, action, reward, critic_value, log_probability, done):
        self.images[step]            = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        self.directions[step]        = torch.as_tensor(direction, dtype=torch.int64, device=self.device)
        self.actions[step]           = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        self.log_probabilities[step] = torch.as_tensor(log_probability, dtype=torch.float32, device=self.device)
        self.rewards[step]           = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.critic_values[step]     = torch.as_tensor(critic_value, dtype=torch.float32, device=self.device)
        self.dones[step]             = torch.as_tensor(done, dtype=torch.bool, device=self.device)

    def compute_gae(self, bootstrap_value, last_done):

        current_advantage = torch.zeros(self.num_envs, device=self.device)

        for timestep in reversed(range(self.rollout_size)):
            if timestep == self.rollout_size - 1:
                next_value = bootstrap_value
            else:
                next_value = self.critic_values[timestep + 1]

            next_done = self.dones[timestep].float()

            # TD Error = Reward(t) + (Discount_Factor * Value(t+1)) - Critic_Value(t)
            TD_error = self.rewards[timestep] + (self.gamma * next_value * (1 - next_done)) - self.critic_values[timestep]

            # Advantage = TD Error(t) + (Discount_Factor * GAE_Lambda) * Advantage(t+1)
            current_advantage = TD_error + (self.gamma * self.gae_lambda) * (1 - next_done) * current_advantage

            # Target Return = Advantages + Critic_Values
            target_return = current_advantage + self.critic_values[timestep]

            # Advantage - Action Value vs Critic Predicted Value (A = Q - V)
            self.advantages[timestep]   = current_advantage

            # Returns - True Discounted State Values (R = A + V)
            self.returns[timestep]      = target_return

        return self.advantages, self.returns

    def get_sequence_batches(self, sequence_length):

        assert self.rollout_size % sequence_length == 0, "SEQUENCE_LENGTH must be divisable by ROLLOUT_SIZE"

        normalised_advantage = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        for start in range(0, self.rollout_size, sequence_length):
            end = start + sequence_length

            yield (
                self.images            [start:end].transpose(0, 1),
                self.directions        [start:end].transpose(0, 1),
                self.actions           [start:end].transpose(0, 1),
                self.log_probabilities [start:end].transpose(0, 1),
                normalised_advantage   [start:end].transpose(0, 1),
                self.returns           [start:end].transpose(0, 1),
                self.critic_values     [start:end].transpose(0, 1),
                self.dones             [start:end].transpose(0, 1),
            )

    def clear(self):
        self.images = torch.zeros(self.rollout_size, self.num_envs, *self.image_shape, dtype=torch.float32, device=self.device)
        self.directions = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.int64, device=self.device)
        self.actions = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.int64, device=self.device)
        self.log_probabilities = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.float32, device=self.device)
        self.critic_values = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.bool, device=self.device)

        self.advantages = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.rollout_size, self.num_envs, dtype=torch.float32, device=self.device)
