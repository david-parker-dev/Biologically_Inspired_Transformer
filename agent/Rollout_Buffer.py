import torch


class RolloutBuffer:
    # Collects State Transitions for PPO Training
    # Returns a rollout of state transitions

    def __init__(self, config, image_shape):

        # Defines Device to be used - GPU(Cuda) or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameters
        self.image_shape = image_shape

        # Hyperparameters
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.rollout_size = config.ROLLOUT_SIZE

        self.clear()


    def store(self, step, image, direction, action, reward, critic_value, log_probability, done):

        # Append Collected Transition Values
        self.images[step]            = torch.as_tensor(image, dtype=torch.float32, device=self.device)
        self.directions[step]        = torch.as_tensor(direction, dtype=torch.int64, device=self.device)
        self.actions[step]           = action
        self.log_probabilities[step] = log_probability
        self.rewards[step]           = reward
        self.critic_values[step]     = critic_value
        self.dones[step]             = done

    def compute_gae(self, bootstrap_value, last_done):

        current_advantage = 0.0

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

        # Normalise advantage over the rollout buffer
        normalised_advantage = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        for start in range(0, self.rollout_size, sequence_length):
            end = start + sequence_length

            yield (
                self.images            [start:end].unsqueeze(0),
                self.directions        [start:end].unsqueeze(0),
                self.actions           [start:end].unsqueeze(0),
                self.log_probabilities [start:end].unsqueeze(0),
                normalised_advantage   [start:end].unsqueeze(0),
                self.returns           [start:end].unsqueeze(0),
                self.critic_values     [start:end].unsqueeze(0),
                self.dones             [start:end].unsqueeze(0),
            )


    def clear(self):
        self.images = torch.zeros(self.rollout_size, *self.image_shape, dtype=torch.float32, device=self.device)
        self.directions = torch.zeros(self.rollout_size, dtype=torch.int64, device=self.device)
        self.actions = torch.zeros(self.rollout_size, dtype=torch.int64, device=self.device)
        self.log_probabilities = torch.zeros(self.rollout_size, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.rollout_size, dtype=torch.float32, device=self.device)
        self.critic_values = torch.zeros(self.rollout_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.rollout_size, dtype=torch.bool, device=self.device)

        self.advantages = torch.zeros(self.rollout_size, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.rollout_size, dtype=torch.float32, device=self.device)
