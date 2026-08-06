import torch


class rollout_buffer:
    # Collects State Transitions for PPO Training
    # Returns a rollout of state transitions

    # Transition Design
        # ((Obs_1, Obs_2, ..., Obs_x), action, log_probability, reward, critic_value, done)
    # Rollout tensors have space
        # observations      = [Rollout_Size, Observation_Size]
        # actions           = [Rollout_Size]
        # log_probabilties  = [Rollout_Size]
        # rewards           = [Rollout_Size]
        # critic_values     = [Rollout_Size]
        # dones             = [Rollout_Size]

    def __init__(self, config, obs_shape):

        # Defines Device to be used - GPU(Cuda) or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameters
        self.obs_shape = obs_shape

        # Hyperparameters
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.rollout_size = config.ROLLOUT_SIZE

        self.clear()


    def store(self, step, observation, action, reward, critic_value, log_probability, done):

        # Append Collected Transition Values
        self.observations[step]      = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        self.actions[step]           = action
        self.log_probabilities[step] = log_probability
        self.rewards[step]           = reward
        self.critic_values[step]     = critic_value
        self.dones[step]             = done

    def compute_gae(self, bootstrap_value, last_done):
        # Outputs
            # Advantage - Tensor, Shape: [Rollout_Size] - Action Value vs Critic Predicted Value (A = Q - V)
            # Returns   - Tensor, Shape: [Rollout_Size] - True Discounted State Values (R = A + V)
        # Math
            # Advantage(t)  = TD Error(t) + (Discount_Factor * GAE_Lambda) * Advantage(t+1)
            # TD Error(t)   = Reward(t) + (Discount_Factor * Value(t+1)) - Critic_Value(t)
            # Returns       = Advantages + Critic_Valued

        current_advantage = 0.0

        for timestep in reversed(range(self.rollout_size)):

            if timestep == self.rollout_size - 1:
                next_value = bootstrap_value
            else:
                next_value = self.critic_values[timestep + 1]

            next_done = self.dones[timestep].float()

            # TD - Temporal Difference
            TD_error = self.rewards[timestep] + (self.gamma * next_value * (1 - next_done)) - self.critic_values[timestep]

            # GAE - Generalised Advantage Estimation
            current_advantage = TD_error + (self.gamma * self.gae_lambda) * (1 - next_done) * current_advantage

            # Target Return (Critic)
            target_return = current_advantage + self.critic_values[timestep]

            self.advantages[timestep]   = current_advantage
            self.returns[timestep]      = target_return

        return self.advantages, self.returns

    def get_sequence_batches(self, sequence_length):

        assert self.rollout_size % sequence_length == 0, "ROLLOUT_SIZE must be a multiple of SEQUENCE_LENGTH"

        # Normalise advantage over the rollout buffer
        normalised_advantage = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        for start in range(0, self.rollout_size, sequence_length):
            end = start + sequence_length

            yield (
                self.observations      [start:end].unsqueeze(0),
                self.actions           [start:end].unsqueeze(0),
                self.log_probabilities [start:end].unsqueeze(0),
                normalised_advantage   [start:end].unsqueeze(0),
                self.returns           [start:end].unsqueeze(0),
                self.critic_values     [start:end].unsqueeze(0),
                self.dones             [start:end].unsqueeze(0),
            )


    def clear(self):
        self.observations       = torch.zeros(self.rollout_size, *self.obs_shape,    dtype=torch.float32,    device=self.device) # Full State Observation
        self.actions            = torch.zeros(self.rollout_size,                    dtype=torch.int64,      device=self.device) # Chosen Action
        self.log_probabilities  = torch.zeros(self.rollout_size,                    dtype=torch.float32,    device=self.device) # Log Probability of Chosen Action
        self.rewards            = torch.zeros(self.rollout_size,                    dtype=torch.float32,    device=self.device) # Reward of the State
        self.critic_values      = torch.zeros(self.rollout_size,                    dtype=torch.float32,    device=self.device) # Critic Value of the State
        self.dones              = torch.zeros(self.rollout_size,                    dtype=torch.bool,       device=self.device) # True/False Terminal State

        # Calcuated Loss Metrics
        self.advantages         = torch.zeros(self.rollout_size, dtype=torch.float32, device=self.device)
        self.returns            = torch.zeros(self.rollout_size, dtype=torch.float32, device=self.device)
