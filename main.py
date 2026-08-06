import gymnasium as gym

from agent.Agent import Agent
from config import config as Config


def main():
    config = Config()
    env = gym.make("CartPole-v1")

    agent = Agent(
        config=config,
        num_observations=env.observation_space.shape,
        num_actions=env.action_space.n,
        env=env,
        episode_max_length=500,
    )

    agent.training()

if __name__ == "__main__":
    main()
