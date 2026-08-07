import gymnasium as gym
import minigrid

from agent.Agent import Agent
from config import config as Config


def main():
    config = Config()
    env = gym.make(config.ENV_NAME)

    agent = Agent(
            config=config,
            num_actions=env.action_space.n,
            env=env,
            episode_max_length=config.EPISODE_MAX_LENGTH,
        )

    agent.training()

if __name__ == "__main__":
    main()
