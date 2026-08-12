import random

import gymnasium as gym
import minigrid
import numpy as np
import torch

from agent.Agent import Agent
from config import config as Config


def set_RNG(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    config = Config()
    env = gym.make(config.ENV_NAME)

    set_RNG(config.SEED)

    agent = Agent(
            config=config,
            num_actions=env.action_space.n,
            env=env,
            episode_max_length=config.EPISODE_MAX_LENGTH,
        )

    agent.training()

if __name__ == "__main__":
    main()
