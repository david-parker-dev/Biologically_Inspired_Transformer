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

def make_env(env_name, seed, index):
    def thunk():
        env = gym.make(env_name)
        return env
    return thunk


def main():
    config = Config()
    # env = gym.make(config.ENV_NAME)
    envs = gym.vector.SyncVectorEnv(
            [make_env(config.ENV_NAME, config.SEED, i) for i in range(config.NUM_ENVS)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,)

    set_RNG(config.SEED)

    agent = Agent(
            config=config,
            num_actions=envs.single_action_space.n,
            env=envs,
            episode_max_length=config.EPISODE_MAX_LENGTH,
        )

    agent.training()

if __name__ == "__main__":
    main()
