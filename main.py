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

def make_env(env_name):
    def thunk():
        env = gym.make(env_name)
        return env
    return thunk


def main():
    config = Config()
    set_RNG(config.SEED)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config.ENV_NAME) for _ in range(config.NUM_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        shared_memory=False,
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(config.ENV_NAME) for _ in range(config.EVAL_EPISODES)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    agent = Agent(
            config=config,
            num_actions=envs.single_action_space.n,
            env=envs,
            eval_env=eval_envs,
            episode_max_length=config.EPISODE_MAX_LENGTH,
        )

    agent.training()

if __name__ == "__main__":
    main()
