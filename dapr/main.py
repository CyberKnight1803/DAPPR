import torch
import numpy as np
import os 
import gym 
import gc 
from models import DDPG
import memory

from constants import (
    MAX_EPISODES,
    MAX_STEPS,
    MAX_BUFFER,
    MAX_TOTAL_REWARD,
    HIDDEN_DIMS,
    BATCH_SIZE,
    LEARNING_RATE,
    GAMMA,
    TAU
)

env = gym.make('BipedalWalker-v3')

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]


def main():
    print(f"State Dimensions :- {S_DIM}")
    print(f"Action Dimensions :- {A_DIM}")
    print(f"Action Max :- {A_MAX}")

    ram = memory.ReplayBuffer(MAX_BUFFER)
    trainer = DDPG(S_DIM, HIDDEN_DIMS, A_DIM, A_MAX, ram, lr=LEARNING_RATE, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU)

    for _ep in range(MAX_EPISODES):
        observation = env.reset()
        print(f"EPISODE :- {_ep}")
        for r in range(MAX_STEPS):
            env.render()
            state = np.float32(observation)

            action = trainer.get_exploration_action(state)

            new_observation, reward, done, info = env.step(action)

            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
                # push this exp in ram
                ram.add(state, action, reward, new_state)

            observation = new_observation

            # perform optimization
            trainer.optimize()
            if done:
                break

        # check memory consumption and clear memory
        gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)

        if _ep%100 == 0:
            trainer.save_models(_ep)


if __name__ == '__main__':
    main()