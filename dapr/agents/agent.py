from abc import ABC, abstractmethod
import os
from typing import List
import uuid

import gym
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


class GenericAgent(ABC):
    def __init__(
        self,
        env_name: str,
        env_seed: int = 42,
        id: uuid.UUID = uuid.uuid4(),
        device: torch.device = torch.device("cpu"),
        writer: SummaryWriter = None,
    ):
        super().__init__()
        # Env
        self.env_name = env_name
        self.env_seed = env_seed

        self.env = gym.make(env_name)
        self.env.seed(env_seed)

        self.device = device
        self.writer = writer

        self.checkpoint_dir = os.path.join("checkpoints", f"DDPG_{env_name}-{id}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.state_dim: int = self.env.observation_space.shape[0]
        self.action_dim: int = self.env.action_space.shape[0]
        self.max_action: int = self.env.action_space.high[0]

        # Metrics
        self.total_steps: int = 0

    @abstractmethod
    def predict(self, state) -> np.ndarray:
        pass

    def eval(self, num_episodes: int = 10):
        eval_env = gym.make(self.env_name)

        rewards = []
        best_epsiode_reward = None

        for _ in range(num_episodes):

            state, done = eval_env.reset(), False
            episode_reward = 0

            while not done:
                action = self.predict(state)
                state, reward, done, _ = eval_env.step(action)

                episode_reward += reward

            rewards.append(episode_reward)
            if best_epsiode_reward is None:
                best_epsiode_reward = episode_reward

            best_epsiode_reward = max(episode_reward, best_epsiode_reward)

        avg_reward = np.mean(rewards)
        print(
            f"Evaluation over {num_episodes} episodes:\nRewards Avg: {avg_reward:.3f} Rewards Best: {best_epsiode_reward:.3f}"
        )

        return avg_reward, best_epsiode_reward

    @abstractmethod
    def optimize(self) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def train(self):
        pass

    def test(self, num_episodes=10):
        avg_reward, best_reward = self.eval(num_episodes=num_episodes)

        if self.writer is not None:
            self.writer.add_scalar("reward/test_avg", avg_reward, global_step=0)
            self.writer.add_scalar("reward/test_best", best_reward, global_step=0)

            self.writer.add_hparams(
                self.hparams, {"hp_metric": avg_reward,}, run_name=f"test",
            )

    @abstractmethod
    def load(self, checkpoint_dir):
        pass

    @abstractmethod
    def save(self):
        pass
