import copy
import gym
import numpy as np

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from dapr.models import Actor, Critic
from dapr.utils import ExperienceBufferDataset, ReplayBuffer


class DDPG(pl.LightningModule):
    def __init__(
        self,
        env: str,
        env_seed: int = 42,
        batch_size: int = 256,
        discount: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_weight_decay: float = 1e-3,
        critic_weight_decay: float = 1e-3,
        max_timesteps: int = int(1e6),
        expl_noise: float = 0.1,
        replay_buffer_maxsize: int = int(1e6),
        avg_reward_len: int = 100,
        warm_start_size: int = int(2.5e4),
        batches_per_epoch: int = int(5e3),
    ):
        super().__init__()

        self.env = gym.make(env)
        self.env.seed(env_seed)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.save_hyperparameters()
        self.replay_buffer = ReplayBuffer(
            self.state_dim, self.action_dim, self.hparams.replay_buffer_maxsize
        )

        # Metrics
        self.done_episodes = 0
        self.total_rewards = [0]
        self.total_episode_steps = [0]
        self.avg_rewards = 0.0
        self.total_steps = 0

        # Turn off automatic optimization. Multiple models must be optimized simultaneously
        self.automatic_optimization = False

    def forward(self, state):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def eval_policy(self, num_episodes):
        rewards = []

        for _ in range(num_episodes):
            state, done = self.env.reset(), False
            episode_reward = 0

            while not done:
                action = self.forward(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)

        return rewards

    def training_step(self, batch, batch_idx):
        actor_opt, critic_opt = self.optimizers()

        # Reorder in replay buffer to be s, a, r, s', d
        # Sample the Replay Buffer
        state, action, next_state, reward, not_done = batch

        # Get the target Q Values
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = (
            reward + (not_done * self.hparams.discount * target_Q).detach()
        )  # Why detatch?

        # Get the current Q Values
        current_Q = self.critic(state, action)

        # Compute Critic Loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Update Critic by one-step gradient descent
        critic_opt.zero_grad()
        self.manual_backward(critic_loss)
        critic_opt.step()

        # Compute the actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update the Actor by one-step gradient descent (not ascent as in paper, since we multiplied by -1)
        actor_opt.zero_grad()
        self.manual_backward(actor_loss)
        actor_opt.step()

        # Update the frozen target models
        for param, tgt_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            tgt_param.data.copy_(
                self.hparams.tau * param.data + (1 - self.hparams.tau) * tgt_param.data
            )

        for param, tgt_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            tgt_param.data.copy_(
                self.hparams.tau * param.data + (1 - self.hparams.tau) * tgt_param.data
            )

        self.log("avg_reward", self.avg_rewards, prog_bar=True)
        self.log_dict(
            {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "total_reward": self.total_rewards[-1],
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

    def val_step(self, *args, **kwargs):
        val_rewards = self.eval_policy(10)
        avg_reward = np.mean(val_rewards)

        return {"val_reward": avg_reward}

    def val_epoch_end(self, outputs) -> None:
        rewards = [x["val_reward"] for x in outputs]
        avg_reward = np.mean(rewards)

        self.log("Val/avg_reward", avg_reward, prog_bar=True)

    def _dataloader(self, batch_size=None) -> DataLoader:
        self.populate(self.hparams.warm_start_size)
        self.dataset = ExperienceBufferDataset(self.gen_train_batch)
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size if batch_size is None else batch_size,
        )

    def train_dataloader(self):
        return self._dataloader()

    def test_dataloader(self):
        return self._dataloader(batch_size=1)

    def populate(self, warm_start_size):
        state = self.env.reset()

        for _ in range(warm_start_size):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.add(state, action, next_state, reward, float(done))
            state = next_state

            if done:
                state = self.env.reset()

    def gen_train_batch(self):
        state, done = self.env.reset(), False
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1

            noise = np.random.normal(
                loc=0, scale=self.hparams.expl_noise, size=self.action_dim
            )
            action = self.forward(state)
            clipped_noisy_action = (action + noise).clip(
                -self.max_action, self.max_action
            )

            next_state, reward, done, _ = self.env.step(clipped_noisy_action)

            self.replay_buffer.add(
                state, clipped_noisy_action, next_state, reward, float(done)
            )

            episode_reward += reward
            state = next_state
            episode_steps += 1

            if done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(
                    np.mean(self.total_rewards[-self.hparams.avg_reward_len :])
                )

                state, done = self.env.reset(), False
                episode_reward = 0
                episode_steps = 0

            # states, actions, next_states, rewards, not_dones = self.replay_buffer.sample(self.hparams.batch_size, device=self.device)
            # for idx, _ in enumerate(not_dones):
            #     yield states[idx], actions[idx], next_states[idx], rewards[idx], not_dones[idx]

            for batch in zip(
                *self.replay_buffer.sample(self.hparams.batch_size, device=self.device)
            ):
                yield batch

            # Simulate epochs.
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break

    def configure_optimizers(self):
        actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.hparams.actor_lr,
            weight_decay=self.hparams.actor_weight_decay,
        )

        critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.hparams.critic_lr,
            weight_decay=self.hparams.critic_weight_decay,
        )

        return actor_opt, critic_opt
