import copy

import gym
import numpy as np
import torch

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dapr.models import Actor, Critic
from dapr.utils import ReplayBuffer


class DDPG:
    def __init__(
        self,
        env_name: str,
        env_seed: int = 42,
        device: torch.device = torch.device("cpu"),
        writer: SummaryWriter = None,
        batch_size: int = 256,
        discount: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        actor_weight_decay: float = 3e-4,
        critic_weight_decay: float = 3e-4,
        start_step: int = int(2.5e4),
        buffer_size: int = int(1e6),
        expl_noise: float = 0.1,
        eval_freq: int = int(5e3),
        max_steps: int = int(1e6),
        log_freq: int = 10,
    ):
        super().__init__()
        # Env
        self.env_name = env_name
        self.env_seed = env_seed

        self.env = gym.make(env_name)
        self.env.seed(env_seed)

        self.device = device
        self.writer = writer

        self.state_dim: int = self.env.observation_space.shape[0]
        self.action_dim: int = self.env.action_space.shape[0]
        self.max_action: int = self.env.action_space.high[0]

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)

        # Hyperparams
        self.batch_size: int = batch_size
        self.discount: float = discount
        self.tau: float = tau
        self.actor_lr: float = actor_lr
        self.critic_lr: float = critic_lr
        self.actor_weight_decay: float = actor_weight_decay
        self.critic_weight_decay: float = critic_weight_decay
        self.start_step: int = start_step
        self.expl_noise: float = expl_noise
        self.eval_freq: int = eval_freq
        self.max_steps: int = max_steps
        self.log_freq = log_freq

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            weight_decay=self.actor_weight_decay,
        )

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=self.critic_weight_decay,
        )

        # Metrics
        self.total_steps: int = 0

    def predict(self, state):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def eval(self, num_episodes: int = 10):
        eval_env = gym.make(self.env_name)
        eval_env.seed(self.env_seed)

        rewards = []

        for _ in range(num_episodes):

            state, done = eval_env.reset(), False
            episode_reward = 0

            while not done:
                action = self.predict(state)
                state, reward, done, _ = eval_env.step(action)

                episode_reward += reward

            rewards.append(episode_reward)

        avg_reward = np.mean(rewards)
        print("---------------------------------------")
        print(f"Evaluation over {num_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")

        return avg_reward

    def optimize(self):

        # Reorder in replay buffer to be s, a, r, s', d
        # Sample the Replay Buffer
        state, action, next_state, reward, not_done = self.buffer.sample(
            self.batch_size, device=self.device
        )

        # Get the target Q Values
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = (
            reward + (not_done * self.discount * target_Q).detach()
        )  # Why detatch?

        # Get the current Q Values
        current_Q = self.critic(state, action)

        # Compute Critic Loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Update Critic by one-step gradient descent
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Compute the actor loss
        actor_loss = -torch.mean(self.critic(state, self.actor(state)))

        # Update the Actor by one-step gradient descent (not ascent as in paper, since we multiplied by -1)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update the frozen target models
        for param, tgt_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            tgt_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * tgt_param.data
            )

        for param, tgt_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            tgt_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * tgt_param.data
            )

        return actor_loss, critic_loss

    def _warm_start(self):
        # print(f"Warm Start... Populating Buffer.")
        state = self.env.reset()

        for _ in tqdm(range(self.start_step)):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)

            self.buffer.add(state, action, next_state, reward, float(done))
            state = next_state

            if done:
                self.env.reset()

    def train(self):
        self._warm_start()

        state, done = self.env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in tqdm(range(self.max_steps)):
            episode_timesteps += 1

            action = self.predict(state)
            noise = np.random.normal(loc=0, scale=self.expl_noise * self.max_action)
            clipped_action_with_noise = (action + noise).clip(
                -self.max_action, self.max_action
            )

            next_state, reward, done, _ = self.env.step(clipped_action_with_noise)

            self.buffer.add(state, action, next_state, reward, float(done))

            state = next_state
            episode_reward += reward

            actor_loss, critic_loss = self.optimize()

            if self.writer is not None and (t + 1) % self.log_freq == 0:
                self.writer.add_scalar("Loss/Actor", actor_loss, global_step=t)
                self.writer.add_scalar("Loss/Critc", critic_loss, global_step=t)

            if done:
                # print(
                #     f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
                # )

                if self.writer is not None and (t + 1) % self.log_freq == 0:
                    self.writer.add_scalar(
                        "episode_timesteps", episode_timesteps, global_step=t
                    )
                    self.writer.add_scalar(
                        "Reward/Train", episode_reward, global_step=t
                    )

                # Reset Environment
                state, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if (t + 1) % self.eval_freq == 0:
                avg_reward = self.eval()
                if self.writer is not None:
                    self.writer.add_scalar("Reward/Test", avg_reward, global_step=t)
