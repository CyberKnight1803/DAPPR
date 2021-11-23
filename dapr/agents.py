import copy
import os
import uuid

import gym
import numpy as np
import torch

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dapr.models import Actor, Critic
from dapr.utils import ReplayBuffer, he_init_weights, xavier_init_weights


class DDPG:
    def __init__(
        self,
        env_name: str,
        env_seed: int = 42,
        id: uuid.UUID = uuid.uuid4(),
        device: torch.device = torch.device("cpu"),
        writer: SummaryWriter = None,
        initialization: str = "he",
        batch_size: int = 100,
        discount: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        actor_weight_decay: float = 1e-3,
        critic_weight_decay: float = 1e-3,
        start_step: int = int(1e4),
        buffer_size: int = int(1e6),
        expl_noise: float = 0.1,
        eval_freq: int = int(5e3),
        max_steps: int = int(1e6),
        log_freq: int = 10,
    ):
        super().__init__()
        # Hyperparams
        self.hparams = {}
        self.hparams["init_weight"] = initialization
        self.hparams["batch_size"] = batch_size
        self.hparams["discount"] = discount
        self.hparams["tau"] = tau
        self.hparams["actor_lr"] = actor_lr
        self.hparams["critic_lr"] = critic_lr
        self.hparams["actor_weight_decay"] = actor_weight_decay
        self.hparams["critic_weight_decay"] = critic_weight_decay
        self.hparams["start_step"] = start_step
        self.hparams["expl_noise"] = expl_noise
        self.hparams["max_steps"] = max_steps

        self.log_freq = log_freq
        self.eval_freq = eval_freq

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

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)

        # Models

        if initialization == "xavier" or initialization == "golort":
            self.init_weights = xavier_init_weights
        else:
            self.init_weights = he_init_weights

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor.apply(self.init_weights)
        self.actor_target = copy.deepcopy(self.actor)

        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.hparams["actor_lr"],
            weight_decay=self.hparams["actor_weight_decay"],
        )

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic.apply(self.init_weights)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.hparams["critic_lr"],
            weight_decay=self.hparams["critic_weight_decay"],
        )

        # Metrics
        self.total_steps: int = 0

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

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

    def optimize(self):
        # self.actor.train()
        # Reorder in replay buffer to be s, a, r, s', d
        # Sample the Replay Buffer
        state, action, next_state, reward, not_done = self.buffer.sample(
            self.hparams["batch_size"], device=self.device
        )

        # Get the target Q Values
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = (
            reward + (not_done * self.hparams["discount"] * target_Q).detach()
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
        actor_loss = -1 * torch.mean(self.critic(state, self.actor(state)))

        # Update the Actor by one-step gradient descent (not ascent as in paper, since we multiplied by -1)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update the frozen target models
        for param, tgt_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            tgt_param.data.copy_(
                self.hparams["tau"] * param.data
                + (1 - self.hparams["tau"]) * tgt_param.data
            )

        for param, tgt_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            tgt_param.data.copy_(
                self.hparams["tau"] * param.data
                + (1 - self.hparams["tau"]) * tgt_param.data
            )

        # self.actor.eval()
        return actor_loss, critic_loss

    def _warm_start(self):
        state = self.env.reset()

        for _ in tqdm(range(self.hparams["start_step"])):
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

        for t in tqdm(range(self.hparams["max_steps"])):
            self.total_steps += 1
            episode_timesteps += 1

            action = self.predict(state)
            noise = np.random.normal(
                loc=0, scale=self.hparams["expl_noise"] * self.max_action
            )
            clipped_action_with_noise = (action + noise).clip(
                -self.max_action, self.max_action
            )

            next_state, reward, done, _ = self.env.step(clipped_action_with_noise)

            self.buffer.add(state, action, next_state, reward, float(done))

            state = next_state
            episode_reward += reward

            actor_loss, critic_loss = self.optimize()

            if (t + 1) % self.log_freq == 0:
                self.writer.add_scalar(
                    "loss/actor", actor_loss, global_step=self.total_steps
                )
                self.writer.add_scalar(
                    "loss/critc", critic_loss, global_step=self.total_steps
                )

            if done:
                if (t + 1) % self.log_freq == 0:
                    self.writer.add_scalar(
                        "episode_timesteps",
                        episode_timesteps,
                        global_step=self.total_steps,
                    )
                    self.writer.add_scalar(
                        "reward/train", episode_reward, global_step=self.total_steps
                    )

                # Reset Environment
                state, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if (t + 1) % self.eval_freq == 0:
                avg_reward, best_reward = self.eval()
                self.writer.add_scalar(
                    "reward/val_avg", avg_reward, global_step=self.total_steps
                )
                self.writer.add_scalar(
                    "reward/val_best", best_reward, global_step=self.total_steps
                )
                self.save()

    def test(self, num_episodes=10):
        avg_reward, best_reward = self.eval(num_episodes=num_episodes)

        if self.writer is not None:
            self.writer.add_scalar("reward/test_avg", avg_reward, global_step=0)
            self.writer.add_scalar("reward/test_best", best_reward, global_step=0)

            self.writer.add_hparams(
                self.hparams,
                {
                    "hp_metric": avg_reward,
                },
                run_name=f"test",
            )

    def load(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir)

        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
        self.total_steps = checkpoint["total_steps"]

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def save(self):
        torch.save(
            {
                "total_steps": self.total_steps,
                "critic_state_dict": self.critic.state_dict(),
                "actor_state_dict": self.actor.state_dict(),
                "critic_opt_state_dict": self.critic_opt.state_dict(),
                "actor_opt_state_dict": self.actor_opt.state_dict(),
            },
            os.path.join(self.checkpoint_dir, "checkpoint.pt"),
        )
