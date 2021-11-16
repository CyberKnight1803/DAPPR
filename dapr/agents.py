import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from dapr.models import Actor, Critic
from dapr.utils import ReplayBuffer


class DDPG(pl.LightningModule):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        batch_size: int = 64,
        discount: float = 0.99,
        tau: float = 0.001,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_weight_decay: float = 1e-3,
        critic_weight_decay: float = 1e-3,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_weight_decay = actor_weight_decay
        self.critic_weight_decay = critic_weight_decay

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # Turn off automatic optimization. Multiple models must be optimized simultaneously
        self.automatic_optimization = False

    def forward(self, state):
        action = self.actor(state)
        Q = self.critic(state, action)

        return action, Q

    def training_step(self, replay_buffer: ReplayBuffer, batch_idx):
        actor_opt, critic_opt = self.optimizers()

        # Reorder in replay buffer to be s, a, r, s', d
        # Sample the Replay Buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            self.batch_size, device=self.device()
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
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Compute the actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update the Actor by one-step gradient descent (not ascent as in paper, since we multiplied by -1)
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # Update the frozen target models
        for param, tgt_param in zip(
            self.critic.parameters, self.critic_target.parameters()
        ):
            tgt_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * tgt_param.data
            )

        for param, tgt_param in zip(
            self.actor.parameters, self.actor_target.parameters()
        ):
            tgt_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * tgt_param.data
            )

        self.log("actor_loss", actor_loss)
        self.log("critic_loss", critic_loss)

    def configure_optimizers(self):
        actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.actor_lr,
            weight_decay=self.actor_weight_decay,
        )

        critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=self.critic_weight_decay,
        )

        return actor_opt, critic_opt
