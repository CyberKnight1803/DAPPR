import copy
import os
import uuid

import numpy as np
import torch

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from dapr.agents.agent import GenericAgent

from dapr.models import Actor, Critic
from dapr.utils import ReplayBuffer, he_init_weights, xavier_init_weights


class DAPR(GenericAgent):
    def __init__(
        self,
        id: uuid.UUID,
        env_name: str,
        env_seed: int = 42,
        device: torch.device = torch.device("cpu"),
        writer: SummaryWriter = None,
        initialization: str = "he",
        batch_size: int = 100,
        discount: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        advsry_lr: float = 1e-3,
        actor_weight_decay: float = 1e-3,
        critic_weight_decay: float = 1e-3,
        advsry_weight_decay: float = 1e-3,
        start_step: int = int(1e4),
        buffer_size: int = int(1e6),
        expl_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 5,
        eval_freq: int = int(5e3),
        max_steps: int = int(1e6),
        log_freq: int = 10,
    ):
        super().__init__(
            env_name, env_seed=env_seed, id=id, device=device, writer=writer
        )

        # Hyperparams
        self.hparams = {}
        self.hparams["init_weight"] = initialization
        self.hparams["batch_size"] = batch_size
        self.hparams["discount"] = discount
        self.hparams["tau"] = tau
        self.hparams["actor_lr"] = actor_lr
        self.hparams["critic_lr"] = critic_lr
        self.hparams["advsry_lr"] = advsry_lr
        self.hparams["actor_weight_decay"] = actor_weight_decay
        self.hparams["critic_weight_decay"] = critic_weight_decay
        self.hparams["advsry_weight_decay"] = advsry_weight_decay
        self.hparams["start_step"] = start_step
        self.hparams["expl_noise"] = expl_noise
        self.hparams["max_steps"] = max_steps
        self.hparams["policy_noise"] = policy_noise
        self.hparams["noise_clip"] = noise_clip
        self.hparams["policy_freq"] = policy_freq

        self.log_freq = log_freq
        self.eval_freq = eval_freq

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, buffer_size)

        # Models

        if initialization == "xavier" or initialization == "golort":
            self.init_weights = xavier_init_weights
        else:
            self.init_weights = he_init_weights

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor.apply(self.init_weights)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic.apply(self.init_weights)
        self.critic_target = copy.deepcopy(self.critic)

        self.advsry = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.advsry.apply(self.init_weights)

        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.hparams["actor_lr"],
            weight_decay=self.hparams["actor_weight_decay"],
        )

        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.hparams["critic_lr"],
            weight_decay=self.hparams["critic_weight_decay"],
        )

        self.advsry_opt = torch.optim.AdamW(
            self.advsry.parameters(),
            lr=self.hparams["advsry_lr"],
            weight_decay=self.hparams["advsry_weight_decay"],
        )

    def predict(self, state) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

