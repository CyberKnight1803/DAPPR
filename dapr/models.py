import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float) -> None:
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            # nn.Dropout(p=0.15),
            nn.Linear(state_dim, 400),
            # nn.BatchNorm1d(400),
            nn.ReLU(),
            # nn.Dropout(p=0.15),
            nn.Linear(400, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            # nn.Dropout(p=0.15),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state) -> torch.Tensor:
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            # nn.Dropout(p=0.15),
            nn.Linear(state_dim + action_dim, 400),
            # nn.BatchNorm1d(400),
            nn.ReLU(),
            # nn.Dropout(p=0.15),
            nn.Linear(400, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),
            # nn.Dropout(p=0.15),
            nn.Linear(300, 1),
        )

    def forward(self, state, action) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))
