import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data.dataset import IterableDataset
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import utils


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,                 # Input state dimensions
        hidden_dims,         # Hidden layer dimensions
        action_dim: int,                # Input action dimensions
        max_action
    ) -> None:

        super(Actor, self).__init__()
        hidden_dims = [state_dim] + hidden_dims

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1])]
            layers += [nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], action_dim)]
        layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        out = self.net(state)
        return self.max_action * out


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims,
        action_dim: int
    ) -> None:
        super(Critic, self).__init__()

        hidden_dims = [state_dim + action_dim] + hidden_dims

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1])]
            layers += [nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        out = self.net(torch.cat([state, action], 1))
        return out


class DDPG(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims,
        action_dim: int,
        # Used to limit action in [-action_lim, action_lim]
        action_lim: int,
        ram,                        # Replay memory buffer
        lr: int,
        batch_size: int,
        gamma: float,
        tau: float
    ) -> None:

        super(DDPG, self).__init__()

        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.noise = utils.ActionNoise(self.action_dim)
        self.lr = lr 
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(self.state_dim, self.hidden_dims, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.hidden_dims, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(self.state_dim, self.hidden_dims, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.hidden_dims, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_exploration_action(self, state):
        state = Variable(torch.from_numpy(state))
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        
        return new_action

    def get_exploitation_action(self, state):
        state = Variable(torch.from_numpy(state))
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()
    
    def optimize(self, itr):

        s_1, a_1, r_1, s_2 = self.ram.sample(self.batch_size)

        s_1 = Variable(torch.from_numpy(s_1))
        a_1 = Variable(torch.from_numpy(a_1))
        r_1 = Variable(torch.from_numpy(r_1))
        s_2 = Variable(torch.from_numpy(s_2))


        # Critic optimization
        a_2 = self.target_actor.forward(s_2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s_2, a_2).detach())
        y_expected = r_1 + self.gamma * next_val
        y_predicted = torch.squeeze(self.critic.forward(s_1, a_1))

        critic_loss = F.smooth_l1_loss(y_predicted, y_expected)
        writer.add_scalar('Critic/loss', critic_loss, itr)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor optimization
        pred_a_1 = self.actor.forward(s_1)
        actor_loss = -1 * torch.sum(self.critic.forward(s_1, pred_a_1))

        writer.add_scalar('Actor/loss', actor_loss, itr)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        utils.soft_update(self.target_actor, self.actor, self.tau)
        utils.soft_update(self.target_critic, self.critic, self.tau)

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print('Models saved successfully')
    
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')

if __name__ == '__main__':
    seed_everything(42)
