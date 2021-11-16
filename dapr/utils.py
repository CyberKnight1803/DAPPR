from os import stat
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.state[idxs], device=device, dtype=torch.float),
            torch.tensor(self.action[idxs], device=device, dtype=torch.float),
            torch.tensor(self.next_state[idxs], device=device, dtype=torch.float),
            torch.tensor(self.reward[idxs], device=device, dtype=torch.float),
            torch.tensor(self.not_done[idxs], device=device, dtype=torch.float),
        )
