import numpy as np
import torch 
import shutil

def soft_update(target, source, tau):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class ActionNoise:

    def __init__(
        self,
        action_dim: int,
        mu: int = 0,
        theta: float = 0.15, 
        sigma: float = 0.2
    ) -> None: 
        self.action_dim = action_dim 
        self.mu = mu 
        self.theta = theta
        self.sigma = sigma

        self.X = np.ones(self.action_dim) * self.mu 

    def sample(self): 
        dx = self.theta * (self.mu - self.X)
        dx += self.sigma * np.random.randn(len(self.X))
        self.X += dx 

        return self.X 

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu 


def save_training_checkpoint(state, is_best, episode_count):
	filename = str(episode_count) + 'checkpoint.path.rar'
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')
