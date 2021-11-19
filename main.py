import time
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything


from dapr.agents import DDPG
from dapr.constants import DEVICE


def main():
    seed_everything(42)
    writer = SummaryWriter(log_dir=f"trial_runs/{int(time.time())}")
    agent = DDPG(env_name="BipedalWalker-v3", device=DEVICE, writer=writer)
    agent.train()


if __name__ == "__main__":
    main()
