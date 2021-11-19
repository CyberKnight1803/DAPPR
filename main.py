import time
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything


from dapr.agents import DDPG
from dapr.constants import DEVICE


def main():
    seed_everything(42)
    env_name = "Walker2d-v2"
    writer = SummaryWriter(log_dir=f"trial_runs/{env_name}/{int(time.time())}")
    agent = DDPG(env_name=env_name, device=DEVICE, writer=writer)
    agent.train()


if __name__ == "__main__":
    main()
