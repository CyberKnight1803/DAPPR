import time
import uuid

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything


from dapr.agents import DDPG
from dapr.constants import DEVICE


def main():
    seed_everything(42)
    env_name = "Walker2d-v2"
    id = uuid.uuid4()

    with SummaryWriter(log_dir=f"trial_runs/{env_name}/DDPG-{id}") as writer:
        print(f"Logging in {writer.get_logdir()}")
        agent = DDPG(
            env_name=env_name,
            id=id,
            device=DEVICE,
            writer=writer,
            critic_weight_decay=0,
            actor_weight_decay=0,
        )
        agent.train()
        agent.test(100)


if __name__ == "__main__":
    main()
