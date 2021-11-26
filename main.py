import uuid

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything


from dapr.agents.ddpg import DDPG
from dapr.constants import DEVICE


def main():
    seed_everything(42)
    env_name = "HalfCheetah-v2"
    model = "DAPR"
    id = uuid.uuid4()

    with SummaryWriter(log_dir=f"trial_runs/{env_name}/{model}-{id}") as writer:
        print(f"Logging in {writer.get_logdir()}")
        agent = DDPG(
            env_name=env_name,
            id=id,
            device=DEVICE,
            writer=writer,
            critic_weight_decay=5e-4,
            actor_weight_decay=5e-4,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
        )
        agent.train()
        agent.test(100)


if __name__ == "__main__":
    main()
