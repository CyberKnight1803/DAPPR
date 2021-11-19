from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dapr.agents import DDPG
from dapr.constants import AVAIL_GPUS


def main():
    seed_everything(42)

    agent = DDPG(env="BipedalWalker-v3")
    trainer = Trainer(
        # gpus=AVAIL_GPUS,
        max_epochs=5,
        log_every_n_steps=1,
        # stochastic_weight_avg=True,
        logger=TensorBoardLogger("trial_runs/"),
    )
    trainer.fit(agent)


if __name__ == "__main__":
    main()
