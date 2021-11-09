import torch
import torch.nn as nn
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything


class DDPG(pl.LightningModule):
    pass


if __name__ == '__main__':
    seed_everything(42)