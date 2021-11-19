import os

import torch


AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")