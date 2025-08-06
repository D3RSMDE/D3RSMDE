import argparse
import random
import os
import numpy as np
import torch
import random


def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Random seed initialized to {seed}. This ensures reproducibility across runs.")


def seed_worker(worker_id):
    """
    Sets the random seed for a DataLoader worker process.
    Ensures that data loading is deterministic when using multiple workers.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    pass
