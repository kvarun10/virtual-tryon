"""Seeding helpers for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    ``deterministic=True`` additionally asks PyTorch for deterministic kernels. Some CUDA ops
    (e.g. certain interpolations' backward) have no deterministic version and will then raise;
    this is why it is off by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive NumPy/random seeds from the torch worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
