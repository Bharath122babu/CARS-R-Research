from pathlib import Path
import torch

def pytest_configure(config):
    torch.set_num_threads(1)
    # A fresh checkout has no ignored runs/ directory yet. Pytest creates its
    # basetemp itself, but expects that directory's parent to already exist.
    if config.option.basetemp:
        Path(config.option.basetemp).resolve().parent.mkdir(parents=True, exist_ok=True)
