"""Run the suite with an explicitly selected execution backend."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--execution', choices=('reference', 'fast'), default='reference')
parser.add_argument('--basetemp', default='runs/test-temp')
args = parser.parse_args()
import torch
import pytest
import cars_fast
torch.set_num_threads(1)

class Backend:
    def pytest_runtest_setup(self, item):
        (cars_fast.enable_fast_path if args.execution == 'fast' else cars_fast.disable_fast_path)()

raise SystemExit(pytest.main(['tests', '-q', '--basetemp', args.basetemp], plugins=[Backend()]))
