"""One complete shuffled pass through the original corpus, with exact resume.

Opt-in wrapper around the unchanged trainer and qualified fast CARS operator.
The original random-window sampler is not an epoch sampler: it can revisit
some spans and miss others. Here every next-token target is scored once.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import cars_fast as fast
from cars_r.data import TokenStream
from cars_r.losses import LossOutput


class EpochStream:
    def __init__(self, tokens, *, batch_size, context, seed, completed_steps=0, progress=None):
        self.tokens = tokens
        self.batch_size, self.context, self.seed = batch_size, context, seed
        self.blocks = math.ceil((len(tokens) - 1) / context)
        self.steps = math.ceil(self.blocks / batch_size)
        self.completed_steps = completed_steps
        self.progress = progress
        self.order = torch.randperm(self.blocks, generator=torch.Generator().manual_seed(seed)).numpy()

    def sample(self, batch_size, length, *, generator, device):
        if batch_size != self.batch_size or length != self.context + 1:
            raise ValueError('Epoch batch/context changed')
        if self.completed_steps >= self.steps:
            raise RuntimeError('The complete epoch has already been consumed')
        begin = self.completed_steps * batch_size
        chosen = self.order[begin:begin + batch_size]
        batch = np.full((batch_size, length), 256, dtype=np.int64)
        valid_targets = 0
        for row, block in enumerate(chosen):
            start = int(block) * self.context
            values = self.tokens[start:min(start + length, len(self.tokens))]
            batch[row, :len(values)] = values
            valid_targets += len(values) - 1
        self.completed_steps += 1
        # Includes each target exactly once, including the last partial block.
        consumed = self.order[:min(self.completed_steps * batch_size, self.blocks)]
        targets_seen = int(np.minimum(self.context, len(self.tokens)-1-consumed*self.context).sum())
        entry = {'step': self.completed_steps, 'epoch': 1,
                 'epoch_fraction': targets_seen / (len(self.tokens)-1),
                 'actual_prediction_tokens': targets_seen,
                 'actual_tokens_this_batch': valid_targets,
                 'note': 'Batch sampled; completed optimizer updates are recorded in metrics/checkpoints.'}
        if self.progress:
            with Path(self.progress).open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(entry)+'\n')
        if self.completed_steps % 100 == 0 or self.completed_steps == self.steps:
            print(json.dumps({'epoch_progress': entry}), flush=True)
        return torch.from_numpy(batch).to(device=device)


def epoch_loss(output, targets, *, mtp_weight=0.0):
    if mtp_weight or output.mtp_logits:
        raise ValueError('This epoch experiment supports the original AR-only model')
    # PAD appears only in the last partial chunk/batch. Causal attention means
    # appended PAD cannot affect predictions of the preceding real tokens.
    ar = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                         targets.reshape(-1), ignore_index=256)
    return LossOutput(total=ar, autoregressive=ar, mtp=ar.new_zeros(()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, choices=(1,), default=1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this laptop experiment')
    from cars_r.comparison_data import prepare_dataset
    prepare_dataset(fast.DATA)
    source = TokenStream.from_bin(fast.DATA / 'train.bin', split='all')
    cfg = dict(fast.ORIGINAL_TRAIN_CONFIG)
    cfg['seed'] = args.seed
    steps = math.ceil((len(source.tokens)-1) / (cfg['batch_size']*cfg['seq_len']))
    cfg['max_steps'] = steps
    out = fast.require_experiment_output(fast.FAST_RUNS / f'epoch1_seed{args.seed}')
    identity = fast.execution_identity()
    plan = {'epochs': 1, 'fresh_training': True, 'seed': args.seed, 'steps': steps,
            'actual_prediction_tokens': len(source.tokens)-1, 'train_config': cfg,
            'sampler': 'shuffled disjoint 128-target chunks; shared boundary input; padded tail ignored',
            'sampler_seed': args.seed+731, 'epoch_runner_sha256': fast.sha256_file(Path(__file__)),
            'execution_identity': identity}
    print(json.dumps({'epoch_plan': plan}), flush=True)
    if args.plan_only:
        return
    qpath = fast.FAST_RUNS / 'QUALIFICATION.json'
    if not qpath.exists() or json.loads(qpath.read_text()).get('execution_identity') != identity or json.loads(qpath.read_text()).get('verdict') != 'PASS':
        print('Qualifying fast operator for the current machine...', flush=True)
        fast.cmd_qualify(argparse.Namespace(qualify_windows=128, timing_steps=5))
    contract = out / 'epoch_contract.json'
    if contract.exists():
        if json.loads(contract.read_text()) != plan:
            raise SystemExit('Epoch run code/config/data changed; choose a new seed/output experiment')
    elif out.exists() and (list(out.glob('*.pt')) or (out / 'metrics.jsonl').exists()):
        raise SystemExit('Existing output is missing its epoch contract')
    out.mkdir(parents=True, exist_ok=True)
    if not contract.exists():
        contract.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    if (out / 'final.pt').exists():
        print(f'Already trained: {out}', flush=True)
    else:
        import cars_r.train as trainer
        from cars_r.checkpoint import read_checkpoint_payload
        resume = out / 'latest.pt'
        completed = int(read_checkpoint_payload(resume, device='cpu')['step']) if resume.exists() else 0
        model_path, train_path = out / 'model_config.json', out / 'train_config.json'
        model_path.write_bytes(fast._model_config_path().read_bytes())
        train_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        original_load, original_identity = trainer.load_sources, trainer.source_identity
        def load_sources(cli, model_cfg, train_cfg):
            train, val = original_load(cli, model_cfg, train_cfg)
            return EpochStream(train.tokens, batch_size=cfg['batch_size'], context=cfg['seq_len'],
                               seed=args.seed+731, completed_steps=completed,
                               progress=out/'epoch_progress.jsonl'), val
        trainer.load_sources = load_sources
        def source_identity(cli, train_cfg):
            return {**original_identity(cli, train_cfg), 'epoch_contract': plan}
        trainer.source_identity = source_identity
        trainer.causal_lm_loss = epoch_loss
        fast.enable_fast_path()
        sys.argv = ['cars_r.train', '--model-config', str(model_path), '--train-config', str(train_path),
                    '--bin', str(fast.DATA/'train.bin'), '--val-bin', str(fast.DATA/'validation.bin'),
                    '--out-dir', str(out), '--device', 'cuda']
        if resume.exists(): sys.argv += ['--resume', str(resume)]
        if args.stop_after: sys.argv += ['--stop-after', str(args.stop_after)]
        start = time.perf_counter()
        try:
            trainer.main()
        finally:
            fast.disable_fast_path()
            with (out/'epoch_invocations.jsonl').open('a', encoding='utf-8') as fh:
                fh.write(json.dumps({'wall_seconds':time.perf_counter()-start})+'\n')
    if not (out/'final.pt').exists():
        print('Stopped safely. Rerun the same command to resume.', flush=True)
        return
    print('EVALUATING: selected best checkpoint with original operator', flush=True)
    fast.disable_fast_path()
    fast.cmd_evaluate(argparse.Namespace(run=str(out), windows=1024, data_dir=fast.DATA, reference=True))
    from cars_r.export_model import export_model
    print(json.dumps(export_model(out/'best.pt', out/'export'), indent=2), flush=True)
    print(f'COMPLETE: {out / "fast_eval.json"}', flush=True)


if __name__ == '__main__':
    main()
