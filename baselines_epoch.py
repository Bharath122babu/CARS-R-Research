"""Sequential one-epoch baselines matched to the completed CARS experiment.

Run from CMD: python -u baselines_epoch.py --epochs 1
Every model starts fresh, uses the original epoch sampler and recipe, and
resumes its own checkpoints. No historical CARS output is modified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import sys
import time

import torch

import cars_fast as fast
from cars_epoch import EpochStream, epoch_loss
from cars_r.checkpoint import read_checkpoint_payload
from cars_r.compare import model_configs, run_logged
from cars_r.config import ModelConfig
from cars_r.data import TokenStream

ROOT = Path(__file__).resolve().parent
DEFAULT_CARS = ROOT.parent / 'experiments/claude-fast-branch/runs/fast/epoch1_seed1337'


def make_plan(architecture, seed, source):
    cfg = {**fast.ORIGINAL_TRAIN_CONFIG, 'seed': seed}
    steps = math.ceil((len(source.tokens) - 1) / (cfg['batch_size'] * cfg['seq_len']))
    cfg['max_steps'] = steps
    models = model_configs('laptop')
    if models['cars'].to_dict() != ModelConfig.from_json(fast._model_config_path()).to_dict():
        raise ValueError('Laptop model recipe differs from the original CARS recipe')
    model = models[architecture].to_dict()
    return {
        'architecture': architecture, 'epochs': 1, 'fresh_training': True,
        'seed': seed, 'steps': steps, 'actual_prediction_tokens': len(source.tokens) - 1,
        'train_config': cfg, 'model_config': model, 'sampler_seed': seed + 731,
        'sampler': 'original EpochStream: shuffled disjoint chunks; padded tail ignored',
        'runner_sha256': fast.sha256_file(Path(__file__)),
        'epoch_sampler_sha256': fast.sha256_file(ROOT / 'cars_epoch.py'),
        'execution_identity': fast.execution_identity(),
    }


def verify_cars_comparison(cars_run, plan):
    """Check corpus, schedule, sampler and reference arithmetic before reuse."""
    contract = json.loads((cars_run / 'epoch_contract.json').read_text(encoding='utf-8'))
    for key in ('train_config', 'steps', 'actual_prediction_tokens', 'sampler_seed'):
        if contract[key] != plan[key]:
            raise ValueError(f'Completed CARS experiment differs: {key}')
    previous = contract['execution_identity']
    current = plan['execution_identity']
    required_data = [f'{name}{suffix}' for name in ('train', 'validation', 'wiki_test')
                     for suffix in ('.bin', '.bin.json')]
    required_data += [f'relations_{split}.jsonl' for split in ('train', 'validation', 'test')]
    if any(previous['dataset'].get(name) != current['dataset'].get(name)
           or name not in current['dataset'] for name in required_data) or previous['source'] != current['source']:
        raise ValueError('Completed CARS data or model/trainer source differs')
    final = read_checkpoint_payload(cars_run / 'final.pt')
    if final['step'] != plan['steps'] or not (cars_run / 'fast_eval.json').exists():
        raise ValueError('CARS reference must have completed training and evaluation')
    original_sampler = cars_run.parents[2] / 'cars_epoch.py'
    def sampler_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        return [ast.dump(node, include_attributes=False) for node in tree.body
                if getattr(node, 'name', None) in ('EpochStream', 'epoch_loss')]
    if (not original_sampler.exists()
            or fast.sha256_file(original_sampler) != contract['epoch_runner_sha256']
            or sampler_ast(original_sampler) != sampler_ast(ROOT / 'cars_epoch.py')):
        raise ValueError('Original and current epoch samplers differ')


def train_one(out, plan, stop_after=None):
    import cars_r.train as trainer
    cfg = plan['train_config']
    resume = out / 'latest.pt'
    completed = int(read_checkpoint_payload(resume)['step']) if resume.exists() else 0
    (out / 'model_config.json').write_text(json.dumps(plan['model_config'], indent=2), encoding='utf-8')
    (out / 'train_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    original_load, original_identity = trainer.load_sources, trainer.source_identity
    original_loss, original_argv = trainer.causal_lm_loss, sys.argv

    def load_sources(cli, model_cfg, train_cfg):
        train, val = original_load(cli, model_cfg, train_cfg)
        return EpochStream(train.tokens, batch_size=cfg['batch_size'], context=cfg['seq_len'],
                           seed=plan['sampler_seed'], completed_steps=completed,
                           progress=out / 'epoch_progress.jsonl'), val

    def source_identity(cli, train_cfg):
        return {**original_identity(cli, train_cfg), 'epoch_contract': plan}

    trainer.load_sources, trainer.source_identity = load_sources, source_identity
    trainer.causal_lm_loss = epoch_loss
    sys.argv = ['cars_r.train', '--model-config', str(out / 'model_config.json'),
                '--train-config', str(out / 'train_config.json'),
                '--bin', str(fast.DATA / 'train.bin'), '--val-bin', str(fast.DATA / 'validation.bin'),
                '--out-dir', str(out), '--device', 'cuda']
    if resume.exists():
        sys.argv += ['--resume', str(resume)]
    if stop_after is not None:
        sys.argv += ['--stop-after', str(stop_after)]
    start = time.perf_counter()
    try:
        trainer.main()
    finally:
        trainer.load_sources, trainer.source_identity = original_load, original_identity
        trainer.causal_lm_loss, sys.argv = original_loss, original_argv
        with (out / 'epoch_invocations.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({'wall_seconds': time.perf_counter() - start}) + '\n')


def write_comparison(cars_run, seed):
    reports = [('cars', cars_run / 'fast_eval.json')]
    reports += [(a, fast.FAST_RUNS / f'{a}_epoch1_seed{seed}' / 'fast_eval.json')
                for a in ('kda', 'gdn2', 'transformer')]
    rows, suite_hash = [], None
    for name, path in reports:
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding='utf-8'))
        identity = (result['provenance']['suites_sha256'], result['provenance']['wiki_blocks'])
        if suite_hash is not None and identity != suite_hash:
            raise ValueError('Evaluation cases/windows differ between models')
        suite_hash = identity
        a = result['accuracy']
        rows.append({'model': name, 'wiki_nll': result['wiki_nll'],
                     'iid': a['iid_original_500'], 'fresh_iid': a['iid_fresh'],
                     'five_transfers': a['five_transfers'],
                     'two_objects': a['two_objects_distractor_last'],
                     'unchanged_object': a['two_objects_unchanged_query'],
                     'training_minutes': result['training_minutes'],
                     'selected_step': result['selected_step'], 'report': str(path)})
    prefix = fast.FAST_RUNS / f'epoch_comparison_seed{seed}'
    prefix.with_suffix('.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# One-epoch comparison', '',
             'Same data, sampler seed, token budget, optimizer and validation selection. '
             'Laptop recipes are approximately parameter matched, not FLOP matched. '
             'KDA/GDN2 are portable adaptations; timings do not represent upstream kernels.', '',
             '| Model | Wiki byte NLL | IID | Fresh IID | 5 transfers | Two objects | Unchanged | Training minutes |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['wiki_nll']:.5f} | {row['iid']:.1%} | "
                     f"{row['fresh_iid']:.1%} | {row['five_transfers']:.1%} | "
                     f"{row['two_objects']:.1%} | {row['unchanged_object']:.1%} | {row['training_minutes']:.2f} |")
    lines += ['', 'Training minutes cover logged training steps; evaluation/export time is additional. '
              'Single-object IID cases admit the latest-recipient shortcut. One seed is preliminary.']
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'COMPARISON: {prefix.with_suffix(".md")}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, choices=(1,), default=1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--architectures', nargs='+', choices=('kda', 'gdn2', 'transformer'),
                        default=['kda', 'gdn2', 'transformer'])
    parser.add_argument('--cars-run', type=Path, default=DEFAULT_CARS)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Safe short execution check per model')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output-prefix', default='', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.architectures)) != len(args.architectures):
        raise ValueError('Architectures must be unique')
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError('--stop-after must be positive')
    if args.output_prefix and not args.stop_after:
        raise ValueError('Alternate output prefix is only for short execution checks')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this experiment')
    from cars_r.comparison_data import prepare_dataset
    prepare_dataset(fast.DATA)
    source = TokenStream.from_bin(fast.DATA / 'train.bin', split='all')
    for architecture in args.architectures:
        plan = make_plan(architecture, args.seed, source)
        verify_cars_comparison(args.cars_run.resolve(), plan)
        out = fast.require_experiment_output(fast.FAST_RUNS / f'{args.output_prefix}{architecture}_epoch1_seed{args.seed}')
        contract = out / 'epoch_contract.json'
        if contract.exists() and json.loads(contract.read_text(encoding='utf-8')) != plan:
            raise ValueError(f'Run code/config/data changed: {out}; use a separate experiment')
        if not contract.exists() and out.exists() and (list(out.glob('*.pt')) or (out / 'metrics.jsonl').exists()):
            raise ValueError(f'Existing run lacks an epoch contract: {out}')
        print(json.dumps({'epoch_plan': plan, 'output': str(out)}), flush=True)
        if args.plan_only:
            continue
        out.mkdir(parents=True, exist_ok=True)
        if not contract.exists():
            contract.write_text(json.dumps(plan, indent=2), encoding='utf-8')
        if not args.worker:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                       '--architectures', architecture, '--seed', str(args.seed),
                       '--cars-run', str(args.cars_run.resolve())]
            if args.stop_after:
                command += ['--stop-after', str(args.stop_after)]
            if args.output_prefix:
                command += ['--output-prefix', args.output_prefix]
            run_logged(command, out / 'console.log')
            if not args.stop_after:
                write_comparison(args.cars_run.resolve(), args.seed)
            continue
        if not (out / 'final.pt').exists():
            print(f'TRAINING: {architecture}, {plan["steps"]} steps; same command resumes', flush=True)
            train_one(out, plan, args.stop_after)
        if not (out / 'final.pt').exists():
            print('Stopped safely; checkpoint saved. Rerun the same command to resume.', flush=True)
            continue
        if read_checkpoint_payload(out / 'final.pt')['step'] != plan['steps']:
            raise ValueError('Final checkpoint does not match the complete epoch')
        print(f'EVALUATING: {architecture}, validation-selected best.pt', flush=True)
        fast.cmd_evaluate(argparse.Namespace(run=str(out), windows=1024, data_dir=fast.DATA, reference=True))
        from cars_r.export_model import export_model
        print(json.dumps(export_model(out / 'best.pt', out / 'export'), indent=2), flush=True)
        print(f'COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    main()
