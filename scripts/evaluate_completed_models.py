"""Read-only post-training comparison of the four validation-selected exports.

Expanded held-out byte-token scoring, ownership stress cases, and compiled
reload checks. Writes new evidence; never changes weights or training results.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cars_r.checkpoint import load_checkpoint
from cars_r.compare import source_sha256
from cars_r.comparison_data import PEOPLE, OBJECTS, relational_case, sha256
from cars_r.data import TokenStream
from cars_r.tokenizer import ByteTokenizer
from cars_r.train import autocast_context

MODELS = ('cars', 'transformer', 'kda', 'gdn2')


def stress_cases(seed, count, unchanged=False):
    rng = random.Random(seed)
    result = []
    seen = set()
    while len(result) < count:
        x, y = rng.sample(OBJECTS, 2)
        owner_x, owner_y = rng.sample(PEOPLE, 2)
        initial = [f'{owner_x} owns {x}.', f'{owner_y} owns {y}.']
        rng.shuffle(initial)
        lines = initial
        if not unchanged:
            for _ in range(2):
                following = rng.choice([p for p in PEOPLE if p != owner_x])
                lines.append(f'{owner_x} gives {x} to {following}.')
                owner_x = following
        for _ in range(2 if unchanged else 1):
            following = rng.choice([p for p in PEOPLE if p not in (owner_x, owner_y)])
            lines.append(f'{owner_y} gives {y} to {following}.')
            owner_y = following
        prompt = ' '.join(lines) + f' Owner of {x}:'
        if prompt in seen:
            continue
        seen.add(prompt)
        assert owner_x != owner_y  # The global last-recipient shortcut is wrong.
        # Independently replay the literal events to check all targets.
        state = {}
        for line in lines:
            parts = line.rstrip('.').split()
            if parts[1] == 'owns':
                state[parts[2]] = parts[0]
            else:
                assert state[parts[2]] == parts[0]
                state[parts[2]] = parts[4]
        assert state[x] == owner_x
        result.append({'prompt': prompt, 'target': ' ' + owner_x,
            'last_recipient': owner_y, 'queried_object': x})
    return result


@torch.inference_mode()
def score_cases(model, cases, device, batch_size):
    tok = ByteTokenizer()
    correct, exact, nlls, predictions = 0, 0, [], []
    for start in range(0, len(cases), batch_size):
        examples = cases[start:start + batch_size]
        prompts = [tok.encode_prompt(case['prompt']) for case in examples]
        active = [p + [ord(' ')] for p in prompts]
        length = max(map(len, active))
        assert length <= model.config.max_seq_len
        ids = torch.full((len(active), length), tok.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, tokens in enumerate(active):
            ids[row, :len(tokens)] = torch.tensor(tokens, device=device)
            mask[row, :len(tokens)] = True
        with autocast_context(device, 'bf16' if device.type == 'cuda' else 'fp32'):
            logits = model(ids, attention_mask=mask).logits.float()
        rows = torch.arange(len(active), device=device)
        last = torch.tensor([len(a)-1 for a in active], device=device)
        owner_logits = logits[rows, last]
        labels = torch.tensor([ord(case['target'].strip()) for case in examples], device=device)
        choices = owner_logits[:, [ord(person) for person in PEOPLE]].argmax(-1).tolist()
        greedy_owner = owner_logits.argmax(-1)
        greedy_space = logits[rows, last-1].argmax(-1) == ord(' ')
        exact += int(((greedy_owner == labels) & greedy_space).sum())
        nlls.extend(F.cross_entropy(owner_logits, labels, reduction='none').tolist())
        for case, choice in zip(examples, choices):
            predicted = PEOPLE[choice]
            correct += predicted == case['target'].strip()
            predictions.append(predicted)
    return {'cases': len(cases), 'candidate_correct': correct,
        'candidate_accuracy': correct/len(cases),
        'unrestricted_greedy_space_and_owner_exact': exact/len(cases),
        'mean_owner_token_nll': float(np.mean(nlls)), 'predictions': predictions,
        'max_input_tokens': max(len(tok.encode_prompt(c['prompt']))+1 for c in cases)}


@torch.inference_mode()
def generate_greedy(model, prompt, device, tokens=48):
    tok = ByteTokenizer()
    ids = tok.encode_prompt(prompt)
    original = len(ids)
    for _ in range(tokens):
        with autocast_context(device, 'bf16' if device.type == 'cuda' else 'fp32'):
            logits = model(torch.tensor([ids], device=device)).logits[0, -1].float()
        logits[tok.pad_token_id] = logits[tok.bos_token_id] = -float('inf')
        following = int(logits.argmax())
        ids.append(following)
        if following == tok.eos_token_id:
            break
    return {'prompt': prompt, 'continuation': tok.decode(ids[original:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/comparison_laptop_delta_seed1337')
    parser.add_argument('--out', default='runs/post_training_test_seed20261005')
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--windows', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = ROOT / 'data/comparison_wikitext2'
    plan = json.loads((run / 'experiment.json').read_text())
    for name, expected in plan['dataset_sha256'].items():
        assert sha256(data / name) == expected, name
    rows = {row['architecture']: row for row in csv.DictReader((run / 'comparison.csv').open())}
    source = TokenStream.from_bin(data / 'wiki_test.bin', split='all').tokens
    context = 128
    # Random unique aligned blocks; adjacent blocks have disjoint scored targets.
    blocks = np.random.default_rng(args.seed).choice((len(source)-1)//context, args.windows, replace=False)
    starts = (blocks * context).tolist()
    wiki = np.stack([source[s:s+context+1] for s in starts]).astype(np.int64)
    iid = [json.loads(line) for line in (data / 'relations_test.jsonl').read_text().splitlines()]
    rng = random.Random(args.seed)
    longer = [dict(zip(('prompt', 'target'), relational_case(rng, 5))) for _ in range(256)]
    suites = {'ownership_iid_all': iid, 'ownership_5_transfers': longer,
        'two_objects_distractor_last': stress_cases(args.seed+1, 256),
        'two_objects_unchanged_query': stress_cases(args.seed+2, 256, unchanged=True)}
    training_prompts = {json.loads(line)['prompt'] for line in (data / 'relations_train.jsonl').read_text().splitlines()}
    assert all(case['prompt'] not in training_prompts for cases in suites.values() for case in cases)
    assert all(len(ByteTokenizer().encode_prompt(c['prompt']))+1 <= context for name, cases in suites.items()
        if name.startswith('two_objects') for c in cases)  # Stress cases stay inside training context.
    (out / 'cases.json').write_text(json.dumps(suites, indent=2)+'\n')
    result = {'seed': args.seed, 'device': str(device), 'source_sha256': source_sha256(),
        'evaluator_sha256': sha256(Path(__file__)), 'run': str(run),
        'wiki_test': {'windows': args.windows, 'context': context, 'tokens_scored': args.windows*context,
            'window_starts': starts, 'sha256': sha256(data/'wiki_test.bin')},
        'case_suite_sha256': sha256(out/'cases.json'),
        'candidate_chance_accuracy': 1/8,
        'notes': ['Validation-selected checkpoints; no retraining or test-based checkpoint selection.',
            'Two-object suites are out of the training task distribution, with familiar syntax and <=128 tokens.',
            'Both two-object suites force global-last-recipient accuracy to 0%; object-aware rule accuracy is 100%.',
            'Bootstrap intervals resample windows for this fixed training seed; not uncertainty across training seeds.',
            'Byte/special-token NLL is not word perplexity; language generation from these tiny models is a qualitative check.'],
        'models': {}}
    for name in MODELS:
        print(f'TESTING {name}: expanded WikiText and four ownership suites', flush=True)
        bundle = run / 'exports' / name
        model, payload = load_checkpoint(bundle / 'model.pt', device=device)
        assert model.config.architecture == name
        assert model.inference_num_parameters() == int(rows[name]['parameters'])
        assert payload['step'] == int(rows[name]['selected_best_step'])
        original_dir = Path(plan['reused_completed_runs'][name]['directory']) if name in plan.get('reused_completed_runs', {}) else run
        assert payload['extra']['source_checkpoint_sha256'] == sha256(original_dir/name/'best.pt')
        model.eval()
        losses = []
        beginning = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(wiki), args.batch_size):
                batch = torch.tensor(wiki[start:start+args.batch_size], device=device)
                with autocast_context(device, 'bf16' if device.type == 'cuda' else 'fp32'):
                    logits = model(batch[:, :-1]).logits.float()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction='none')
                losses.extend(loss.reshape(len(batch), context).mean(-1).tolist())
                if (start//args.batch_size) % 16 == 15:
                    print(f'  {name}: {min(start+args.batch_size,len(wiki))}/{len(wiki)} Wiki windows', flush=True)
        entry = {'export_weights_sha256': sha256(bundle/'model.pt'),
            'selected_step': payload['step'], 'wiki_mean_token_nll': float(np.mean(losses)),
            'wiki_window_nll': losses, 'wiki_evaluation_seconds': time.perf_counter()-beginning,
            'original_training_seconds': float(rows[name]['training_seconds']),
            'original_peak_cuda_memory_mb': float(rows[name]['peak_cuda_memory_mb']),
            'ownership': {}}
        for label, cases in suites.items():
            entry['ownership'][label] = score_cases(model, cases, device, args.batch_size)
            print(f"  {name}: {label} {entry['ownership'][label]['candidate_accuracy']:.2%}", flush=True)
        entry['generation'] = [generate_greedy(model, prompt, device) for prompt in
            ('The history of science', 'A owns x. A gives x to B. Owner of x:',
             'A owns x. B owns y. A gives x to C. B gives y to D. Owner of x:')]
        del payload
        model = model.to('cpu')
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            compiled = torch.jit.load(str(bundle/'compiled_cpu.pt'), map_location='cpu').eval()
            maximum = 0.0
            tok = ByteTokenizer()
            with torch.inference_mode():
                for prompt in ('New test.', suites['two_objects_distractor_last'][0]['prompt']+' ', 'A'*127):
                    tokens = tok.encode_prompt(prompt)
                    assert len(tokens) <= context
                    ids = torch.full((1, context), tok.pad_token_id, dtype=torch.long)
                    mask = torch.zeros_like(ids, dtype=torch.bool)
                    ids[0,:len(tokens)] = torch.tensor(tokens)
                    mask[0,:len(tokens)] = True
                    eager = model(ids, attention_mask=mask).logits
                    traced = compiled(ids, mask)
                    torch.testing.assert_close(traced, eager, rtol=3e-4, atol=3e-5)
                    maximum = max(maximum, float((traced-eager).abs().max()))
            entry['compiled_reload'] = {'status': 'PASS', 'new_prefixes': 3,
                'maximum_absolute_logit_difference': maximum,
                'sha256': sha256(bundle/'compiled_cpu.pt')}
            del compiled
        finally:
            torch.set_num_threads(threads)
        result['models'][name] = entry
        (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    # Paired resampling: the identical windows are compared for every model pair.
    result['paired_wiki_differences'] = []
    random_bootstrap = np.random.default_rng(args.seed+3)
    draws = random_bootstrap.integers(0, args.windows, size=(2000, args.windows))
    for index, a in enumerate(MODELS):
        for b in MODELS[index+1:]:
            difference = np.array(result['models'][a]['wiki_window_nll'])-np.array(result['models'][b]['wiki_window_nll'])
            low, high = np.quantile(difference[draws].mean(axis=1), [.025,.975])
            result['paired_wiki_differences'].append({'a': a, 'b': b,
                'mean_nll_a_minus_b': float(difference.mean()),
                'window_bootstrap_95_percent_interval': [float(low),float(high)]})
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Expanded post-training tests', '',
        f'Identical {args.windows:,} held-out WikiText windows ({args.windows*context:,} byte/special tokens), all 500 ownership test cases, 256 new five-transfer cases, and 256 cases per two-object stress suite. No retraining.', '',
        '| Model | Wiki NLL ↓ | IID | 5 transfers | Two objects, distractor last | Unchanged queried object | Training minutes | Peak training MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, entry in result['models'].items():
        accuracy = [entry['ownership'][label]['candidate_accuracy'] for label in suites]
        lines.append(f"| {name} | {entry['wiki_mean_token_nll']:.5f} | "+' | '.join(f'{value:.1%}' for value in accuracy)+
            f" | {entry['original_training_seconds']/60:.2f} | {entry['original_peak_cuda_memory_mb']:.1f} |")
    lines += ['', 'Ownership columns score the best of eight owner letters after an answer-space token (chance 12.5%). Unrestricted greedy answer accuracy is also saved in results.json.', '',
        'The two-object tests use familiar words and stay within 128 tokens, but their task structure was absent from training. Global-last-recipient copying gets 0%; an object-aware symbolic rule gets 100%. Failure diagnoses a generalization gap, not an architecture capacity limit.', '',
        '## Paired WikiText differences', '', 'Negative A−B favors A. Intervals resample windows for this one trained seed; they do not establish significance across training seeds.', '',
        '| A | B | Mean NLL A−B | Window bootstrap 95% interval |', '|---|---|---:|---|']
    for pair in result['paired_wiki_differences']:
        low, high = pair['window_bootstrap_95_percent_interval']
        lines.append(f"| {pair['a']} | {pair['b']} | {pair['mean_nll_a_minus_b']:+.5f} | [{low:+.5f}, {high:+.5f}] |")
    lines += ['', '## Compiled models', '', 'All four saved CPU TorchScript exports reload and match their eager FP32 models on three new prefixes, including a full 128-token input.', '',
        '## Greedy generation samples', '']
    for name, entry in result['models'].items():
        lines += [f'### {name}', '']
        for sample in entry['generation']:
            lines += [f"Prompt: {sample['prompt']}", '', '```text', sample['continuation'], '```', '']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'COMPLETE: {out / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
