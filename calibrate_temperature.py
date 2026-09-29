"""Validation-only temperature sweep for a frozen checkpoint (E10)."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from common import ROOT, load_data, make_model, setup
from evaluate import score


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint


def eval_t(model, temperature, data_split, device):
    model.temperature = float(temperature)
    return score(model, *data_split, device, 'fp32')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/e10-temp-sweep')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--fine-center', type=float, default=None)
    args = p.parse_args()
    device, _ = setup(args.device, 'fp32', args.threads)
    data = load_data()
    val = data['validation']
    model, checkpoint = load_model(args.checkpoint, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.fine_center is None:
        grid = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
        tag = 'coarse'
    else:
        c = args.fine_center
        grid = [round(c + d, 2) for d in (-0.04, -0.02, 0.0, 0.02, 0.04)]
        tag = 'fine'

    started = time.perf_counter()
    rows = []
    for t in grid:
        r = eval_t(model, t, val, device)
        rows.append({'temperature': t, 'bpb': r['bpb'], 'seconds': r['seconds']})
        print(json.dumps(rows[-1]), flush=True)

    best = min(rows, key=lambda x: x['bpb'])
    out = {
        'checkpoint': str(args.checkpoint),
        'grid': tag,
        'results': rows,
        'best': best,
        'total_seconds': time.perf_counter() - started,
        'checkpoint_sha256': checkpoint.get('best_step'),
    }
    out_path = args.run_dir / f'temperature_{tag}.json'
    out_path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'best': best, 'output': str(out_path)}, indent=2))


if __name__ == '__main__':
    main()
