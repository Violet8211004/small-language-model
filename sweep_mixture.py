"""Validation mixture-weight search for 2+ checkpoints (E14)."""
import argparse
import json
from pathlib import Path
import time
from common import ROOT, load_data, setup
from ensemble import build_ensemble
from evaluate import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--temperatures', nargs='+', type=float, default=None)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/e14-mixture-sweep')
    p.add_argument('--steps', type=int, default=9, help='Search alphas 0.1..0.9 for first vs rest')
    args = p.parse_args()
    if len(args.checkpoints) == 2:
        alphas = [i / 10 for i in range(1, 10)]
    else:
        p.error('Only 2-checkpoint weight line search implemented.')
    device, _ = setup('cpu', 'fp32', 4)
    val = load_data()['validation']
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for a in alphas:
        model = build_ensemble(args.checkpoints, [a, 1 - a], device, args.temperatures)
        r = score(model, *val, device, 'fp32')
        rows.append({'alpha': a, 'weight_first': a, 'bpb': r['bpb'], 'seconds': r['seconds']})
        print(json.dumps(rows[-1]), flush=True)
    best = min(rows, key=lambda x: x['bpb'])
    out = {'checkpoints': [str(c) for c in args.checkpoints], 'results': rows, 'best': best,
           'total_seconds': time.perf_counter() - started}
    (args.run_dir / 'mixture_sweep.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'best': best}, indent=2))


if __name__ == '__main__':
    main()
