"""CPU FP32 validation for probability ensemble (E13–E15)."""
import argparse
import json
from pathlib import Path
import torch
from common import PROTOCOL, ROOT, setup, sha
from evaluate import score
from ensemble import build_ensemble
from common import load_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--weights', nargs='+', type=float, required=True)
    p.add_argument('--temperatures', nargs='+', type=float, default=None,
                   help='Per-checkpoint temperature (default: each ckpt config)')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--output', type=Path)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    device, precision = setup('cpu', 'fp32', args.threads)
    data = load_data()
    tokens, byte_count = data[args.split]
    if args.temperatures is not None and len(args.temperatures) != len(args.checkpoints):
        p.error('--temperatures count must match --checkpoints')
    model = build_ensemble(args.checkpoints, args.weights, device, args.temperatures)
    result = score(model, tokens, byte_count, device, precision)
    result.pop('window_nll_nats')
    result.update(
        protocol=PROTOCOL,
        split=args.split,
        precision='fp32',
        checkpoints=[str(c) for c in args.checkpoints],
        weights=args.weights,
        temperatures=args.temperatures,
    )
    out = args.output or ROOT / 'runs' / 'ensemble-last' / f'{args.split}_cpu_fp32.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
