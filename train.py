"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    lr_peak = float(config.get('lr', 0.001))
    weight_decay = float(config.get('weight_decay', 0.1))
    warmup_steps = int(config.get('warmup_steps', 100))
    grad_clip = float(config.get('grad_clip', 1.0))
    lr_min_ratio = float(config.get('lr_min_ratio', 0.1))
    swa_checkpoints = int(config.get('swa_checkpoints', 0))
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    swa_states = []
    best_val_bpb = float('inf')
    best_step = 0
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        warmup = min(1., (step + 1) / max(1, warmup_steps))
        cosine = lr_min_ratio + (1. - lr_min_ratio) * .5 * (1. + math.cos(math.pi * step / args.steps))
        learning_rate = lr_peak * warmup * cosine
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
            if intermediate['bpb'] < best_val_bpb:
                best_val_bpb = intermediate['bpb']
                best_step = step + 1
                best_ckpt = args.run_dir / 'checkpoint_best.pt'
                torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                            'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            'seed': args.seed, 'train_tokens': (step + 1) * args.batch_size * 256,
                            'best_validation_bpb': best_val_bpb, 'best_step': best_step}, best_ckpt)
            if swa_checkpoints > 0:
                swa_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                swa_states = swa_states[-swa_checkpoints:]
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    if swa_states:
        averaged = {k: torch.stack([s[k].float() for s in swa_states]).mean(0).to(swa_states[-1][k].dtype)
                    for k in swa_states[-1]}
        model.load_state_dict(averaged)
        if device.type == 'cuda':
            model.to(device)
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_hparams':{'lr':lr_peak,'weight_decay':weight_decay,'warmup_steps':warmup_steps,
                               'grad_clip':grad_clip,'lr_min_ratio':lr_min_ratio,'swa_checkpoints':swa_checkpoints},
              'best_validation_step':best_step,'best_validation_bpb_gpu_eval':best_val_bpb if best_step else None,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
