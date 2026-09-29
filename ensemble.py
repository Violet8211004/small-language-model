"""Probability-space ensemble for frozen student checkpoints."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ProbabilityEnsemble(nn.Module):
    """Mixture of causal predictors: p = sum_k w_k p_k, w_k >= 0, sum w = 1."""

    def __init__(self, members, weights):
        super().__init__()
        if len(members) != len(weights):
            raise ValueError('members and weights length mismatch')
        w = torch.tensor(weights, dtype=torch.float64)
        if (w < 0).any():
            raise ValueError('weights must be non-negative')
        w = w / w.sum()
        self.weights = w
        self.members = nn.ModuleList(members)
        self.context = members[0].context
        self.temperature = 1.0

    def forward(self, ids):
        raise NotImplementedError('Ensemble is evaluation-only via predict_log_probs.')

    @torch.no_grad()
    def predict_log_probs(self, ids):
        log_ws = torch.log(self.weights.to(ids.device))
        stacked = []
        for i, m in enumerate(self.members):
            lp = m.predict_log_probs(ids).float()
            stacked.append(lp + log_ws[i])
        return torch.logsumexp(torch.stack(stacked, dim=0), dim=0)


def build_ensemble(checkpoint_paths, weights, device, temperatures=None):
    from common import make_model
    import torch as th
    if temperatures is not None and len(temperatures) != len(checkpoint_paths):
        raise ValueError('temperatures length must match checkpoints')
    members = []
    for path in checkpoint_paths:
        ckpt = th.load(path, map_location='cpu', weights_only=True)
        model, _ = make_model(ckpt['implementation'], ckpt['config'], device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        members.append(model)
    if temperatures is not None:
        for model, temp in zip(members, temperatures):
            model.temperature = float(temp)
    return ProbabilityEnsemble(members, weights).to(device)
