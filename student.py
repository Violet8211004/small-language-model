"""Student GPT: controlled structural experiments on the classroom baseline."""
import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE cos/sin cache; applied to Q/K with shape [batch, heads, time, head_dim]."""

    def __init__(self, head_dim, max_seq_len=256, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError('RoPE requires an even head dimension.')
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

    def forward(self, x):
        length = x.shape[2]
        cos = self.cos[:length][None, None, :, :]
        sin = self.sin[:length][None, None, :, :]
        return x * cos + rotate_half(x) * sin


def swiglu_hidden(width):
    """Match E01 at width=128 (344): scale hidden as (8/3)*width."""
    return int(round(344 * width / 128))


class SwiGLU(nn.Module):
    """Parameter-matched gated FFN: 3 * width * hidden ≈ 8 * width²."""

    def __init__(self, width):
        super().__init__()
        hidden = swiglu_hidden(width)
        self.gate = nn.Linear(width, hidden)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width=128, heads=4, ffn='gelu', rope=None, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.rope = rope
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.norm1 = nn.RMSNorm(width, eps=1e-5)
        self.norm2 = nn.RMSNorm(width, eps=1e-5)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        if ffn == 'swiglu':
            self.mlp = SwiGLU(width)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + (self.dropout(attn_out) if self.dropout is not None else attn_out)
        mlp_out = self.mlp(self.norm2(x))
        return x + (self.dropout(mlp_out) if self.dropout is not None else mlp_out)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        ffn = config.get('ffn', 'gelu')
        dropout = float(config.get('dropout', 0.0))
        head_dim = width // heads
        use_rope = config.get('position', 'learned') == 'rope'
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = None if use_rope else nn.Embedding(self.context, width)
        rope = RotaryEmbedding(head_dim, max_seq_len=self.context) if use_rope else None
        self.blocks = nn.ModuleList([
            Block(width, heads, ffn=ffn, rope=rope, dropout=dropout)
            for _ in range(config['depth'])])
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        self.temperature = float(config.get('temperature', 1.0))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        t = max(self.temperature, 1e-4)
        return F.log_softmax(self(ids).float() / t, dim=-1)


def build_model(config):
    return GPT(config)
