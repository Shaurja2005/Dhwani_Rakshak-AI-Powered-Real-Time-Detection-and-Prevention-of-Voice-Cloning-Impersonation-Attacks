"""Head A back-ends (B4-T03).

* ``Nes2Net`` (primary, ADR 0001): nested Res2Net blocks that operate directly
  on the high-dimensional SSL features — no dimensionality-reduction layer
  in front — followed by attentive statistics pooling.
* ``AASISTLite`` (secondary / ensemble): temporal + channel graph attention
  over SSL features, in the spirit of AASIST's heterogeneous graph layers.

Both map ``[B, T, D] -> [B, emb_dim]``. Scoring lives in the model / loss.
These are faithful-in-spirit re-implementations, not ports of the reference
code: verify parity against the published repos during B4-T01.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SEGate(nn.Module):
    def __init__(self, ch: int, r: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 4)), nn.ReLU(), nn.Linear(max(ch // r, 4), ch), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B,C,T
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class Res2Block(nn.Module):
    """Res2Net-style block: split channels into ``scale`` groups, hierarchical convs."""

    def __init__(self, ch: int, scale: int, kernel: int = 3, dilation: int = 1) -> None:
        super().__init__()
        if ch % scale:
            raise ValueError(f"channels {ch} not divisible by scale {scale}")
        self.scale = scale
        w = ch // scale
        pad = dilation * (kernel - 1) // 2
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(w, w, kernel, padding=pad, dilation=dilation),
                nn.BatchNorm1d(w),
                nn.ReLU(),
            )
            for _ in range(scale - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.chunk(x, self.scale, dim=1)
        out = [parts[0]]
        prev: torch.Tensor | None = None
        for i, conv in enumerate(self.convs, start=1):
            y = parts[i] if prev is None else parts[i] + prev
            prev = conv(y)
            out.append(prev)
        return torch.cat(out, dim=1)


class NestedRes2Block(nn.Module):
    """Outer Res2 split whose groups are themselves Res2 blocks + SE + residual."""

    def __init__(self, ch: int, outer_scale: int, inner_scale: int, dilation: int) -> None:
        super().__init__()
        self.outer_scale = outer_scale
        w = ch // outer_scale
        self.inner = nn.ModuleList(
            Res2Block(w, inner_scale, dilation=dilation) for _ in range(outer_scale - 1)
        )
        self.se = SEGate(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.chunk(x, self.outer_scale, dim=1)
        out = [parts[0]]
        prev: torch.Tensor | None = None
        for i, blk in enumerate(self.inner, start=1):
            y = parts[i] if prev is None else parts[i] + prev
            prev = blk(y)
            out.append(prev)
        return x + self.se(torch.cat(out, dim=1))


class AttentiveStatsPool(nn.Module):
    def __init__(self, ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.att = nn.Sequential(nn.Conv1d(ch, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B,C,T -> B,2C
        w = torch.softmax(self.att(x), dim=-1)
        mu = (w * x).sum(-1)
        sigma = torch.sqrt(((w * x * x).sum(-1) - mu * mu).clamp(min=1e-6))
        return torch.cat([mu, sigma], dim=1)


def _fit_channels(d: int, divisor: int) -> int:
    return d - d % divisor


class Nes2Net(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 160,
        num_blocks: int = 3,
        outer_scale: int = 8,
        inner_scale: int = 2,
    ) -> None:
        super().__init__()
        ch = _fit_channels(in_dim, outer_scale * inner_scale)
        self.trim = ch  # drop a few trailing dims instead of projecting (no bottleneck)
        self.blocks = nn.Sequential(
            *[
                NestedRes2Block(ch, outer_scale, inner_scale, dilation=2**i)
                for i in range(num_blocks)
            ]
        )
        self.pool = AttentiveStatsPool(ch)
        self.out = nn.Sequential(nn.BatchNorm1d(2 * ch), nn.Linear(2 * ch, emb_dim))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # B,T,D
        x = feats[..., : self.trim].transpose(1, 2)
        return self.out(self.pool(self.blocks(x)))


class GraphAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.att = nn.Linear(dim, 1)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:  # B,N,D
        pair = nodes.unsqueeze(2) * nodes.unsqueeze(1)  # B,N,N,D
        a = torch.softmax(self.att(torch.tanh(pair)).squeeze(-1), dim=-1)
        return F.selu(self.proj(a @ nodes)) + nodes


class AASISTLite(nn.Module):
    def __init__(
        self, in_dim: int, emb_dim: int = 160, gdim: int = 64, t_nodes: int = 32, c_nodes: int = 16
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, gdim)
        self.t_nodes, self.c_nodes, self.gdim = t_nodes, c_nodes, gdim
        # Channel nodes: time-max-pooled SSL dims mapped to c_nodes node vectors.
        self.c_proj = nn.Linear(in_dim, c_nodes * gdim)
        self.g_t = GraphAttention(gdim)
        self.g_c = GraphAttention(gdim)
        self.g_hetero = GraphAttention(gdim)
        self.out = nn.Linear(4 * gdim, emb_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # B,T,D
        x = self.proj(feats)  # B,T,G
        t = F.adaptive_max_pool1d(x.transpose(1, 2), self.t_nodes).transpose(1, 2)  # B,Nt,G
        c = self.c_proj(feats.max(dim=1).values).view(-1, self.c_nodes, self.gdim)  # B,Nc,G
        t, c = self.g_t(t), self.g_c(c)
        h = self.g_hetero(torch.cat([t, c], dim=1))
        pooled = [h.max(1).values, h.mean(1), t.mean(1), c.mean(1)]
        return self.out(torch.cat(pooled, dim=-1))


def build_backend(name: str, in_dim: int, emb_dim: int) -> nn.Module:
    if name == "nes2net":
        return Nes2Net(in_dim, emb_dim)
    if name == "aasist":
        return AASISTLite(in_dim, emb_dim)
    raise ValueError(f"unknown backend {name!r}")
