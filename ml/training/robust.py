"""ml.training.robust — Stage 3 robustness techniques (B4-T08).

Implemented here, each switchable from the training config:
* SAM optimizer wrapper
* mixup across bona fide / spoof pairs (applied to the loss, see train loop)
* feature-consistency self-distillation between two augmented views
* gradient-reversal domain-adversarial head (codec-ID or generator-ID)
* LoRA adapters for linear layers of the SSL front-end

Not yet implemented: two-front-end ensemble (XLS-R + WavLM) — train two
checkpoints and average in fusion (B9) instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021) around a base optimizer."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        **kwargs: object,
    ) -> None:
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base.param_groups

    @torch.no_grad()
    def first_step(self) -> None:
        grads = [
            p.grad.norm(2) for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        norm = torch.norm(torch.stack(grads), 2) if grads else torch.tensor(0.0)
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale
                p.add_(e)
                self.state[p]["e"] = e

    @torch.no_grad()
    def second_step(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                if "e" in self.state[p]:
                    p.sub_(self.state[p].pop("e"))
        self.base.step()

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> None:  # type: ignore[override]
        if closure is None:
            raise ValueError("SAM needs a closure that recomputes the loss")
        self.first_step()
        self.zero_grad()
        with torch.enable_grad():
            closure()
        self.second_step()


def mixup(
    wav: torch.Tensor, labels: torch.Tensor, alpha: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Returns mixed wav, labels_a, labels_b, lam. Loss = lam*L(a) + (1-lam)*L(b)."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    perm = torch.randperm(wav.shape[0], generator=generator)
    return lam * wav + (1 - lam) * wav[perm], labels, labels[perm], lam


def consistency_loss(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Symmetric stop-gradient cosine between two augmented views."""
    return 1 - 0.5 * (
        F.cosine_similarity(emb_a, emb_b.detach()).mean()
        + F.cosine_similarity(emb_b, emb_a.detach()).mean()
    )


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]  # noqa: E501
        ctx.lam = lam  # type: ignore[attr-defined]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]  # noqa: E501
        return -ctx.lam * grad, None  # type: ignore[attr-defined]


class DomainAdversary(nn.Module):
    """Predicts a nuisance domain (codec / generator) through a gradient-reversal layer,
    pushing the embedding to be invariant to it."""

    def __init__(self, emb_dim: int, n_domains: int, lam: float = 0.1) -> None:
        super().__init__()
        self.lam = lam
        self.clf = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_domains))

    def forward(self, emb: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        logits = self.clf(_GradReverse.apply(emb, self.lam))
        return F.cross_entropy(logits, domain)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scaling


def apply_lora(
    module: nn.Module, target_names: tuple[str, ...] = ("q_proj", "v_proj"), rank: int = 8
) -> int:
    """Replace matching Linear children in-place; returns number replaced."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank))
            n += 1
        else:
            n += apply_lora(child, target_names, rank)
    return n
