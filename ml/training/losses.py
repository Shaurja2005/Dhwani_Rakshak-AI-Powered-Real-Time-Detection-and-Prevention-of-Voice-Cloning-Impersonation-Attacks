"""ml.training.losses — one-class and margin losses for Head A (B4-T04).

Convention matches HeadAModel: ``score = cos(emb, centre)``, higher = bona fide.
Labels: 1 = bona fide, 0 = spoof.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class OCSoftmax(nn.Module):
    """One-class softmax (Zhang et al., 2021).

    Pull bona fide above ``m_real`` cosine to the centre, push spoof below
    ``m_fake``. Spoof is never modelled as a compact class, which is why it
    generalises to unseen generators better than BCE.
    """

    def __init__(self, m_real: float = 0.9, m_fake: float = 0.2, alpha: float = 20.0) -> None:
        super().__init__()
        self.m_real, self.m_fake, self.alpha = m_real, m_fake, alpha

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bona = labels.float()
        margin = torch.where(bona > 0.5, self.m_real - scores, scores - self.m_fake)
        return F.softplus(self.alpha * margin).mean()


class AMSoftmax(nn.Module):
    """Additive-margin softmax over two class centres (bona fide = the model centre)."""

    def __init__(self, emb_dim: int, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.spoof_center = nn.Parameter(F.normalize(torch.randn(emb_dim), dim=0))
        self.margin, self.scale = margin, scale

    def forward(
        self, emb: torch.Tensor, bona_center: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        centers = F.normalize(
            torch.stack([self.spoof_center, bona_center]), dim=1
        )  # 0=spoof,1=bona
        logits = emb @ centers.T
        onehot = F.one_hot(labels.long(), 2).float()
        return F.cross_entropy(self.scale * (logits - self.margin * onehot), labels.long())


def build_loss(name: str, emb_dim: int) -> nn.Module:
    if name == "oc_softmax":
        return OCSoftmax()
    if name == "am_softmax":
        return AMSoftmax(emb_dim)
    raise ValueError(f"unknown loss {name!r}")
