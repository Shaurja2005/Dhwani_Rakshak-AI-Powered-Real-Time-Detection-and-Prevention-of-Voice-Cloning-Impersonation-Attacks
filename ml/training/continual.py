"""ml.training.continual — adding a new attack family without forgetting (B4-T10).

Procedure (run when a new generator family shows up, e.g. via drift alerts in B17):

1. Build a manifest for the new family with ml/data (clone_job + degrade), bona fide matched.
2. Rebuild splits with the new family *held out* and run ``make eval`` on the
   current model: this documents the pre-update miss rate.
3. ``python -m ml.training.continual --base runs/<run>/best.pt --config <cfg> --new-manifest <m>``
   trains on new data + a replay buffer from the old corpus, with an EWC
   penalty that anchors parameters important to the old task.
4. Run ``make eval`` again. Accept only if the new family improves AND no
   existing family / language regresses beyond tolerance (B15 fairness gate).
5. Blue/green roll out via models/registry.yaml (B17-T07).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from ml.data.manifest import ManifestRow
from ml.training.dataset import group_key


class EWC:
    """Elastic Weight Consolidation: diagonal-Fisher quadratic anchor."""

    def __init__(
        self,
        model: nn.Module,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.params = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        n_batches = 0
        model.eval()
        for wav, y in batches:
            model.zero_grad()
            _, score = model(wav.to(device))  # type: ignore[misc]
            loss_fn(score, y.to(device)).backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1
        self.fisher = {n: f / max(n_batches, 1) for n, f in fisher.items()}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        total = torch.zeros(())
        for n, p in model.named_parameters():
            if n in self.fisher:
                total = total + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return total


def replay_buffer(
    old_rows: Sequence[ManifestRow], per_group: int, seed: int = 0
) -> list[ManifestRow]:
    """Stratified sample of the old corpus: equal rows per bona fide / attack family."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[ManifestRow]] = {}
    for r in old_rows:
        groups.setdefault(group_key(r), []).append(r)
    out: list[ManifestRow] = []
    for _, members in sorted(groups.items()):
        k = min(per_group, len(members))
        out += [members[int(i)] for i in rng.choice(len(members), size=k, replace=False)]
    return out
