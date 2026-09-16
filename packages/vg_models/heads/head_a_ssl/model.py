"""Head A model assembly: front-end → layer weighted sum → back-end → score."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from packages.vg_models.heads.head_a_ssl.backends import build_backend
from packages.vg_models.heads.head_a_ssl.frontend import (
    LayerWeightedSum,
    SSLFrontend,
    build_frontend,
)


@dataclass
class HeadAConfig:
    frontend: str = "facebook/wav2vec2-xls-r-300m"
    frontend_revision: str | None = None
    frontend_layers: list[int] = field(default_factory=lambda: [5, 6, 7, 8, 9])
    freeze_frontend: bool = True
    backend: str = "nes2net"
    emb_dim: int = 160
    version: str = "0.1.0"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HeadAConfig:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


class HeadAModel(nn.Module):
    """Outputs an L2-normalised embedding; ``score`` is cosine to the bona fide centre.

    Convention: higher ``score`` = more bona fide (OC-Softmax); p_spoof comes
    from calibration of ``-score``.
    """

    def __init__(self, cfg: HeadAConfig, frontend: SSLFrontend | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        if frontend is None:
            kwargs: dict[str, object] = {}
            if cfg.frontend != "tiny":
                kwargs = {"revision": cfg.frontend_revision, "freeze": cfg.freeze_frontend}
            frontend = build_frontend(cfg.frontend, **kwargs)
        self.frontend = frontend
        self.layer_sum = LayerWeightedSum(cfg.frontend_layers)
        self.backend = build_backend(cfg.backend, frontend.hidden_size, cfg.emb_dim)
        # Bona fide class centre, trained by the loss (OC-Softmax / AM-Softmax).
        self.center = nn.Parameter(F.normalize(torch.randn(cfg.emb_dim), dim=0))

    @property
    def model_version(self) -> str:
        return f"A@{self.frontend.name}-{self.cfg.backend}-v{self.cfg.version}"

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backend(self.layer_sum(self.frontend(wav))), dim=-1)

    def score(self, emb: torch.Tensor) -> torch.Tensor:
        return emb @ F.normalize(self.center, dim=0)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(wav)
        return emb, self.score(emb)


def trainable_state(model: HeadAModel) -> dict[str, torch.Tensor]:
    """Everything except a frozen pretrained front-end (keeps checkpoints small)."""
    frozen = model.cfg.freeze_frontend and model.cfg.frontend != "tiny"
    return {
        k: v for k, v in model.state_dict().items() if not (frozen and k.startswith("frontend."))
    }


def save_checkpoint(model: HeadAModel, path: Path | str, meta: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.cfg), "state": trainable_state(model), "meta": meta},
        str(path),
    )


def load_checkpoint(
    path: Path | str, frontend: SSLFrontend | None = None
) -> tuple[HeadAModel, dict[str, Any]]:
    blob = torch.load(
        str(path), map_location="cpu", weights_only=False
    )  # noqa: S614 - own artifacts only
    model = HeadAModel(HeadAConfig.from_dict(blob["config"]), frontend=frontend)
    missing, unexpected = model.load_state_dict(blob["state"], strict=False)
    if unexpected or any(not k.startswith("frontend.") for k in missing):
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    return model, blob.get("meta", {})
