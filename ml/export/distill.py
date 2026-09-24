"""ml.export.distill — distil the Head A teacher (or ensemble) into a real-time student (B14-T01).

Student = SSL front-end truncated to its first ``keep_layers`` transformer layers
+ a small back-end. The large teacher stays server-side for post-call forensics
and adjudicating borderline calls.

Loss (per batch):
    α · MSE(student_score, teacher_score)                      score regression
  + β · (1 − cos(student_emb_proj, teacher_emb))               embedding alignment
  + γ · OC-Softmax(student_score, labels)   (if labels given)  keep the task signal

Usage:
    python -m ml.export.distill --config ml/training/configs/distill_head_a.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import nn

from ml.training.losses import OCSoftmax
from packages.vg_models.heads.head_a_ssl.frontend import truncate_frontend
from packages.vg_models.heads.head_a_ssl.model import (
    HeadAConfig,
    HeadAModel,
    load_checkpoint,
    save_checkpoint,
)


def build_student(
    teacher: HeadAModel, keep_layers: int, backend: str = "nes2net", emb_dim: int | None = None
) -> HeadAModel:
    fe = truncate_frontend(teacher.frontend, keep_layers)
    layers = [i for i in teacher.cfg.frontend_layers if i <= keep_layers] or [keep_layers]
    cfg = HeadAConfig(
        frontend=teacher.cfg.frontend,
        frontend_revision=teacher.cfg.frontend_revision,
        frontend_layers=layers,
        freeze_frontend=teacher.cfg.freeze_frontend,
        backend=backend,
        emb_dim=emb_dim or teacher.cfg.emb_dim,
        version=teacher.cfg.version + "-distilled",
    )
    return HeadAModel(cfg, frontend=fe)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class DistillConfig:
    epochs: int = 5
    lr: float = 1e-3
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.2
    freeze_frontend: bool = True


class Distiller:
    def __init__(self, teachers: list[HeadAModel], student: HeadAModel, cfg: DistillConfig) -> None:
        self.teachers = [t.eval() for t in teachers]
        self.student = student
        self.cfg = cfg
        t_dim, s_dim = teachers[0].cfg.emb_dim, student.cfg.emb_dim
        self.proj = nn.Linear(s_dim, t_dim, bias=False)
        if cfg.freeze_frontend:
            student.frontend.requires_grad_(False)
        params = [p for p in student.parameters() if p.requires_grad] + list(self.proj.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr)
        self.oc = OCSoftmax()

    @torch.no_grad()
    def teacher_targets(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embs, scores = zip(*(t(wav) for t in self.teachers), strict=True)
        return F.normalize(torch.stack(embs).mean(0), dim=-1), torch.stack(scores).mean(0)

    def step(self, wav: torch.Tensor, labels: torch.Tensor | None = None) -> float:
        self.student.train()
        t_emb, t_score = self.teacher_targets(wav)
        s_emb, s_score = self.student(wav)
        loss = self.cfg.alpha * F.mse_loss(s_score, t_score)
        loss = loss + self.cfg.beta * (1 - F.cosine_similarity(self.proj(s_emb), t_emb).mean())
        if labels is not None and self.cfg.gamma > 0:
            loss = loss + self.cfg.gamma * self.oc(s_score, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def agreement(self, wav: torch.Tensor) -> dict[str, float]:
        self.student.eval()
        _, t = self.teacher_targets(wav)
        _, s = self.student(wav)
        corr = (
            float(np.corrcoef(t.numpy(), s.numpy())[0, 1])
            if len(t) > 2 and t.std() > 0 and s.std() > 0
            else float("nan")
        )
        return {"mse": float(F.mse_loss(s, t)), "pearson": corr}


def main(argv: list[str] | None = None) -> int:
    from torch.utils.data import DataLoader

    from ml.data.license_gate import DataPolicy
    from ml.training.dataset import ManifestAudioDataset
    from ml.training.train_head_a import load_split_rows, seed_everything

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(cfg.get("seed", 0))
    teachers = [load_checkpoint(p)[0] for p in cfg["teachers"]]
    student = build_student(
        teachers[0], cfg["keep_layers"], cfg.get("backend", "nes2net"), cfg.get("emb_dim")
    )
    d = Distiller(teachers, student, DistillConfig(**cfg.get("distill", {})))
    policy = DataPolicy.from_config(cfg)
    tr, dv, load = load_split_rows(cfg)
    dl = DataLoader(
        ManifestAudioDataset(
            tr, policy, load, cfg.get("seconds", 4.0), True, None, cfg.get("seed", 0)
        ),
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        drop_last=True,
    )
    dv_dl = DataLoader(
        ManifestAudioDataset(dv, policy, load, cfg.get("seconds", 4.0), False, None, 0, "eval"),
        batch_size=64,
    )
    for epoch in range(d.cfg.epochs):
        losses = [d.step(wav, y) for wav, y, _ in dl]
        wav, _, _ = next(iter(dv_dl))
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), **d.agreement(wav)}))
    out = Path(cfg.get("out", "runs/distilled/head_a_student.pt"))
    save_checkpoint(
        student,
        out,
        {
            "lineage": policy.lineage,
            "distilled_from": cfg["teachers"],
            "keep_layers": cfg["keep_layers"],
            "params": param_count(student),
        },
    )
    print(
        f"student {param_count(student):,} params (teacher {param_count(teachers[0]):,}) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
