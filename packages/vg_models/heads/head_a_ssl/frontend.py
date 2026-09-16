"""Head A SSL front-ends with layer selection (B4-T02).

Every front-end returns *all* hidden states so the back-end can use a learned
weighted sum over intermediate layers (default 5–9). Intermediate layers keep
the artifact information; final layers trade it for semantics.

* ``HFSSLFrontend`` — wav2vec2 XLS-R / WavLM via ``transformers`` (lazy import).
* ``TinySSLFrontend`` — small random-init stand-in with the same interface, for
  tests and the untrained pipeline. It detects nothing.
"""

from __future__ import annotations

import torch
from torch import nn


class LayerWeightedSum(nn.Module):
    """Softmax-weighted sum over selected hidden-state layers."""

    def __init__(self, layers: list[int]) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one layer must be selected")
        self.layers = list(layers)
        self.logits = nn.Parameter(torch.zeros(len(layers)))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        n = len(hidden_states)
        bad = [i for i in self.layers if i >= n]
        if bad:
            raise ValueError(f"layers {bad} out of range for {n} hidden states")
        stack = torch.stack([hidden_states[i] for i in self.layers], dim=0)  # L,B,T,D
        w = torch.softmax(self.logits, dim=0).view(-1, 1, 1, 1)
        return (w * stack).sum(0)

    def weights(self) -> list[float]:
        return torch.softmax(self.logits.detach(), 0).tolist()


class SSLFrontend(nn.Module):
    hidden_size: int
    num_hidden_states: int
    name: str

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, ...]:  # B,T -> (L+1)×[B,F,D]
        raise NotImplementedError


class HFSSLFrontend(SSLFrontend):
    NAMES = {
        "facebook/wav2vec2-xls-r-300m": "xlsr300m",
        "facebook/wav2vec2-xls-r-1b": "xlsr1b",
        "microsoft/wavlm-large": "wavlmlarge",
    }

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        freeze: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel  # lazy: heavy + optional

        cfg = AutoConfig.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only
        )
        self.hidden_size = int(cfg.hidden_size)
        self.num_hidden_states = int(cfg.num_hidden_layers) + 1
        self.name = self.NAMES.get(model_id, model_id.split("/")[-1].replace("_", "-").lower())
        self.freeze = freeze
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # SSL models expect zero-mean unit-variance input.
        wav = (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + 1e-5)
        with torch.set_grad_enabled(not self.freeze and self.training):
            out = self.model(wav, output_hidden_states=True)
        return tuple(out.hidden_states)


class TinySSLFrontend(SSLFrontend):
    """wav2vec2-shaped toy: conv feature encoder (320× hop) + transformer layers."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 10, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.hidden_size = hidden_size
        self.num_hidden_states = num_layers + 1
        self.name = "tiny"
        convs: list[nn.Module] = []
        ch_in = 1
        for k, s in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            convs += [nn.Conv1d(ch_in, hidden_size, k, s), nn.GELU()]
            ch_in = hidden_size
        self.encoder = nn.Sequential(*convs)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                hidden_size, nhead=4, dim_feedforward=hidden_size * 2, batch_first=True
            )
            for _ in range(num_layers)
        )
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    p.copy_(torch.randn(p.shape, generator=g) * (1.0 / p.shape[1]) ** 0.5)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.encoder(wav.unsqueeze(1)).transpose(1, 2)  # B,F,D
        states = [x]
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        return tuple(states)


def build_frontend(name: str, **kwargs: object) -> SSLFrontend:
    if name == "tiny":
        return TinySSLFrontend(**kwargs)  # type: ignore[arg-type]
    return HFSSLFrontend(name, **kwargs)  # type: ignore[arg-type]
