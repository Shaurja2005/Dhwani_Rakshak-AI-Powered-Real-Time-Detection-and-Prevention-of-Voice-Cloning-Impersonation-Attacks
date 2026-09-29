"""Base-model pipeline end to end on a miniature ASVspoof 5 layout (tiny frontend, CPU).

extract (tar + md5) -> importer (protocols -> manifests + splits) -> feature cache
(resumable memmap) -> cached back-end training (resume) -> standard checkpoint ->
eval scorer. Proves the chain the real run uses; says nothing about accuracy.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml


def _flac(seed: int, spoof: bool, seconds: float = 2.5) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000
    f0 = rng.uniform(100, 220)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    x = x + (0.02 if spoof else 0.3) * rng.standard_normal(len(t))
    buf = io.BytesIO()
    sf.write(buf, (0.3 * x / np.max(np.abs(x))).astype(np.float32), 16000, format="FLAC")
    return buf.getvalue()


def _add(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


@pytest.fixture()
def mini_asv5(tmp_path: Path) -> Path:
    src = tmp_path / "data" / "asvspoof5"
    src.mkdir(parents=True)
    protos = {}
    for split, letter, folder, tar_name in (
        ("train", "T", "flac_T", "flac_T_aa.tar"),
        ("dev", "D", "flac_D", "flac_D_aa.tar"),
        ("eval", "E", "flac_E_eval", "flac_E_aa.tar"),
    ):
        lines = []
        with tarfile.open(src / tar_name, "w") as tf:
            for i in range(24):
                spoof = i % 2 == 1
                utt = f"{letter}_{i:010d}"
                attack = f"A0{(i // 2) % 3 + 1}" if spoof else "bonafide"
                codec = "C01 1" if (split != "train" and i % 4 == 0) else "- -"
                lines.append(
                    f"{letter}_{i % 5:04d} {utt} {'F' if i % 3 else 'M'} {codec} - "
                    f"{'AC1' if spoof else '-'} {attack} {'spoof' if spoof else 'bonafide'} -"
                )
                _add(tf, f"{folder}/{utt}.flac", _flac(i + ord(letter), spoof))
        protos[
            {
                "train": "ASVspoof5.train.tsv",
                "dev": "ASVspoof5.dev.track_1.tsv",
                "eval": "ASVspoof5.eval.track_1.tsv",
            }[split]
        ] = (
            "\n".join(lines) + "\n"
        )
    with tarfile.open(src / "ASVspoof5_protocols.tar", "w") as tf:
        for name, text in protos.items():
            _add(tf, name, text.encode())
    md5 = hashlib.md5((src / "flac_T_aa.tar").read_bytes()).hexdigest()  # noqa: S324
    (src / "README.txt").write_text(
        f"{md5}  ASVspoof5_train/flac_T_aa.tar   24\n", encoding="utf-8"
    )
    return tmp_path


def test_base_pipeline_end_to_end(mini_asv5: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.data.importers import asvspoof5 as imp
    from ml.training import feature_cache as fc
    from ml.training import train_head_a_cached as tr
    from packages.vg_models.heads.head_a_ssl.model import load_checkpoint
    from scripts.setup import extract_asvspoof5 as ex

    root = mini_asv5
    src = root / "data/asvspoof5"
    assert ex.published_md5(src / "README.txt") == {
        "flac_T_aa.tar": hashlib.md5((src / "flac_T_aa.tar").read_bytes()).hexdigest()
    }  # noqa: S324
    assert ex.main(["--src", str(src)]) == 0
    assert (src / "extracted/flac_T/T_0000000003.flac").exists()
    assert ex.main(["--src", str(src)]) == 0  # resumable: nothing left to do

    assert (
        imp.main(
            [
                "--extracted",
                str(src / "extracted"),
                "--data-root",
                str(root / "data"),
                "--manifests",
                str(root / "data/manifests"),
                "--splits",
                str(root / "splits/asvspoof5.json"),
            ]
        )
        == 0
    )
    rows = [
        json.loads(x)
        for x in (root / "data/manifests/asvspoof5_dev.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert (
        len(rows) == 24
        and rows[1]["generator_family"] == "A01"
        and rows[0]["codec_chain"] == ["codec:C01q1"]
    )
    assert rows[0]["gender"] == "male" and rows[1]["label"] == "spoof" and rows[0]["duration_s"] > 2

    monkeypatch.setattr(fc, "ROOT", root)
    cfg = {
        "run_name": "mini",
        "lineage": "research",
        "allow_noncommercial": True,
        "seed": 0,
        "out_dir": str(root / "runs"),
        "frontend": "tiny",
        "frontend_layers": [3, 4, 5],
        "freeze_frontend": True,
        "backend": "nes2net",
        "emb_dim": 16,
        "loss": "oc_softmax",
        "version": "0.0.1",
        "device": "cpu",
        "data": {
            "manifests": str(root / "data/manifests/asvspoof5_*.jsonl"),
            "splits": str(root / "splits/asvspoof5.json"),
            "data_root": str(root / "data"),
        },
        "cache": {
            "dir": str(root / "data/features/tiny"),
            "batch": 4,
            "workers": 0,
            "train": {"max_bona": None, "spoof_per_family": 3},
            "dev": {"max_bona": 6, "spoof_per_family": 2},
        },
        "train": {
            "epochs": 3,
            "batch_size": 8,
            "lr": 1e-3,
            "num_workers": 0,
            "seconds": 1.0,
            "patience": 5,
        },
    }
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert fc.main(["--config", str(cfg_path)]) == 0
    meta = json.loads((root / "data/features/tiny/train/meta.json").read_text(encoding="utf-8"))
    assert meta["n"] == 12 + 9 and meta["layers"] == [3, 4, 5]  # all bona fide + 3 per attack
    feats = np.load(root / "data/features/tiny/train/feats.npy", mmap_mode="r")
    assert feats.dtype == np.float16 and feats.shape[:2] == (21, 3) and np.abs(feats).sum() > 0
    assert fc.main(["--config", str(cfg_path)]) == 0  # resumable: already complete

    res = tr.train(cfg)
    assert Path(res["checkpoint"]).exists() and 0.0 <= res["eer"] <= 1.0
    cfg["train"]["epochs"] = 4
    res2 = tr.train(cfg)  # resumes from last.pt, runs only the new epoch
    lines = (root / "runs/mini/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [0, 1, 2, 3] and res2["checkpoint"] == res[
        "checkpoint"
    ]

    model, ck = load_checkpoint(res["checkpoint"])
    assert ck["train_datasets"] == ["asvspoof5"] and ck["train_families"] == ["A01", "A02", "A03"]
    assert "calibration" in ck and model.model_version == "A@tiny-nes2net-v0.0.1"
    # the loaded (full) model reproduces the cached-feature scores exactly on the same crop
    from ml.training.train_head_a_cached import CachedFeatures, forward_cached

    ds = CachedFeatures(root / "data/features/tiny/dev", False)
    x, _, i = ds[0]
    wav = fc.CropDataset([str(root / "data" / rows[0]["path"])], 1.0, False, 0)[0][0]
    ids = json.loads((root / "data/features/tiny/dev/meta.json").read_text(encoding="utf-8"))[
        "utt_ids"
    ]
    assert ids[0] == rows[0]["utt_id"]
    with torch.no_grad():
        s_cache = forward_cached(model.eval(), x[None].float())[1]
        s_audio = model(wav[None])[1]
    assert torch.allclose(s_cache, s_audio, atol=2e-2)

    from ml.eval.scorers import HeadAScorer

    out = HeadAScorer(checkpoint=res["checkpoint"]).score(wav.numpy())
    assert np.isfinite(out.score) and 0 <= out.p_spoof <= 1


def test_selection_and_stratified_eval_sample() -> None:
    from ml.data.manifest import ManifestRow
    from ml.eval.run_eval import stratified
    from ml.training.feature_cache import select_rows

    def row(i: int, fam: str | None, codec: str | None = None) -> ManifestRow:
        return ManifestRow(
            utt_id=f"u{i}",
            path="x.flac",
            label="spoof" if fam else "bona_fide",
            generator_family=fam,
            language="en",
            speaker_id="s",
            source_corpus="asvspoof5",
            license="x",
            commercial_use=False,
            duration_s=1.0,
            sample_rate=16000,
            codec_chain=[codec] if codec else [],
        )

    rows = [row(i, None) for i in range(10)] + [row(100 + i, f"A0{i % 3}") for i in range(60)]
    sel = select_rows(rows, max_bona=4, spoof_per_family=5, seed=0)
    assert sum(r.label == "bona_fide" for r in sel) == 4 and len(sel) == 4 + 15
    assert select_rows(rows, None, None, 0) == sorted(rows, key=lambda r: r.utt_id)
    rows += [row(500 + i, "A09", "codec:C05") for i in range(3)]  # tiny rare group
    s = stratified(rows, 20, 0)
    assert any(r.generator_family == "A09" for r in s)  # rare attack+codec kept
    assert 15 <= len(s) <= 30
