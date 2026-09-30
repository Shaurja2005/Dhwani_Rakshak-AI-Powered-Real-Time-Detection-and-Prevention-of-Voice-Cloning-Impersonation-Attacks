"""Per-language continual training (B3-T13, B4-T13) on a miniature layout, offline.

Importer pieces (row-group plan, record conversion, speaker-disjoint splits, finalize),
replay caches, fine-tuning from a previous model, and the orchestrator chain
cache -> train -> replay -> eval -> compare for two languages with the tiny front-end.
The Hub reading layer is exercised against local parquet files when pyarrow is present.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from ml.data.importers import indic
from ml.data.manifest import ManifestRow

GEOM = {"layers": [3, 4, 5], "frames": 49, "dim": 64, "seconds": 1.0}


def _wav_bytes(seed: int, sr: int = 24000, seconds: float = 1.6, spoof: bool = False) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    x = np.sin(2 * np.pi * rng.uniform(110, 230) * t) * 0.3
    x = x + (0.02 if spoof else 0.2) * rng.standard_normal(len(t))
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------- importer pieces
def test_plan_units_spreads_and_is_deterministic() -> None:
    units = [
        indic.Unit("indicsynth", f"B/train-{f}.parquet", g, 100, 1000)
        for f in range(10)
        for g in range(5)
    ]
    a = indic.plan_units(units, 1200, seed=7)
    assert a == indic.plan_units(units, 1200, seed=7) and sum(u.rows for u in a) >= 1200
    assert len({u.path for u in a}) >= 6  # many shards, not the first few
    big = indic.Unit("kathbath", "b/train-0.parquet", 0, 5000, 1)
    keep = indic.rows_to_keep(big, 200, seed=1)
    assert len(keep) == 200 == len(set(keep)) and keep == indic.rows_to_keep(big, 200, seed=1)
    assert indic.rows_to_keep(units[0], 200, 1) == list(range(100))


def test_convert_record_same_speaker_namespace(tmp_path: Path) -> None:
    unit = indic.Unit("indicsynth", "Bengali/train-00001.parquet", 3, 100, 1)
    fake = {
        "audio": {"bytes": _wav_bytes(1, spoof=True)},
        "Generative Model": "XTTS_v2",
        "Target Speaker ID": "1178",
        "Source Speaker_ID": None,
        "Gender": "Male",
    }
    row, prov = indic.convert_record(fake, "indicsynth", "bn", unit, 5, tmp_path)
    real = {
        "audio_filepath": {"bytes": _wav_bytes(2, sr=16000)},
        "fname": "844424930544036-1178-f.wav",
        "speaker_id": 1178,
        "gender": "female",
    }
    row2, _ = indic.convert_record(real, "kathbath", "bn", unit, 0, tmp_path)
    assert row.speaker_id == row2.speaker_id == "kb_bn_1178"  # clone and genuine voice: one speaker
    assert (row.label, row.generator_family, row.gender) == ("spoof", "xtts_v2", "male")
    assert (row2.label, row2.generator_family, row2.utt_id) == (
        "bona_fide",
        None,
        "kb_bn_844424930544036-1178-f",
    )
    x, sr = sf.read(tmp_path / row.path)
    assert sr == 16000 and abs(len(x) / sr - 1.6) < 0.01 and prov["original_sr"] == 24000
    assert row.license == "CC-BY-NC-4.0" and not row.commercial_use
    short = {**real, "audio_filepath": {"bytes": _wav_bytes(3, sr=16000, seconds=0.5)}}
    assert indic.convert_record(short, "kathbath", "bn", unit, 1, tmp_path) is None


def _rows(
    iso: str, n_spk: int, bona_per: int, spoof_per: int, gens: tuple[str, ...]
) -> list[ManifestRow]:
    rows = []
    for s in range(n_spk):
        for i in range(bona_per):
            rows.append(
                ManifestRow(
                    utt_id=f"kb_{iso}_{s}_{i}",
                    path=f"indic/{iso}/kathbath/kb_{iso}_{s}_{i}.flac",
                    label="bona_fide",
                    language=iso,
                    gender="female" if s % 2 else "male",
                    speaker_id=f"kb_{iso}_{s}",
                    source_corpus="kathbath",
                    license="CC-BY-4.0",
                    commercial_use=False,
                    duration_s=1.6,
                    sample_rate=16000,
                )
            )
        if s < n_spk - 2:  # some speakers have no clones, as in the real data
            for g in gens:
                for i in range(spoof_per):
                    u = f"isyn_{iso}_{s}_{g}_{i}"
                    rows.append(
                        ManifestRow(
                            utt_id=u,
                            path=f"indic/{iso}/indicsynth/{u}.flac",
                            label="spoof",
                            generator_family=g,
                            generator_name=f"indicsynth/{g}",
                            language=iso,
                            gender="female" if s % 2 else "male",
                            speaker_id=f"kb_{iso}_{s}",
                            source_corpus="indicsynth",
                            license="CC-BY-NC-4.0",
                            commercial_use=False,
                            duration_s=1.6,
                            sample_rate=16000,
                        )
                    )
    return rows


def test_assign_splits_speaker_disjoint_and_balanced() -> None:
    rows = _rows("bn", 30, 10, 4, ("freevc24", "xtts_v2"))
    sp = indic.assign_splits(rows, {"train": 0.7, "dev": 0.1, "eval": 0.2}, seed=1)
    assert sp == indic.assign_splits(rows, {"train": 0.7, "dev": 0.1, "eval": 0.2}, seed=1)
    by = {r.utt_id: r for r in rows}
    spk = {s: {by[u].speaker_id for u in ids} for s, ids in sp.items()}
    assert not (spk["train"] & spk["dev"] or spk["train"] & spk["eval"] or spk["dev"] & spk["eval"])
    for s, lo, hi in (("dev", 0.05, 0.2), ("eval", 0.12, 0.3)):
        for lab in ("bona_fide", "spoof"):
            share = sum(by[u].label == lab for u in sp[s]) / sum(r.label == lab for r in rows)
            assert lo <= share <= hi, (s, lab, share)


# ---------------------------------------------------------------- miniature world
def _fake_cache(d: Path, labels: list[int], fams: list[str], lang: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = len(labels)
    d.mkdir(parents=True, exist_ok=True)
    x = rng.standard_normal((n, 3, GEOM["frames"], GEOM["dim"])).astype(np.float16)
    x[np.asarray(labels) == 1] += np.float16(0.5)  # learnable
    np.save(d / "feats.npy", x)
    np.save(d / "done.npy", np.ones(n, dtype=bool))
    meta = {
        **GEOM,
        "split": d.name,
        "n": n,
        "frontend": "tiny",
        "frontend_revision": "tiny-rev",
        "utt_ids": [f"{d.name}_{i}" for i in range(n)],
        "labels": labels,
        "families": fams,
        "speakers": ["s"] * n,
        "corpora": ["asvspoof5"] * n,
        "languages": [lang] * n,
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_language(root: Path, iso: str, seed: int) -> None:
    """Audio + parts for one tiny language, then the importer's own finalize()."""
    rows = _rows(iso, 12, 4, 2, ("freevc24", "xtts_v2"))
    work = root / "data/indic" / iso
    (work / "parts").mkdir(parents=True)
    lines = []
    for k, r in enumerate(rows):
        p = root / "data" / r.path
        p.parent.mkdir(parents=True, exist_ok=True)
        x, _ = sf.read(io.BytesIO(_wav_bytes(seed * 1000 + k, 16000, spoof=r.label == "spoof")))
        sf.write(p, x, 16000)
        lines.append(json.dumps({"row": r.model_dump(), "prov": {"original_sr": 16000}}))
    (work / "parts/all_00000.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plan = {
        "iso": iso,
        "language": iso,
        "seed": 1,
        "revisions": {},
        "fracs": {"train": 0.6, "dev": 0.2, "eval": 0.2},
    }
    rep = indic.finalize(plan, work / "parts", root / "data/manifests", root / "splits", work)
    assert all(rep[s]["bona_fide"] and rep[s]["spoof"] for s in ("train", "dev", "eval"))


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from ml.eval import run_eval
    from ml.training import feature_cache as fc
    from ml.training import train_head_a_cached as tr
    from scripts import train_base_model as tbm
    from scripts import train_indic_language as tl

    root = tmp_path
    monkeypatch.chdir(root)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # stay off the GPU
    for mod, attr, val in (
        (fc, "ROOT", root),
        (tbm, "ROOT", root),
        (tl, "ROOT", root),
        (tl, "CONFIG", root / "indic.yaml"),
        (tl, "LEDGER", root / "runs/indic/ledger.json"),
        (tl, "CFG_DIR", root / "runs/indic/configs"),
        (tl, "RUNS_DIR", root / "docs/benchmarks/runs"),
        (run_eval, "RUNS", root / "docs/benchmarks/runs"),
        (run_eval, "REPORT", root / "docs/benchmarks/REPORT.md"),
    ):
        monkeypatch.setattr(mod, attr, val)

    def run_inproc(*args: str) -> int:  # the orchestrator's subprocesses, in-process
        if args[0] == "-m":
            return int(importlib.import_module(args[1]).main(list(args[2:])) or 0)
        assert args[0] == "ml/eval/run_eval.py"
        return int(run_eval.main(list(args[1:])) or 0)

    monkeypatch.setattr(tl, "run", run_inproc)

    # ASVspoof 5 side: clean caches (fake features) + a few eval clips with audio
    cache = root / "data/features/tiny"
    fams = ["A01", "A02"]
    lab = [1] * 10 + [0] * 10
    _fake_cache(cache / "train", lab, ["bona_fide"] * 10 + fams * 5, "en", 0)
    _fake_cache(cache / "dev", lab, ["bona_fide"] * 10 + fams * 5, "en", 1)
    ev = []
    for i in range(12):
        spoof = i % 2 == 1
        rel = f"asvspoof5/eval/E_{i}.flac"
        (root / "data/asvspoof5/eval").mkdir(parents=True, exist_ok=True)
        x, _ = sf.read(io.BytesIO(_wav_bytes(500 + i, 16000, spoof=spoof)))
        sf.write(root / "data" / rel, x, 16000)
        ev.append(
            ManifestRow(
                utt_id=f"asv5_E_{i}",
                path=rel,
                label="spoof" if spoof else "bona_fide",
                generator_family="A20" if spoof else None,
                language="en",
                gender="male" if i % 4 < 2 else "female",
                speaker_id=f"e{i % 3}",
                source_corpus="asvspoof5",
                license="x",
                commercial_use=False,
                duration_s=1.6,
                sample_rate=16000,
            )
        )
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    (root / "data/manifests/asvspoof5_eval.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in ev), encoding="utf-8"
    )
    (root / "splits").mkdir(exist_ok=True)
    (root / "splits/asvspoof5.json").write_text(
        json.dumps({"splits": {"train": [], "dev": [], "eval": [r.utt_id for r in ev]}}),
        encoding="utf-8",
    )

    model = {
        "frontend": "tiny",
        "frontend_revision": "tiny-rev",
        "frontend_layers": [3, 4, 5],
        "freeze_frontend": True,
        "backend": "nes2net",
        "emb_dim": 16,
        "loss": "oc_softmax",
    }
    aug = {
        "seed": 3,
        "p_rawboost": 0.0,
        "p_channel": 1.0,
        "channel": {"p_noise": 1.0, "codec_weights": {"clean": 1.0, "g711u": 1.0}},
    }
    (root / "aug.yaml").write_text(yaml.safe_dump({"augment": aug}), encoding="utf-8")
    base_train = {
        "epochs": 2,
        "batch_size": 8,
        "lr": 1e-3,
        "num_workers": 0,
        "seconds": 1.0,
        "patience": 3,
    }
    # the starting model ("v0.2"), trained on the fake ASVspoof caches
    tr.train(
        {
            "run_name": "head_a_aug",
            "lineage": "research",
            "allow_noncommercial": True,
            "seed": 0,
            "out_dir": str(root / "runs"),
            **model,
            "version": "0.2.0",
            "device": "cpu",
            "cache": {"dir": str(cache)},
            "train": base_train,
        }
    )
    cfg = {
        "lineage": "research",
        "allow_noncommercial": True,
        "seed": 1,
        "init_from": "runs/head_a_aug/best.pt",
        **model,
        "import": {"spoof": 10, "bona": 10, "workers": 1},
        "cache": {"root": "data/features/tiny", "batch": 4, "workers": 0, "aug_fraction": 0.5},
        "augment_from": "aug.yaml",
        "replay": {
            "asvspoof5": {"train": {"bona": 4, "spoof": 4}, "dev": {"bona": 3, "spoof": 3}},
            "language": {"train": {"bona": 3, "spoof": 3}, "dev": {"bona": 2, "spoof": 2}},
        },
        "train": {**base_train, "early_stop": "mean_by_set"},
        "eval": {"asvspoof5_sample": 12, "full": False, "bootstrap": 5},
        "compare": {"max_regression_eer": None},
    }
    (root / "indic.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _write_language(root, "bn", 1)
    _write_language(root, "or", 2)
    return root


def test_two_languages_end_to_end(world: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from ml.training.train_head_a_cached import CachedFeatures
    from packages.vg_models.heads.head_a_ssl.model import load_checkpoint
    from scripts import train_indic_language as tl

    root = world
    assert tl.main(["bengali", "train"]) == 2  # cache first: refused with a message
    for step in ("cache", "train", "replay", "eval"):
        assert tl.main(["bengali", step]) == 0, step
    st = tl.status(tl.entry("bn"), tl.load_cfg())
    assert all(st.values()), st

    cfg1 = yaml.safe_load((root / "runs/indic/configs/01_bn.yaml").read_text(encoding="utf-8"))
    assert cfg1["train"]["init_from"] == "runs/head_a_aug/best.pt"
    assert cfg1["train"]["train_sets"] == [
        "indic_bn/train",
        "indic_bn/train_aug1",
        "replay/asv5_train",
    ]
    _, ck = load_checkpoint(root / "runs/indic_01_bn/best.pt")
    assert (
        ck["init_from"] == "runs/head_a_aug/best.pt"
        and "bn" in ck["languages"]
        and "en" in ck["languages"]
    )
    chains = json.loads(
        (root / "data/features/tiny/indic_bn/train_aug1/chains.json").read_text("utf-8")
    )
    assert all(chains)
    rep = CachedFeatures(root / "data/features/tiny/replay/bn_train", False)
    assert sorted(rep.labels.tolist()) == [0, 0, 0, 1, 1, 1]
    assert (root / "docs/benchmarks/runs/A_tiny-nes2net-v0.2.0__indic_01.json").exists()
    assert (root / "docs/benchmarks/runs/A_tiny-nes2net-v0.3.1__indic_01.json").exists()
    out = capsys.readouterr().out
    assert "You can now delete the folder" in out

    # second language: starts from the first, replays it, and both are evaluated
    assert tl.main(["odia", "all"]) == 0
    cfg2 = yaml.safe_load((root / "runs/indic/configs/02_or.yaml").read_text(encoding="utf-8"))
    assert cfg2["train"]["init_from"] == "runs/indic_01_bn/best.pt"
    assert (
        "replay/bn_train" in cfg2["train"]["train_sets"]
        and "replay/bn_dev" in cfg2["train"]["dev_sets"]
    )
    ev2 = yaml.safe_load((root / "runs/indic/configs/02_or_eval.yaml").read_text(encoding="utf-8"))
    assert [s["name"] for s in ev2["eval_sets"]] == [
        "indic_bn_test",
        "indic_or_test",
        "asvspoof5_eval",
    ]
    rec = json.loads((root / "runs/indic_02_or/metrics.jsonl").read_text("utf-8").splitlines()[-1])
    assert rec["early_stop"] == "mean_by_set" and "replay/bn_dev" in rec["dev_eer_by_set"]
    capsys.readouterr()
    assert tl.main(["compare"]) == 0
    out = capsys.readouterr().out
    assert "step 01 bengali" in out and "step 02 odia" in out and "(new)" in out
    report = (root / "docs/benchmarks/REPORT.md").read_text(encoding="utf-8")
    assert "· indic_02" in report and "indic_bn_test" in report
    assert tl.main(["queue", "bengali", "odia"]) == 0  # everything done: the queue just skips
    with pytest.raises(SystemExit, match="unknown language"):  # typo caught before any work
        tl.main(["queue", "bengali", "klingon"])


def test_hub_layer_on_local_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """list_units + row-group reads + finalize against local parquet shaped like the Hub's."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    hub = tmp_path / "hub"

    def shard(repo: str, path: str, recs: list[dict[str, Any]]) -> None:
        p = hub / repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(recs), p, row_group_size=5)

    synth = [
        {
            "audio": {"bytes": _wav_bytes(i, spoof=True), "path": None},
            "Generative Model": "freevc24" if i < 20 else "xtts_v2",
            "Target Speaker ID": str(i % 6),
            "Source Speaker_ID": float(9) if i < 20 else None,
            "Gender": "Female" if i < 20 else "Male",
        }
        for i in range(40)
    ]
    shard(indic.INDICSYNTH, "Bengali/train-00000-of-00002.parquet", synth[:20])
    shard(indic.INDICSYNTH, "Bengali/train-00001-of-00002.parquet", synth[20:])
    kb = [
        {
            "audio_filepath": {"bytes": _wav_bytes(100 + i, 16000), "path": None},
            "fname": f"8444{i:05d}-{i % 8}-f.wav",
            "speaker_id": i % 8,
            "gender": "female",
        }
        for i in range(30)
    ]
    shard(indic.KATHBATH, "bengali/train-00000-of-00001.parquet", kb[:25])
    shard(indic.KATHBATH, "bengali/valid-00000-of-00001.parquet", kb[25:])

    class Api:
        def dataset_info(self, repo: str) -> Any:  # noqa: ANN401
            return type("I", (), {"sha": "rev0"})()

        def list_repo_tree(
            self, repo: str, path_in_repo: str, **_: Any
        ) -> list[Any]:  # noqa: ANN401
            base = hub / repo
            return [
                type("F", (), {"path": p.relative_to(base).as_posix()})()
                for p in sorted((base / path_in_repo).rglob("*.parquet"))
            ]

    class FS:
        def open(self, path: str, mode: str = "rb", **_: Any) -> Any:  # noqa: ANN401
            _, rest = path.split("/", 1)  # "datasets/<org>/<name>@rev/<file>"
            org, rest = rest.split("/", 1)
            name_rev, file = rest.split("/", 1)
            return (hub / f"{org}/{name_rev.split('@')[0]}" / file).open(mode)

    monkeypatch.setattr(indic, "_hub", lambda: (pq, Api(), FS()))
    rep = indic.import_language(
        "bengali",
        20,
        20,
        seed=3,
        data_root=tmp_path / "data",
        manifests=tmp_path / "man",
        splits_dir=tmp_path / "splits",
        workers=2,
    )
    assert rep["n"] >= 40 and set(rep["original_sample_rates"]["indicsynth"]) == {"24000"}
    plan = json.loads((tmp_path / "data/indic/bn/plan.json").read_text(encoding="utf-8"))
    assert plan["revisions"] == {"indicsynth": "rev0", "kathbath": "rev0"}
    again = indic.import_language(
        "bengali",
        20,
        20,
        seed=3,
        data_root=tmp_path / "data",
        manifests=tmp_path / "man",
        splits_dir=tmp_path / "splits",
    )
    assert again["n"] == rep["n"]  # resumed: nothing re-downloaded, same result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
