"""B3 — data & corpus engineering unit tests."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from ml.data import download, manifest, splits
from ml.data.channel.base import ChannelRecord
from ml.data.channel.codecs import (
    G711,
    FFmpegCodec,
    alaw_roundtrip,
    codec_available,
    mulaw_roundtrip,
)
from ml.data.channel.noise import AdditiveNoise
from ml.data.channel.packet_loss import PacketLoss
from ml.data.channel.rir import RIRConvolve
from ml.data.channel.webrtc_chain import ChannelSimulator
from ml.data.clone_job import ConsentViolation, docker_command, load_generators, plan, row_for
from ml.data.consent import ConsentRegister, CorpusConsent, SpeakerConsent, load_consent
from ml.data.license_gate import DataPolicy, LicenseViolation, enforce
from ml.data.manifest import ManifestRow
from ml.data.registry import load_registry

SR = 16000


def _row(i: int = 0, **kw: object) -> ManifestRow:
    base: dict[str, object] = dict(
        utt_id=f"u{i}",
        path=f"a/u{i}.wav",
        label="bona_fide",
        language="hi",
        speaker_id=f"s{i}",
        source_corpus="common_voice_indic",
        license="CC0-1.0",
        commercial_use=True,
        duration_s=3.0,
        sample_rate=SR,
    )
    base.update(kw)
    return ManifestRow.model_validate(base)


def _tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ---------------------------------------------------------------- registry (T01)
def test_registry_loads_and_every_row_declares_commercial_use() -> None:
    reg = load_registry()
    assert "in_the_wild" in reg.names()
    assert reg.get("in_the_wild").eval_only
    assert reg.get("indicsynth").commercial_use is False
    assert reg.get("sea_spoof").commercial_use is False


def test_registry_rejects_missing_or_non_bool_commercial_use(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text(
        "datasets:\n  - {name: x, kind: bona_fide, role: train, license: l, "
        "access: public, url: u}\n"
    )
    with pytest.raises(ValueError, match="commercial_use"):
        load_registry(p)
    p.write_text(
        "datasets:\n  - {name: x, kind: bona_fide, role: train, license: l, access: public, "
        "url: u, commercial_use: maybe}\n"
    )
    with pytest.raises(ValueError):
        load_registry(p)


# ---------------------------------------------------------------- downloader (T02/T03)
def test_verify_reports_missing_unpinned_ok_and_mismatch(tmp_path: Path) -> None:
    entry = load_registry().get("rirs_noises").model_copy(deep=True)
    assert download.verify_dataset(entry, tmp_path)[0].status == "missing"
    f = tmp_path / "rirs_noises" / entry.files[0].name
    f.parent.mkdir(parents=True)
    f.write_bytes(b"hello")
    res = download.verify_dataset(entry, tmp_path)[0]
    assert res.status == "unpinned"
    entry.files[0].sha256 = res.sha256
    assert download.verify_dataset(entry, tmp_path)[0].status == "ok"
    entry.files[0].sha256 = "0" * 64
    assert download.verify_dataset(entry, tmp_path)[0].status == "mismatch"


def test_gated_dataset_prints_instructions_instead_of_fetching(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    download.download_dataset(load_registry().get("asvspoof5"), tmp_path)
    assert "EULA" in capsys.readouterr().out


# ---------------------------------------------------------------- manifest (T04)
def test_manifest_roundtrip_and_report(tmp_path: Path) -> None:
    rows = [
        _row(0),
        _row(
            1,
            label="spoof",
            generator_family="tts_xtts",
            codec_chain=["resample:8000", "codec:g711u"],
        ),
    ]
    p = tmp_path / "m.jsonl"
    assert manifest.write_manifest(rows, p) == 2
    assert list(manifest.read_manifest(p)) == rows
    report = manifest.corpus_report(rows)
    assert "tts_xtts" in report and "g711u" in report and "bona_fide" in report


def test_manifest_enforces_label_consistency() -> None:
    with pytest.raises(ValueError):
        _row(label="spoof")
    with pytest.raises(ValueError):
        _row(generator_family="tts_xtts")


def test_read_manifests_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest.write_manifest([_row(0)], tmp_path / "a.jsonl")
    manifest.write_manifest([_row(0)], tmp_path / "b.jsonl")
    with pytest.raises(ValueError, match="duplicate"):
        manifest.read_manifests(str(tmp_path / "*.jsonl"))


# ---------------------------------------------------------------- license gate (T09)
def test_commercial_lineage_rejects_noncommercial_rows() -> None:
    policy = DataPolicy.from_config({"lineage": "commercial", "allow_noncommercial": False})
    bad = _row(1, source_corpus="indicsynth", license="CC-BY-NC-4.0", commercial_use=False)
    with pytest.raises(LicenseViolation, match="non-commercial"):
        enforce([bad], policy)


def test_row_flag_cannot_override_registry_flag() -> None:
    policy = DataPolicy("commercial", allow_noncommercial=False)
    lying = _row(1, source_corpus="indicsynth", commercial_use=True)
    with pytest.raises(LicenseViolation):
        enforce([lying], policy)


def test_research_lineage_allows_noncommercial_but_never_eval_only() -> None:
    policy = DataPolicy("research", allow_noncommercial=True)
    ok = _row(1, source_corpus="indicsynth", commercial_use=False)
    assert enforce([ok], policy) == [ok]
    itw = _row(2, source_corpus="in_the_wild", commercial_use=False)
    with pytest.raises(LicenseViolation, match="eval-only"):
        enforce([itw], policy, purpose="train")
    assert enforce([itw], policy, purpose="eval") == [itw]


def test_policy_config_must_be_explicit() -> None:
    with pytest.raises(LicenseViolation):
        DataPolicy.from_config({"lineage": "research"})
    with pytest.raises(LicenseViolation):
        DataPolicy("commercial", allow_noncommercial=True)


# ---------------------------------------------------------------- splits (T10)
def test_splits_are_speaker_disjoint_deterministic_and_hold_out_families() -> None:
    rows = [_row(i, speaker_id=f"s{i % 50}") for i in range(500)]
    rows += [
        _row(1000 + i, label="spoof", generator_family=fam, speaker_id=f"s{i % 50}")
        for i, fam in enumerate(["tts_xtts", "vc_rvc"] * 100)
    ]
    rows.append(_row(9999, source_corpus="in_the_wild", speaker_id="itw1", commercial_use=False))
    a = splits.make_splits(rows, seed=3, holdout_families=["vc_rvc"])
    b = splits.make_splits(rows, seed=3, holdout_families=["vc_rvc"])
    assert a == b
    splits.check_disjoint(rows, a)
    by_id = {r.utt_id: r for r in rows}
    assert "u9999" in a["eval"]
    assert all(by_id[u].generator_family != "vc_rvc" for u in a["train"] + a["dev"])
    assert a["train"]


def test_splits_cli_writes_json(tmp_path: Path) -> None:
    manifest.write_manifest([_row(i, speaker_id=f"s{i}") for i in range(30)], tmp_path / "m.jsonl")
    out = tmp_path / "s.json"
    assert splits.main(["--manifests", str(tmp_path / "*.jsonl"), "--out", str(out)]) == 0
    assert sum(len(v) for v in json.loads(out.read_text())["splits"].values()) == 30


# ---------------------------------------------------------------- consent + clone job (T05/T06/T11)
def test_committed_consent_register_is_valid() -> None:
    load_consent()


def test_consent_register_allows_corpus_or_unexpired_speaker() -> None:
    reg = ConsentRegister(
        corpora=[
            CorpusConsent(
                name="common_voice_indic", approved_by="x", date=dt.date(2026, 1, 1), basis="b"
            )
        ],
        speakers=[
            SpeakerConsent(
                speaker_id="vol1",
                name="V",
                date=dt.date(2026, 1, 1),
                scope="s",
                expires=dt.date(2026, 6, 1),
                withdrawal_contact="e",
                form_ref="f",
            )
        ],
    )
    assert reg.allows("common_voice_indic", "anyone")
    assert reg.allows("own", "vol1", today=dt.date(2026, 5, 1))
    assert not reg.allows("own", "vol1", today=dt.date(2026, 7, 1))
    assert not reg.allows("own", "ceo")


def test_clone_plan_refuses_unconsented_speakers() -> None:
    gens = [load_generators()["xtts_v2"]]
    with pytest.raises(ConsentViolation):
        plan([_row(0)], gens, ["namaste"], 1, "clones", ConsentRegister())


def test_clone_plan_builds_commands_and_provenance_rows() -> None:
    zoo = load_generators()
    assert len(zoo) == 13
    consent = ConsentRegister(
        corpora=[
            CorpusConsent(
                name="common_voice_indic", approved_by="x", date=dt.date(2026, 1, 1), basis="b"
            )
        ]
    )
    refs = [_row(i, speaker_id=f"s{i}") for i in range(5)]
    tasks = plan(refs, [zoo["xtts_v2"], zoo["knnvc"]], ["namaste"], 3, "clones", consent)
    assert len(tasks) == 6
    tts = next(t for t in tasks if t.generator.kind == "tts")
    vc = next(t for t in tasks if t.generator.kind == "vc")
    assert "--text" in docker_command(tts) and "--source-audio" in docker_command(vc)
    assert vc.source is not None and vc.source.speaker_id != vc.ref.speaker_id
    row = row_for(tts, 2.5, 24000)
    assert row.label == "spoof" and row.generator_family == "tts_xtts"
    assert row.speaker_id == tts.ref.speaker_id
    assert row.commercial_use is False  # XTTS license is non-commercial


# ---------------------------------------------------------------- channel simulator (T07)
def test_g711_roundtrips_are_close_but_lossy() -> None:
    x = _tone()
    for fn in (mulaw_roundtrip, alaw_roundtrip):
        y = fn(x)
        err = float(np.mean((x - y) ** 2) / np.mean(x**2))
        assert 0 < err < 0.01


def test_transforms_record_themselves() -> None:
    rng = np.random.default_rng(0)
    rec = ChannelRecord()
    x = _tone()
    y, sr = G711("a").apply(x, SR, rng, rec)
    assert sr == 8000 and len(y) == len(x) // 2
    y, sr = PacketLoss(0.2).apply(y, sr, rng, rec)
    y, sr = AdditiveNoise((10, 10)).apply(y, sr, rng, rec)
    y, sr = RIRConvolve().apply(y, sr, rng, rec)
    assert rec.codec_chain[:3] == ["resample:8000", "codec:g711a", "loss:0.2"]
    assert rec.snr_db == 10 and rec.rir_id and np.all(np.isfinite(y))


def test_additive_noise_hits_target_snr() -> None:
    rng = np.random.default_rng(1)
    x = _tone(2.0)
    y, _ = AdditiveNoise((15, 15)).apply(x, SR, rng, ChannelRecord())
    snr = 10 * np.log10(np.mean(x**2) / np.mean((y - x) ** 2))
    assert abs(snr - 15) < 0.5


@pytest.mark.skipif(not codec_available("opus"), reason="ffmpeg libopus not available")
def test_ffmpeg_opus_roundtrip_preserves_length() -> None:
    rec = ChannelRecord()
    x = _tone()
    y, sr = FFmpegCodec("opus", 12).apply(x, SR, np.random.default_rng(0), rec)
    assert sr == SR and len(y) == len(x) and "codec:opus@12k" in rec.codec_chain


def test_simulator_is_deterministic_per_utt_id_and_outputs_16k() -> None:
    sim = ChannelSimulator()
    x = _tone()
    a = sim.simulate(x, SR, "utt-1")
    b = sim.simulate(x, SR, "utt-1")
    assert a[1] == SR and np.array_equal(a[0], b[0]) and a[2] == b[2]
