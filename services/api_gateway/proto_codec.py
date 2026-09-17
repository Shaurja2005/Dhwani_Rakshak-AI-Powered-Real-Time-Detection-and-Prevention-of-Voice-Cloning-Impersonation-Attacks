"""Protobuf stubs + conversion between Pydantic contract models and proto messages (B12-T01).

Stubs are generated from ``proto/voiceguard.proto`` into the git-ignored
``packages/vg_core/generated/`` on first use (same output as ``make proto``), so a
fresh checkout works without a manual codegen step.

Proto enums are prefixed (``RISK_STATE_HIGH``, ``CHANNEL_PSTN``); Pydantic enums
use bare values (``HIGH``, ``pstn``). Timestamps are RFC 3339 strings in proto.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from google.protobuf import json_format
from google.protobuf.message import Message
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "proto" / "voiceguard.proto"
OUT = ROOT / "packages" / "vg_core" / "generated"

_ENUM_FIELDS = {
    "state": ("RISK_STATE_", str.upper),
    "channel": ("CHANNEL_", str.upper),
    "encoding": ("AUDIO_ENCODING_", str.upper),
    "abstain_reason": ("ABSTAIN_REASON_", str.upper),
}
_CONSENT = {
    "legitimate_use": "CONSENT_BASIS_LEGITIMATE_USE",
    "explicit_consent": "CONSENT_BASIS_EXPLICIT",
    "none": "CONSENT_BASIS_NONE",
}


def load() -> tuple[ModuleType, ModuleType]:
    """Return (voiceguard_pb2, voiceguard_pb2_grpc), generating them if needed."""
    pb2_path = OUT / "voiceguard_pb2.py"
    if not pb2_path.exists() or pb2_path.stat().st_mtime < PROTO.stat().st_mtime:
        from grpc_tools import protoc

        OUT.mkdir(parents=True, exist_ok=True)
        rc = protoc.main(
            [
                "protoc",
                f"-I{PROTO.parent}",
                f"--python_out={OUT}",
                f"--grpc_python_out={OUT}",
                str(PROTO),
            ]
        )
        if rc != 0:
            raise RuntimeError("protoc failed to generate voiceguard stubs")
    if str(OUT) not in sys.path:
        sys.path.insert(0, str(OUT))
    return importlib.import_module("voiceguard_pb2"), importlib.import_module("voiceguard_pb2_grpc")


def _to_proto_enums(d: Any) -> Any:
    if isinstance(d, list):
        return [_to_proto_enums(x) for x in d]
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if k in _ENUM_FIELDS and isinstance(v, str):
            prefix, fn = _ENUM_FIELDS[k]
            out[k] = prefix + fn(v)
        elif k == "consent_basis" and isinstance(v, str):
            out[k] = _CONSENT.get(v, "CONSENT_BASIS_UNSPECIFIED")
        elif isinstance(v, dict | list):
            out[k] = _to_proto_enums(v)
        elif v is None:
            continue
        else:
            out[k] = v
    return out


def _from_proto_enums(d: Any) -> Any:
    if isinstance(d, list):
        return [_from_proto_enums(x) for x in d]
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if k in _ENUM_FIELDS and isinstance(v, str):
            prefix, _ = _ENUM_FIELDS[k]
            val = v.removeprefix(prefix)
            out[k] = val if k == "state" else val.lower()
        elif k == "consent_basis" and isinstance(v, str):
            out[k] = {b: a for a, b in _CONSENT.items()}.get(v, "legitimate_use")
        else:
            out[k] = _from_proto_enums(v)
    return out


def to_proto(model: BaseModel, message_cls: type[Message]) -> Message:
    msg = message_cls()
    json_format.ParseDict(
        _to_proto_enums(model.model_dump(mode="json")), msg, ignore_unknown_fields=True
    )
    return msg


def from_proto(message: Message, model_cls: type[BaseModel]) -> BaseModel:
    d = json_format.MessageToDict(
        message, preserving_proto_field_name=True, always_print_fields_with_no_presence=True
    )
    d = {k: v for k, v in _from_proto_enums(d).items() if v not in ("", None)}
    return model_cls.model_validate(d)
