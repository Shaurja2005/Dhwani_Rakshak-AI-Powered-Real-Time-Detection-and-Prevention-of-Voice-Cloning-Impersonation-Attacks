"""vg_core.config — layered configuration (env vars + YAML override).

Loading order (highest priority first):
  1. Environment variables prefixed with VG_
  2. YAML file pointed to by VG_CONFIG_FILE (default: config/vg_config.yaml)
  3. Built-in defaults

Usage::

    from packages.vg_core.config import settings
    settings.tenant_default          # "demo"
    settings.infer.device            # "cuda"
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-settings (nested models)
# ---------------------------------------------------------------------------


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_DB_")
    url: str = "postgresql://vg:vg@localhost:5432/vg"
    pool_size: int = 5
    max_overflow: int = 10


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_REDIS_")
    url: str = "redis://localhost:6379/0"
    feature_cache_ttl_s: int = 60  # invariant: cache windows for at most 60 s


class S3Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_S3_")
    endpoint: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket_audio: str = "vg-audio"
    bucket_evidence: str = "vg-evidence"


class InferSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_INFER_")
    device: Literal["cuda", "cpu", "mps"] = "cpu"
    batch_size: int = 4
    head_a_budget_ms: int = 120
    head_b_budget_ms: int = 15
    head_c_budget_ms: int = 25
    head_d_budget_ms: int = 30
    head_f_budget_ms: int = 5


class ContextSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_CONTEXT_")
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    asr_model: str = "ai4bharat/indicconformer"
    # I10: context never blocks the audio path
    max_latency_ms: int = 5000


class VGSettings(BaseSettings):
    """Top-level settings object.  Access via ``from packages.vg_core.config import settings``."""

    model_config = SettingsConfigDict(
        env_prefix="VG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    tenant_default: str = "demo"
    shadow_mode: bool = True  # new tenants default to shadow (I11 / B11-T07)

    # Window geometry (ADR 0002)
    window_duration_s: float = 3.0
    window_hop_s: float = 1.0

    # Quality-gate thresholds (B2-T05)
    min_voiced_s: float = 1.5
    min_snr_db: float = 5.0
    max_clipping_ratio: float = 0.05

    # Nested sub-settings
    db: DBSettings = Field(default_factory=DBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    infer: InferSettings = Field(default_factory=InferSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)

    config_file: Path | None = Field(None, alias="VG_CONFIG_FILE")

    @field_validator("window_duration_s", "window_hop_s")
    @classmethod
    def positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Window geometry must be positive")
        return v


def _load_yaml_overrides(path: Path) -> dict[str, Any]:
    """Load a YAML override file, silently skipping if absent."""
    if path.exists():
        with path.open() as fh:
            return yaml.safe_load(fh) or {}
    return {}


@lru_cache(maxsize=1)
def get_settings() -> VGSettings:
    """Return the singleton settings object (cached after first call)."""
    base = VGSettings()
    if base.config_file:
        overrides = _load_yaml_overrides(base.config_file)
        if overrides:
            base = VGSettings(**{**base.model_dump(), **overrides})
    return base


#: Module-level singleton — ``from packages.vg_core.config import settings``
settings: VGSettings = get_settings()
