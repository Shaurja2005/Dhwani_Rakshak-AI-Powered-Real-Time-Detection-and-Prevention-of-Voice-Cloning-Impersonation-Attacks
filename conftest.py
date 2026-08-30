"""pytest conftest.py — global test fixtures and path setup.

Sets VG_ENV=test so the bus uses InMemoryBus and no real Redis is needed
for unit tests.  Integration tests that need Redis should mark themselves
with @pytest.mark.integration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path so tests can import packages/ directly
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Force test environment before any config module is loaded
os.environ.setdefault("VG_ENV", "test")
os.environ.setdefault("VG_SHADOW_MODE", "true")
os.environ.setdefault("VG_LOG_JSON", "false")
os.environ.setdefault("VG_LOG_LEVEL", "WARNING")

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks."""
    config.addinivalue_line("markers", "integration: requires running services (Redis, Postgres)")
    config.addinivalue_line("markers", "slow: takes more than 5 seconds")
    config.addinivalue_line("markers", "gpu: requires a CUDA-capable GPU")
