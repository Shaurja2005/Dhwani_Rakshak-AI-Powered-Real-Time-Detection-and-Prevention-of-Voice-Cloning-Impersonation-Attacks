from packages.vg_core.config import get_settings
from packages.vg_core.versioning import ModelVersion, validate_calibration_version
import pytest

def test_config() -> None:
    settings = get_settings()
    assert settings.env == "test"
    assert settings.window_duration_s > 0

def test_versioning() -> None:
    mv = ModelVersion.parse("A@xlsr-nes2-v1.0.0")
    assert mv.head == "A"
    assert mv.frontend == "xlsr"
    assert mv.backend == "nes2"
    assert mv.semver == "1.0.0"

    mv2 = ModelVersion.parse("A@xlsr-nes2-v1.2.0")
    assert mv.is_compatible_with(mv2)

    with pytest.raises(ValueError):
        ModelVersion.parse("invalid")

    assert validate_calibration_version("cal-2026-01-01") == "cal-2026-01-01"
    with pytest.raises(ValueError):
        validate_calibration_version("invalid")
