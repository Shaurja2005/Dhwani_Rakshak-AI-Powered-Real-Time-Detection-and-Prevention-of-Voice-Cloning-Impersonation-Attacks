"""vg_core — shared library for the VoiceGuard platform.

Public API (import from here, not from sub-modules, to keep upgrade paths clean)::

    from packages.vg_core import models, config, logging, bus, errors, tracing
    from packages.vg_core.head_api import DetectionHead, HeadRegistry, BaseDetectionHead
    from packages.vg_core.stub_head import StubHead
    from packages.vg_core.versioning import ModelVersion, validate_calibration_version
"""

__version__ = "0.1.0"
