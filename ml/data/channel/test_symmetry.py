"""CI gate for invariant I3: augmentation must be symmetric."""

import pytest

@pytest.mark.skip(reason="TODO(B3-T08)")
def test_channel_features_cannot_separate_classes() -> None:
    # Train a tiny classifier on channel features only; assert ~chance accuracy.
    raise NotImplementedError
