"""Tests for the statistical drift-classification layer: windowed
per-scene offset estimation plus RANSAC title-level pattern classification.

Inject a known, named drift pattern via `generate_piecewise_offset_fixture`,
run the real pipeline (windowed estimation, then classification), and
assert the classifier recovers the pattern (constant / drift-early /
drift-late / intermittent) against synthetic ground truth.
"""
import pytest

from syncsentry.lipsync.scene_offsets import estimate_windowed_offsets
from syncsentry.lipsync.title_drift import classify_title_drift
from syncsentry.synth.fixture_gen import generate_piecewise_offset_fixture


def _classify(tmp_path, name, control_points, duration_s=16.0):
    video = generate_piecewise_offset_fixture(
        tmp_path / f"{name}.mkv", duration_s=duration_s, control_points=control_points,
        period_s=1.0, pulse_ms=80,
    )
    windows = estimate_windowed_offsets(str(video), window_s=3.0)
    return classify_title_drift(windows), windows


def test_in_sync_title(tmp_path):
    result, _ = _classify(tmp_path, "insync", [(0.0, 5.0), (16.0, 5.0)])
    assert result.pattern == "in_sync"


def test_constant_offset_title(tmp_path):
    result, _ = _classify(tmp_path, "constant", [(0.0, 120.0), (16.0, 120.0)])
    assert result.pattern == "constant_offset"
    assert abs(result.intercept_ms - 120.0) < 20.0


def test_drift_increasing_title(tmp_path):
    result, _ = _classify(tmp_path, "drift_inc", [(0.0, 0.0), (18.0, 300.0)], duration_s=18.0)
    assert result.pattern == "drift_increasing"
    assert result.slope_ms_per_s > 0


def test_drift_decreasing_title(tmp_path):
    result, _ = _classify(tmp_path, "drift_dec", [(0.0, 300.0), (18.0, 0.0)], duration_s=18.0)
    assert result.pattern == "drift_decreasing"
    assert result.slope_ms_per_s < 0


def test_intermittent_title(tmp_path):
    result, _ = _classify(
        tmp_path, "intermittent",
        [(0.0, 0.0), (7.0, 0.0), (7.3, 250.0), (10.7, 250.0), (11.0, 0.0), (18.0, 0.0)],
        duration_s=18.0,
    )
    assert result.pattern == "intermittent"
    assert result.intermittent_window_s is not None
    lo, hi = result.intermittent_window_s
    assert 6.0 <= lo <= 8.5
    assert 9.5 <= hi <= 12.0
    assert abs(result.intermittent_offset_ms - 250.0) < 30.0


def test_raises_with_too_few_scenes():
    from syncsentry.lipsync.scene_offsets import SceneOffset
    with pytest.raises(ValueError):
        classify_title_drift([SceneOffset(1.0, 0.0, 2.0, 10.0, 0.9)])
