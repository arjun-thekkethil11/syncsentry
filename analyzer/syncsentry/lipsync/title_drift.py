"""Classify a title's sync-drift pattern from its per-scene offset estimates.

Follows DiVAS (CVPR 2024): fit a robust line to (scene_time, scene_offset_ms)
via RANSAC, then use the line's slope/intercept plus the outlier structure to
tell apart the four failure modes that a single global-offset number can't
distinguish:

  * in_sync           -- flat line near zero
  * constant_offset   -- flat line, away from zero
  * drift_increasing  -- offset grows steadily over the title (e.g. a
                          progressive clock-rate mismatch)
  * drift_decreasing  -- offset shrinks/goes negative steadily
  * intermittent      -- most of the title fits a flat line, but a
                          *contiguous* block of scenes sits well off that
                          line (e.g. one badly-spliced segment)
  * unstable          -- no line fits most of the scenes; per-scene
                          estimates disagree too much to classify with
                          confidence (report the raw per-scene data instead
                          of guessing)

This module only depends on numpy (no scikit-learn) -- RANSAC for a 1-D line
fit is simple enough not to need a dependency for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from syncsentry.lipsync.scene_offsets import SceneOffset


@dataclass
class TitleDriftResult:
    pattern: str
    slope_ms_per_s: float
    intercept_ms: float
    inlier_ratio: float
    n_scenes: int
    intermittent_window_s: tuple[float, float] | None = None
    intermittent_offset_ms: float | None = None
    scene_offsets: list[SceneOffset] = field(default_factory=list)


def _ransac_line(t: np.ndarray, y: np.ndarray, residual_threshold_ms: float,
                  iterations: int = 300, seed: int = 42) -> tuple[float, float, np.ndarray]:
    """Minimal RANSAC line fit. Returns (slope, intercept, inlier_mask)."""
    n = len(t)
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)

    if n < 2:
        return 0.0, float(y[0]) if n == 1 else 0.0, np.ones(n, dtype=bool)

    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        if t[i] == t[j]:
            continue
        slope = (y[j] - y[i]) / (t[j] - t[i])
        intercept = y[i] - slope * t[i]
        residuals = np.abs(y - (slope * t + intercept))
        inliers = residuals <= residual_threshold_ms
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() >= 2:
        A = np.vstack([t[best_inliers], np.ones(best_inliers.sum())]).T
        slope, intercept = np.linalg.lstsq(A, y[best_inliers], rcond=None)[0]
    else:
        # Degenerate: fall back to a flat line at the median.
        slope, intercept = 0.0, float(np.median(y))

    residuals = np.abs(y - (slope * t + intercept))
    inliers = residuals <= residual_threshold_ms
    return float(slope), float(intercept), inliers


def _is_contiguous_block(sorted_indices: np.ndarray) -> bool:
    if len(sorted_indices) == 0:
        return False
    return bool(np.all(np.diff(sorted_indices) == 1))


def classify_title_drift(scenes: list[SceneOffset], slope_threshold_ms_per_s: float = 8.0,
                          in_sync_threshold_ms: float = 40.0,
                          residual_threshold_ms: float = 30.0,
                          min_inlier_ratio: float = 0.5) -> TitleDriftResult:
    if len(scenes) < 3:
        raise ValueError("Need at least 3 scene offset estimates to classify a title's drift pattern.")

    ordered = sorted(scenes, key=lambda s: s.t_center_s)
    t = np.array([s.t_center_s for s in ordered])
    y = np.array([s.offset_ms for s in ordered])

    slope, intercept, inlier_mask = _ransac_line(t, y, residual_threshold_ms)
    inlier_ratio = float(inlier_mask.mean())

    result = TitleDriftResult(
        pattern="unstable", slope_ms_per_s=slope, intercept_ms=intercept,
        inlier_ratio=inlier_ratio, n_scenes=len(ordered), scene_offsets=ordered,
    )

    if inlier_ratio < min_inlier_ratio:
        return result  # pattern stays "unstable"

    outlier_idx = np.where(~inlier_mask)[0]

    # A contiguous run of outlier scenes surrounded by inliers = intermittent,
    # regardless of what the majority-fit line otherwise looks like.
    if len(outlier_idx) > 0 and _is_contiguous_block(outlier_idx) and outlier_idx[0] > 0 \
            and outlier_idx[-1] < len(ordered) - 1:
        result.pattern = "intermittent"
        result.intermittent_window_s = (float(t[outlier_idx[0]]), float(t[outlier_idx[-1]]))
        result.intermittent_offset_ms = float(np.mean(y[outlier_idx]))
        return result

    if abs(slope) < slope_threshold_ms_per_s:
        mid_t = float(np.median(t))
        line_value_at_mid = slope * mid_t + intercept
        result.pattern = "in_sync" if abs(line_value_at_mid) <= in_sync_threshold_ms else "constant_offset"
    else:
        result.pattern = "drift_increasing" if slope > 0 else "drift_decreasing"

    return result
