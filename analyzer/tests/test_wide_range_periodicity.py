"""Unit tests for the periodicity self-check in `wide_range_offset.py`.

Unlike `test_wide_range_offset_real.py`, these are pure-numpy tests of the
two helper functions directly: no video fixture, no ffmpeg, no face
detector, so they run in milliseconds and always execute (no skip
markers). They exist to prove the periodicity check actually works (finds
a rhythm-spaced ambiguous peak when one is synthetically constructed, and
stays quiet on non-periodic/generic-noise curves), since the real-asset
smoke test in `test_wide_range_offset_real.py` only proves it doesn't
false-trigger on the handful of real assets available locally; it can't
prove the check catches anything, because none of those assets happen to
trip it.
"""
from __future__ import annotations

import numpy as np
import pytest

from syncsentry.lipsync.wide_range_offset import (
    AMBIGUITY_HEIGHT_RATIO,
    RHYTHM_PEAK_MIN_HEIGHT,
    WideRangeEstimate,
    _dominant_rhythm_period_ms,
    _rhythm_aliased_ambiguity,
    estimate_wide_range_offset,
)


def _periodic_impulse_train(period_ms: float, rate_hz: float, duration_s: float,
                             jitter_seed: int = 0) -> np.ndarray:
    """A synthetic stand-in for rhythmic speech energy: narrow Gaussian
    bumps at a fixed period, each with a small random amplitude/width
    jitter so this isn't a perfectly clean sinusoid (which would be an
    unrealistically easy case)."""
    rng = np.random.RandomState(jitter_seed)
    n = int(duration_s * rate_hz)
    t = np.arange(n) / rate_hz
    period_s = period_ms / 1000.0
    sig = np.zeros(n)
    n_bumps = int(duration_s / period_s) + 2
    for k in range(n_bumps):
        center = k * period_s + rng.uniform(-0.02, 0.02) * period_s
        width = period_s * 0.15
        amp = 1.0 + rng.uniform(-0.15, 0.15)
        sig += amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    sig += rng.normal(0, 0.05, size=n)
    return sig - np.mean(sig)


class TestDominantRhythmPeriod:
    def test_detects_clear_periodicity(self):
        rate_hz = 200.0
        period_ms = 400.0
        sig = _periodic_impulse_train(period_ms, rate_hz, duration_s=30.0)
        detected = _dominant_rhythm_period_ms(sig, rate_hz)
        assert detected is not None
        assert abs(detected - period_ms) <= 30.0  # within one grid step or two

    def test_returns_none_for_white_noise(self):
        rng = np.random.RandomState(42)
        sig = rng.normal(0, 1.0, size=6000)  # 30s at 200Hz, no structure at all
        sig = sig - np.mean(sig)
        detected = _dominant_rhythm_period_ms(sig, 200.0)
        assert detected is None

    def test_returns_none_for_too_short_signal(self):
        assert _dominant_rhythm_period_ms(np.array([0.1, 0.2, 0.3]), 200.0) is None

    def test_returns_none_for_constant_signal(self):
        # Zero-variance input (already-normalized constant); zero_lag_val ~ 0.
        assert _dominant_rhythm_period_ms(np.zeros(2000), 200.0) is None


class TestRhythmAliasedAmbiguity:
    def test_flags_comparably_tall_rhythm_spaced_secondary_peak(self):
        rate_hz = 200.0
        # lags from -2000ms to +2000ms in 5ms steps
        lags_ms = np.arange(-2000, 2005, 5, dtype=float)
        corr = np.full(lags_ms.shape, 0.05)
        top_lag = 300.0
        period_ms = 500.0
        # top peak
        corr += 0.9 * np.exp(-0.5 * ((lags_ms - top_lag) / 15.0) ** 2)
        # secondary peak exactly 1 period away, 88% as tall: above the
        # AMBIGUITY_HEIGHT_RATIO threshold
        corr += (0.9 * AMBIGUITY_HEIGHT_RATIO + 0.02) * np.exp(
            -0.5 * ((lags_ms - (top_lag + period_ms)) / 15.0) ** 2)

        ratio = _rhythm_aliased_ambiguity(lags_ms, corr, top_lag, period_ms)
        assert ratio is not None
        assert ratio >= AMBIGUITY_HEIGHT_RATIO

    def test_does_not_flag_when_secondary_peak_much_smaller(self):
        rate_hz = 200.0
        lags_ms = np.arange(-2000, 2005, 5, dtype=float)
        corr = np.full(lags_ms.shape, 0.05)
        top_lag = 300.0
        period_ms = 500.0
        corr += 0.9 * np.exp(-0.5 * ((lags_ms - top_lag) / 15.0) ** 2)
        # secondary peak only 30% as tall: well below the ambiguity threshold
        corr += 0.27 * np.exp(-0.5 * ((lags_ms - (top_lag + period_ms)) / 15.0) ** 2)

        ratio = _rhythm_aliased_ambiguity(lags_ms, corr, top_lag, period_ms)
        assert ratio is None

    def test_does_not_flag_peak_not_spaced_at_rhythm_period(self):
        """A comparably-tall secondary peak that is not spaced at a multiple
        of the rhythm period is generic cross-correlation noise, not
        rhythm-driven aliasing, and must not be flagged. This is what
        distinguishes this check from a naive "is there any other tall
        peak" heuristic, which would false-positive constantly."""
        lags_ms = np.arange(-2000, 2005, 5, dtype=float)
        corr = np.full(lags_ms.shape, 0.05)
        top_lag = 300.0
        period_ms = 500.0
        corr += 0.9 * np.exp(-0.5 * ((lags_ms - top_lag) / 15.0) ** 2)
        # tall secondary peak, but offset by 220ms, not near any multiple of 500ms
        corr += 0.85 * np.exp(-0.5 * ((lags_ms - (top_lag + 220.0)) / 15.0) ** 2)

        ratio = _rhythm_aliased_ambiguity(lags_ms, corr, top_lag, period_ms)
        assert ratio is None

    def test_ignores_the_top_peaks_own_shoulder(self):
        """A secondary 'peak' immediately adjacent to the top one (its own
        shoulder, not a distinct mode) must not be flagged even if tall,
        regardless of rhythm spacing: guarded by MIN_PEAK_SEPARATION_MS."""
        lags_ms = np.arange(-2000, 2005, 5, dtype=float)
        corr = np.full(lags_ms.shape, 0.05)
        top_lag = 300.0
        corr += 0.9 * np.exp(-0.5 * ((lags_ms - top_lag) / 40.0) ** 2)  # wide peak -> has shoulders
        ratio = _rhythm_aliased_ambiguity(lags_ms, corr, top_lag, 500.0)
        assert ratio is None


class TestEstimateWideRangeOffsetIntegration:
    """Exercises `estimate_wide_range_offset`'s periodicity gate itself
    (not just the two helpers above) by monkeypatching its two real
    dependencies, face-presence detection and the mouth/audio
    cross-correlation diagnostics, so the ambiguity path can be tested
    deterministically without a real video file or face detector."""

    def test_raises_when_diagnostics_are_rhythm_ambiguous(self, monkeypatch):
        import syncsentry.lipsync.wide_range_offset as wro

        monkeypatch.setattr(wro, "_face_intervals", lambda video_path: [(0.0, 30.0)])

        rate_hz = 200.0
        lags_ms = np.arange(-2000, 2005, 5, dtype=float)
        top_lag = 900.0
        period_ms = 400.0
        corr = np.full(lags_ms.shape, 0.05)
        corr += 0.9 * np.exp(-0.5 * ((lags_ms - top_lag) / 15.0) ** 2)
        corr += 0.87 * np.exp(-0.5 * ((lags_ms - (top_lag - period_ms)) / 15.0) ** 2)

        audio = _periodic_impulse_train(period_ms, rate_hz, duration_s=30.0)

        class _FakeEstimate:
            offset_ms = top_lag
            confidence = float(np.max(corr))
            direction = "audio_lags"

        class _FakeDiag:
            pass

        fake_diag = _FakeDiag()
        fake_diag.estimate = _FakeEstimate()
        fake_diag.audio_normalized = audio
        fake_diag.normalized_corr = corr
        fake_diag.lags_ms = lags_ms
        fake_diag.rate_hz = rate_hz

        monkeypatch.setattr(
            wro, "estimate_mouth_sync_offset_with_diagnostics",
            lambda *a, **k: fake_diag,
        )

        with pytest.raises(ValueError, match="ambiguous"):
            estimate_wide_range_offset("fake_path.mp4", min_face_coverage_s=1.0)

    def test_does_not_raise_when_no_rhythm_detected(self, monkeypatch):
        """Sanity check for the common case: non-periodic audio -> the gate
        is a no-op and the underlying estimate passes through unchanged."""
        import syncsentry.lipsync.wide_range_offset as wro

        monkeypatch.setattr(wro, "_face_intervals", lambda video_path: [(0.0, 30.0)])

        rate_hz = 200.0
        rng = np.random.RandomState(7)
        audio = rng.normal(0, 1.0, size=6000)
        audio -= np.mean(audio)
        lags_ms = np.arange(-2000, 2005, 5, dtype=float)
        corr = np.full(lags_ms.shape, 0.05)
        corr += 0.9 * np.exp(-0.5 * ((lags_ms - 900.0) / 15.0) ** 2)

        class _FakeEstimate:
            offset_ms = 900.0
            confidence = float(np.max(corr))
            direction = "audio_lags"

        class _FakeDiag:
            pass

        fake_diag = _FakeDiag()
        fake_diag.estimate = _FakeEstimate()
        fake_diag.audio_normalized = audio
        fake_diag.normalized_corr = corr
        fake_diag.lags_ms = lags_ms
        fake_diag.rate_hz = rate_hz

        monkeypatch.setattr(
            wro, "estimate_mouth_sync_offset_with_diagnostics",
            lambda *a, **k: fake_diag,
        )

        result = estimate_wide_range_offset("fake_path.mp4", min_face_coverage_s=1.0)
        assert isinstance(result, WideRangeEstimate)
        assert result.offset_ms == 900.0
        assert result.rhythm_period_ms is None
