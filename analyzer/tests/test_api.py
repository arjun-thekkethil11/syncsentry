"""End-to-end HTTP test for the API: upload a broken synthetic asset over
HTTP, get back a report + download links, and verify the corrected file
served back is actually fixed (round-trips through the real HTTP layer,
not just the Python pipeline function directly).
"""
from fastapi.testclient import TestClient

from syncsentry.api import app
from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.synth.fixture_gen import (
    FixtureSpec,
    generate_captions,
    generate_fixture,
    generate_piecewise_offset_fixture,
)

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_fix_endpoint_end_to_end(tmp_path):
    spec = FixtureSpec(duration_s=8.0, period_s=2.0, pulse_ms=80, offset_ms=160.0)
    video = generate_fixture(tmp_path / "broken.mkv", spec)
    captions = generate_captions(tmp_path / "broken.vtt", spec, caption_offset_ms=-90.0)

    with video.open("rb") as vf, captions.open("rb") as cf:
        resp = client.post(
            "/v1/fix",
            files={
                "video": ("broken.mkv", vf, "video/x-matroska"),
                "captions": ("broken.vtt", cf, "text/vtt"),
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_status"] == "resolved"

    av_issue = next(i for i in body["issues"] if i["name"] == "A/V sync")
    assert av_issue["had_issue"] is True
    assert abs(av_issue["residual_offset_ms"]) < 15.0

    job_id = body["job_id"]
    video_url = body["download_urls"]["corrected_video"]
    assert video_url == f"/v1/jobs/{job_id}/files/broken.corrected.mkv"

    download_resp = client.get(video_url)
    assert download_resp.status_code == 200

    # Save the downloaded bytes and prove *they* are actually fixed too --
    # not just trusting the report.
    out_path = tmp_path / "downloaded.mkv"
    out_path.write_bytes(download_resp.content)
    estimate = estimate_av_offset(str(out_path))
    assert estimate.direction == "in_sync"


def test_classify_drift_endpoint(tmp_path):
    video = generate_piecewise_offset_fixture(
        tmp_path / "drift.mkv", duration_s=18.0, control_points=[(0.0, 0.0), (18.0, 300.0)],
        period_s=1.0, pulse_ms=80,
    )
    with video.open("rb") as vf:
        resp = client.post(
            "/v1/classify-drift",
            files={"video": ("drift.mkv", vf, "video/x-matroska")},
            data={"window_s": "3.0"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pattern"] == "drift_increasing"
    assert body["slope_ms_per_s"] > 0


def test_fix_endpoint_video_only(tmp_path):
    spec = FixtureSpec(duration_s=6.0, period_s=2.0, pulse_ms=80, offset_ms=0.0)
    video = generate_fixture(tmp_path / "clean.mkv", spec)

    with video.open("rb") as vf:
        resp = client.post("/v1/fix", files={"video": ("clean.mkv", vf, "video/x-matroska")})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["issues"]) == 1
    assert "corrected_captions" not in body["download_urls"]
