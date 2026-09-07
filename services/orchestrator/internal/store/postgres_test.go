package store

import (
	"context"
	"encoding/json"
	"os"
	"testing"
)

// TestPostgresRoundTrip only runs against a real database -- set
// SYNCSENTRY_TEST_DATABASE_URL to opt in (see README.md). This mirrors the
// analyzer's "skip if the real dependency isn't present" pattern for its
// own real-content fixtures: fast unit tests (memory_test.go) always run
// in CI; this one is for local/manual verification that the actual SQL
// (schema, SKIP LOCKED claim query, JSONB round-trip) is correct against a
// real Postgres, not just against Go's type system.
func TestPostgresRoundTrip(t *testing.T) {
	dsn := os.Getenv("SYNCSENTRY_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("SYNCSENTRY_TEST_DATABASE_URL not set; skipping real-Postgres test")
	}
	ctx := context.Background()
	pg, err := Open(ctx, dsn)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer pg.Close()

	captions := "/videos/asset.vtt"
	job := &Job{
		AssetName:            "asset.mkv",
		VideoPath:            "/videos/asset.mkv",
		CaptionsPath:         &captions,
		UseSyncnet:           true,
		SyncnetMinConfidence: 3.0,
	}
	if err := pg.CreateJob(ctx, job); err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	if job.ID == "" || job.Status != StatusQueued {
		t.Fatalf("unexpected job after create: %+v", job)
	}

	claimed, err := pg.ClaimNextQueuedJob(ctx)
	if err != nil {
		t.Fatalf("ClaimNextQueuedJob: %v", err)
	}
	if claimed == nil || claimed.ID != job.ID {
		t.Fatalf("expected to claim job %s, got %+v", job.ID, claimed)
	}
	if claimed.Status != StatusRunning {
		t.Fatalf("expected claimed job to be running, got %s", claimed.Status)
	}

	// A second claim should find nothing -- the job is already running.
	second, err := pg.ClaimNextQueuedJob(ctx)
	if err != nil {
		t.Fatalf("second ClaimNextQueuedJob: %v", err)
	}
	if second != nil {
		t.Fatalf("expected no queued jobs left, got %+v", second)
	}

	result := json.RawMessage(`{"overall_status":"fixed"}`)
	if err := pg.CompleteJob(ctx, job.ID, result); err != nil {
		t.Fatalf("CompleteJob: %v", err)
	}

	got, err := pg.GetJob(ctx, job.ID)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if got.Status != StatusDone {
		t.Fatalf("expected status done, got %s", got.Status)
	}
	// Postgres's JSONB round-trips semantically, not byte-for-byte (it
	// reformats whitespace) -- compare parsed values instead of raw bytes.
	var gotResult, wantResult map[string]any
	if err := json.Unmarshal(got.Result, &gotResult); err != nil {
		t.Fatalf("unmarshal got.Result: %v", err)
	}
	if err := json.Unmarshal(result, &wantResult); err != nil {
		t.Fatalf("unmarshal want result: %v", err)
	}
	if gotResult["overall_status"] != wantResult["overall_status"] {
		t.Fatalf("expected result %v, got %v", wantResult, gotResult)
	}
	if got.CaptionsPath == nil || *got.CaptionsPath != captions {
		t.Fatalf("expected captions_path %q, got %+v", captions, got.CaptionsPath)
	}

	_, err = pg.db.ExecContext(ctx, `DELETE FROM jobs WHERE id = $1`, job.ID)
	if err != nil {
		t.Fatalf("cleanup: %v", err)
	}
}
