package store

import (
	"context"
	"encoding/json"
	"testing"
)

func TestMemoryClaimOrdersByCreatedAtAndSkipsClaimedJobs(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()

	first := &Job{AssetName: "a.mkv", VideoPath: "/a.mkv"}
	second := &Job{AssetName: "b.mkv", VideoPath: "/b.mkv"}
	if err := m.CreateJob(ctx, first); err != nil {
		t.Fatalf("CreateJob: %v", err)
	}
	if err := m.CreateJob(ctx, second); err != nil {
		t.Fatalf("CreateJob: %v", err)
	}

	claimed, err := m.ClaimNextQueuedJob(ctx)
	if err != nil {
		t.Fatalf("ClaimNextQueuedJob: %v", err)
	}
	if claimed.ID != first.ID {
		t.Fatalf("expected to claim the first-created job %s, got %s", first.ID, claimed.ID)
	}
	if claimed.Status != StatusRunning {
		t.Fatalf("expected claimed job status running, got %s", claimed.Status)
	}

	claimed2, err := m.ClaimNextQueuedJob(ctx)
	if err != nil {
		t.Fatalf("ClaimNextQueuedJob: %v", err)
	}
	if claimed2.ID != second.ID {
		t.Fatalf("expected to claim the second job %s, got %s", second.ID, claimed2.ID)
	}

	none, err := m.ClaimNextQueuedJob(ctx)
	if err != nil {
		t.Fatalf("ClaimNextQueuedJob: %v", err)
	}
	if none != nil {
		t.Fatalf("expected no queued jobs left, got %+v", none)
	}
}

func TestMemoryCompleteAndFailJob(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()

	j := &Job{AssetName: "a.mkv", VideoPath: "/a.mkv"}
	_ = m.CreateJob(ctx, j)

	result := json.RawMessage(`{"overall_status":"fixed"}`)
	if err := m.CompleteJob(ctx, j.ID, result); err != nil {
		t.Fatalf("CompleteJob: %v", err)
	}
	got, err := m.GetJob(ctx, j.ID)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if got.Status != StatusDone || string(got.Result) != string(result) {
		t.Fatalf("unexpected job after CompleteJob: %+v", got)
	}

	j2 := &Job{AssetName: "b.mkv", VideoPath: "/b.mkv"}
	_ = m.CreateJob(ctx, j2)
	if err := m.FailJob(ctx, j2.ID, "boom"); err != nil {
		t.Fatalf("FailJob: %v", err)
	}
	got2, err := m.GetJob(ctx, j2.ID)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if got2.Status != StatusFailed || got2.Error == nil || *got2.Error != "boom" {
		t.Fatalf("unexpected job after FailJob: %+v", got2)
	}
}

func TestMemoryGetJobNotFound(t *testing.T) {
	m := NewMemory()
	if _, err := m.GetJob(context.Background(), "nope"); err != ErrNotFound {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

func TestMemoryListJobsFiltersByStatus(t *testing.T) {
	ctx := context.Background()
	m := NewMemory()
	j1 := &Job{AssetName: "a.mkv", VideoPath: "/a.mkv"}
	j2 := &Job{AssetName: "b.mkv", VideoPath: "/b.mkv"}
	_ = m.CreateJob(ctx, j1)
	_ = m.CreateJob(ctx, j2)
	_, _ = m.ClaimNextQueuedJob(ctx) // j1 -> running

	running := StatusRunning
	got, err := m.ListJobs(ctx, &running, 10)
	if err != nil {
		t.Fatalf("ListJobs: %v", err)
	}
	if len(got) != 1 || got[0].ID != j1.ID {
		t.Fatalf("expected only j1 running, got %+v", got)
	}

	all, err := m.ListJobs(ctx, nil, 10)
	if err != nil {
		t.Fatalf("ListJobs: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("expected 2 jobs total, got %d", len(all))
	}
}
