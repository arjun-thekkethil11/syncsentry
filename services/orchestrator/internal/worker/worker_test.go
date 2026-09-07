package worker

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/arjunt/syncsentry-orchestrator/internal/pyclient"
	"github.com/arjunt/syncsentry-orchestrator/internal/store"
)

func writeTempVideo(t *testing.T) string {
	dir := t.TempDir()
	path := filepath.Join(dir, "asset.mkv")
	if err := os.WriteFile(path, []byte("fake video bytes"), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return path
}

func TestWorkerProcessesQueuedJobToDone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(pyclient.FixResult{
			JobID: "x", AssetName: "asset.mkv", OverallStatus: "resolved",
		})
	}))
	defer srv.Close()

	s := store.NewMemory()
	job := &store.Job{AssetName: "asset.mkv", VideoPath: writeTempVideo(t)}
	if err := s.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("CreateJob: %v", err)
	}

	w := New(s, pyclient.New(srv.URL), 10*time.Millisecond)
	w.tick(context.Background())

	got, err := s.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if got.Status != store.StatusDone {
		t.Fatalf("expected job to be done, got %s (err=%v)", got.Status, got.Error)
	}
	var result pyclient.FixResult
	if err := json.Unmarshal(got.Result, &result); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if result.OverallStatus != "resolved" {
		t.Fatalf("expected overall_status=resolved, got %q", result.OverallStatus)
	}
}

func TestWorkerMarksJobFailedOnAnalyzerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail":"Processing failed: bad video"}`))
	}))
	defer srv.Close()

	s := store.NewMemory()
	job := &store.Job{AssetName: "asset.mkv", VideoPath: writeTempVideo(t)}
	_ = s.CreateJob(context.Background(), job)

	w := New(s, pyclient.New(srv.URL), 10*time.Millisecond)
	w.tick(context.Background())

	got, err := s.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("GetJob: %v", err)
	}
	if got.Status != store.StatusFailed {
		t.Fatalf("expected job to be failed, got %s", got.Status)
	}
	if got.Error == nil || *got.Error == "" {
		t.Fatal("expected a non-empty error message on the failed job")
	}
}

func TestWorkerTickIsNoOpWhenNoQueuedJobs(t *testing.T) {
	s := store.NewMemory()
	w := New(s, pyclient.New("http://unused.invalid"), 10*time.Millisecond)
	w.tick(context.Background()) // must not panic or error with no jobs queued
}
