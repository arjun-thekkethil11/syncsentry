package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/arjunt/syncsentry-orchestrator/internal/store"
)

func newTestServer(t *testing.T) (*Server, *http.ServeMux) {
	s := &Server{Store: store.NewMemory()}
	mux := http.NewServeMux()
	s.routes(mux)
	return s, mux
}

func writeTempVideo(t *testing.T) string {
	dir := t.TempDir()
	path := filepath.Join(dir, "asset.mkv")
	if err := os.WriteFile(path, []byte("fake"), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return path
}

func TestCreateJobRejectsMissingVideoPath(t *testing.T) {
	_, mux := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/jobs", bytes.NewBufferString(`{}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing video_path, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestCreateJobRejectsUnreadableVideoPath(t *testing.T) {
	_, mux := newTestServer(t)
	body, _ := json.Marshal(createJobRequest{VideoPath: "/no/such/file.mkv"})
	req := httptest.NewRequest(http.MethodPost, "/jobs", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for unreadable video_path, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestCreateJobAndGetJobRoundTrip(t *testing.T) {
	_, mux := newTestServer(t)
	videoPath := writeTempVideo(t)

	body, _ := json.Marshal(createJobRequest{VideoPath: videoPath, UseSyncnet: true})
	req := httptest.NewRequest(http.MethodPost, "/jobs", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rec.Code, rec.Body.String())
	}

	var created store.Job
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("unmarshal created job: %v", err)
	}
	if created.Status != store.StatusQueued {
		t.Fatalf("expected status queued, got %s", created.Status)
	}
	if created.AssetName != filepath.Base(videoPath) {
		t.Fatalf("expected default asset_name derived from path, got %q", created.AssetName)
	}
	if !created.UseSyncnet {
		t.Fatalf("expected use_syncnet=true to round-trip")
	}

	getReq := httptest.NewRequest(http.MethodGet, "/jobs/"+created.ID, nil)
	getRec := httptest.NewRecorder()
	mux.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("expected 200 on GET /jobs/{id}, got %d: %s", getRec.Code, getRec.Body.String())
	}
	var fetched store.Job
	if err := json.Unmarshal(getRec.Body.Bytes(), &fetched); err != nil {
		t.Fatalf("unmarshal fetched job: %v", err)
	}
	if fetched.ID != created.ID {
		t.Fatalf("expected fetched job ID %s, got %s", created.ID, fetched.ID)
	}
}

func TestGetJobReturns404ForUnknownID(t *testing.T) {
	_, mux := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/jobs/does-not-exist", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestListJobsFiltersByStatusQueryParam(t *testing.T) {
	srv, mux := newTestServer(t)
	videoPath := writeTempVideo(t)

	j1 := &store.Job{AssetName: "a.mkv", VideoPath: videoPath}
	j2 := &store.Job{AssetName: "b.mkv", VideoPath: videoPath}
	_ = srv.Store.CreateJob(context.Background(), j1)
	_ = srv.Store.CreateJob(context.Background(), j2)
	_, _ = srv.Store.ClaimNextQueuedJob(context.Background()) // j1 -> running

	req := httptest.NewRequest(http.MethodGet, "/jobs?status=running", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Jobs []store.Job `json:"jobs"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal list response: %v", err)
	}
	if len(resp.Jobs) != 1 || resp.Jobs[0].ID != j1.ID {
		t.Fatalf("expected only the running job, got %+v", resp.Jobs)
	}
}

func TestHealthzHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()

	healthzHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("failed to decode response body: %v", err)
	}
	if body["status"] != "ok" {
		t.Errorf("expected status=ok, got %q", body["status"])
	}
	if body["service"] != "syncsentry-orchestrator" {
		t.Errorf("expected service=syncsentry-orchestrator, got %q", body["service"])
	}
}
