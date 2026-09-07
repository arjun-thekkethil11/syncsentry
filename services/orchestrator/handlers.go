package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strconv"

	"github.com/arjunt/syncsentry-orchestrator/internal/store"
)

// Server holds the dependencies HTTP handlers need. A struct (rather than
// package-level globals) so tests can inject store.NewMemory() instead of
// a real Postgres connection -- see handlers_test.go.
type Server struct {
	Store store.Store
}

func healthzHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":  "ok",
		"service": "syncsentry-orchestrator",
	})
}

// createJobRequest is the POST /jobs body. VideoPath must be a path the
// orchestrator process can read -- see README.md for why this milestone
// assumes shared/local disk access rather than its own upload endpoint
// (the analyzer API already has one; duplicating it here would just be
// two copies of the same multipart-handling code).
type createJobRequest struct {
	AssetName            string  `json:"asset_name"`
	VideoPath            string  `json:"video_path"`
	CaptionsPath         string  `json:"captions_path"`
	UseSyncnet           bool    `json:"use_syncnet"`
	SyncnetMinConfidence float64 `json:"syncnet_min_confidence"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func (s *Server) createJob(w http.ResponseWriter, r *http.Request) {
	var req createJobRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if req.VideoPath == "" {
		writeError(w, http.StatusBadRequest, "video_path is required")
		return
	}
	if _, err := os.Stat(req.VideoPath); err != nil {
		writeError(w, http.StatusBadRequest, "video_path not readable: "+err.Error())
		return
	}
	if req.AssetName == "" {
		req.AssetName = filepath.Base(req.VideoPath)
	}
	if req.SyncnetMinConfidence == 0 {
		req.SyncnetMinConfidence = 3.0 // matches the CLI/API default
	}

	job := &store.Job{
		AssetName:            req.AssetName,
		VideoPath:            req.VideoPath,
		UseSyncnet:           req.UseSyncnet,
		SyncnetMinConfidence: req.SyncnetMinConfidence,
		Status:               store.StatusQueued,
	}
	if req.CaptionsPath != "" {
		job.CaptionsPath = &req.CaptionsPath
	}

	if err := s.Store.CreateJob(r.Context(), job); err != nil {
		writeError(w, http.StatusInternalServerError, "create job: "+err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, job)
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, err := s.Store.GetJob(r.Context(), id)
	if errors.Is(err, store.ErrNotFound) {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "get job: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (s *Server) listJobs(w http.ResponseWriter, r *http.Request) {
	var statusFilter *store.Status
	if raw := r.URL.Query().Get("status"); raw != "" {
		st := store.Status(raw)
		statusFilter = &st
	}
	limit := 100
	if raw := r.URL.Query().Get("limit"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}

	jobs, err := s.Store.ListJobs(r.Context(), statusFilter, limit)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "list jobs: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"jobs": jobs})
}

// routes registers all HTTP handlers on mux. Split out from main() so
// handlers_test.go can build a real *http.ServeMux against an in-memory
// Store without starting a TCP listener.
func (s *Server) routes(mux *http.ServeMux) {
	mux.HandleFunc("GET /healthz", healthzHandler)
	mux.HandleFunc("POST /jobs", s.createJob)
	mux.HandleFunc("GET /jobs", s.listJobs)
	mux.HandleFunc("GET /jobs/{id}", s.getJob)
}
