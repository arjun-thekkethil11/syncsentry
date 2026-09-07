// Package store defines the job model + persistence interface for the
// orchestrator's catalog batch runner (docs/ROADMAP.md M4).
//
// Two implementations exist (postgres.go, memory.go) behind the same
// interface: handlers and the worker only ever depend on Store, so unit
// tests can use the fast in-memory one while main.go wires up the real
// Postgres one. This mirrors the analyzer's own "fake vs. real" pattern
// (e.g. real_content fixtures gated behind an availability check) applied
// to the Go side.
package store

import (
	"context"
	"encoding/json"
	"time"
)

// Status is a job's lifecycle state. Deliberately a small closed set --
// see docs/ROADMAP.md M4 for why a full workflow engine is out of scope
// here.
type Status string

const (
	StatusQueued  Status = "queued"
	StatusRunning Status = "running"
	StatusDone    Status = "done"
	StatusFailed  Status = "failed"
)

// Job is one asset's trip through the pipeline: submitted with a video (and
// optional captions) path, processed by exactly one worker, and left with
// either a Result (the analyzer's report.json, verbatim) or an Error.
type Job struct {
	ID                   string          `json:"id"`
	AssetName            string          `json:"asset_name"`
	VideoPath            string          `json:"video_path"`
	CaptionsPath         *string         `json:"captions_path,omitempty"`
	UseSyncnet           bool            `json:"use_syncnet"`
	SyncnetMinConfidence float64         `json:"syncnet_min_confidence"`
	Status               Status          `json:"status"`
	Result               json.RawMessage `json:"result,omitempty"`
	Error                *string         `json:"error,omitempty"`
	CreatedAt            time.Time       `json:"created_at"`
	UpdatedAt            time.Time       `json:"updated_at"`
}

// Store is the persistence boundary. ClaimNextQueuedJob is the one method
// that has to be atomic under concurrent workers -- see postgres.go for how
// (SELECT ... FOR UPDATE SKIP LOCKED) and memory.go for the single-mutex
// equivalent.
type Store interface {
	CreateJob(ctx context.Context, j *Job) error
	GetJob(ctx context.Context, id string) (*Job, error)
	ListJobs(ctx context.Context, status *Status, limit int) ([]*Job, error)
	ClaimNextQueuedJob(ctx context.Context) (*Job, error)
	CompleteJob(ctx context.Context, id string, result json.RawMessage) error
	FailJob(ctx context.Context, id string, errMsg string) error
}

// ErrNotFound is returned by GetJob when no job with that ID exists.
var ErrNotFound = &notFoundError{}

type notFoundError struct{}

func (*notFoundError) Error() string { return "job not found" }
