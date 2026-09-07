package store

import (
	"context"
	"encoding/json"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Memory is a Store implementation with no external dependency -- used by
// unit tests (handlers_test.go, worker_test.go) so they run in
// milliseconds and don't need a real Postgres instance. main.go never uses
// this; it's test-only, analogous to the analyzer's synthetic fixtures
// standing in for real content in fast tests.
type Memory struct {
	mu   sync.Mutex
	jobs map[string]*Job
}

func NewMemory() *Memory {
	return &Memory{jobs: make(map[string]*Job)}
}

func (m *Memory) CreateJob(_ context.Context, j *Job) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if j.ID == "" {
		j.ID = uuid.NewString()
	}
	now := time.Now().UTC()
	j.CreatedAt, j.UpdatedAt = now, now
	if j.Status == "" {
		j.Status = StatusQueued
	}
	cp := *j
	m.jobs[j.ID] = &cp
	return nil
}

func (m *Memory) GetJob(_ context.Context, id string) (*Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[id]
	if !ok {
		return nil, ErrNotFound
	}
	cp := *j
	return &cp, nil
}

func (m *Memory) ListJobs(_ context.Context, status *Status, limit int) ([]*Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*Job, 0, len(m.jobs))
	for _, j := range m.jobs {
		if status != nil && j.Status != *status {
			continue
		}
		cp := *j
		out = append(out, &cp)
	}
	sort.Slice(out, func(i, k int) bool { return out[i].CreatedAt.Before(out[k].CreatedAt) })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// ClaimNextQueuedJob is the in-memory equivalent of the Postgres
// implementation's `SELECT ... FOR UPDATE SKIP LOCKED`: a single mutex
// makes "find the oldest queued job and mark it running" atomic here,
// which is sufficient because this implementation is single-process
// (test-only) -- the real safety-under-concurrency guarantee only matters
// for the Postgres implementation, which real workers actually use.
func (m *Memory) ClaimNextQueuedJob(_ context.Context) (*Job, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var oldest *Job
	for _, j := range m.jobs {
		if j.Status != StatusQueued {
			continue
		}
		if oldest == nil || j.CreatedAt.Before(oldest.CreatedAt) {
			oldest = j
		}
	}
	if oldest == nil {
		return nil, nil
	}
	oldest.Status = StatusRunning
	oldest.UpdatedAt = time.Now().UTC()
	cp := *oldest
	return &cp, nil
}

func (m *Memory) CompleteJob(_ context.Context, id string, result json.RawMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[id]
	if !ok {
		return ErrNotFound
	}
	j.Status = StatusDone
	j.Result = result
	j.UpdatedAt = time.Now().UTC()
	return nil
}

func (m *Memory) FailJob(_ context.Context, id string, errMsg string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	j, ok := m.jobs[id]
	if !ok {
		return ErrNotFound
	}
	j.Status = StatusFailed
	j.Error = &errMsg
	j.UpdatedAt = time.Now().UTC()
	return nil
}
