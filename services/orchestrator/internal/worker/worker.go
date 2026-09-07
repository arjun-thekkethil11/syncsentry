// Package worker implements the poll loop that turns queued Jobs into
// done/failed ones by calling the analyzer's HTTP API (docs/ROADMAP.md M4).
//
// Scaling model: each Worker.Run call is one independent poll loop. main.go
// starts N of them as goroutines; Postgres's SKIP LOCKED claim query (see
// internal/store/postgres.go) is what makes that safe -- no coordination
// between the N loops is needed, they just race to claim distinct rows.
// The same property holds across multiple orchestrator *processes* against
// the same database, which is the actual point of using Postgres here
// instead of an in-memory queue: catalog-scale batch runs need more
// throughput than one process's goroutines can give a CPU-bound Python
// subprocess call.
package worker

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/arjunt/syncsentry-orchestrator/internal/pyclient"
	"github.com/arjunt/syncsentry-orchestrator/internal/store"
)

// defaultAVMinConfidence matches the CLI's own default (see
// analyzer/syncsentry/cli.py) -- the orchestrator doesn't expose a way to
// override it per-job today; that's a reasonable next increment, not
// something this milestone needs.
const defaultAVMinConfidence = 0.3

type Worker struct {
	Store        store.Store
	PyClient     *pyclient.Client
	PollInterval time.Duration
}

func New(s store.Store, pc *pyclient.Client, pollInterval time.Duration) *Worker {
	return &Worker{Store: s, PyClient: pc, PollInterval: pollInterval}
}

// Run polls Store for queued jobs until ctx is cancelled. Blocking; call it
// in a goroutine (main.go starts one or more).
func (w *Worker) Run(ctx context.Context) {
	ticker := time.NewTicker(w.PollInterval)
	defer ticker.Stop()

	w.tick(ctx) // don't wait a full interval before the first poll
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			w.tick(ctx)
		}
	}
}

func (w *Worker) tick(ctx context.Context) {
	job, err := w.Store.ClaimNextQueuedJob(ctx)
	if err != nil {
		log.Printf("worker: claim job: %v", err)
		return
	}
	if job == nil {
		return // nothing queued right now
	}
	w.process(ctx, job)
}

func (w *Worker) process(ctx context.Context, job *store.Job) {
	log.Printf("worker: processing job %s (%s, use_syncnet=%t)", job.ID, job.AssetName, job.UseSyncnet)

	captionsPath := ""
	if job.CaptionsPath != nil {
		captionsPath = *job.CaptionsPath
	}

	result, err := w.PyClient.Fix(ctx, job.VideoPath, captionsPath, pyclient.FixOptions{
		AVMinConfidence:      defaultAVMinConfidence,
		UseSyncnet:           job.UseSyncnet,
		SyncnetMinConfidence: job.SyncnetMinConfidence,
	})
	if err != nil {
		log.Printf("worker: job %s failed: %v", job.ID, err)
		if failErr := w.Store.FailJob(ctx, job.ID, err.Error()); failErr != nil {
			log.Printf("worker: mark job %s failed: %v", job.ID, failErr)
		}
		return
	}

	raw, err := json.Marshal(result)
	if err != nil {
		log.Printf("worker: marshal result for job %s: %v", job.ID, err)
		_ = w.Store.FailJob(ctx, job.ID, "internal error marshalling result: "+err.Error())
		return
	}
	if err := w.Store.CompleteJob(ctx, job.ID, raw); err != nil {
		log.Printf("worker: mark job %s done: %v", job.ID, err)
	} else {
		log.Printf("worker: job %s done (%s)", job.ID, result.OverallStatus)
	}
}
