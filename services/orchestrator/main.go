// Command orchestrator is the SyncSentry Go service: catalog batch runner +
// REST API over Tier 1/Tier 2 sync-drift reports (see /docs/ROADMAP.md, M4).
//
// Current scope (M4 in progress): a minimal, real HTTP service with a health
// endpoint and a job-status model, wired up now so the queue/DB/API pieces
// land incrementally against a working binary rather than all at once.
// Deliberately not a placeholder: `go build && ./bin/orchestrator` runs.
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"
)

// JobStatus mirrors the analyzer's report schema closely enough that the
// Go side can track job lifecycle without needing to understand detector
// internals (those stay in Python; see analyzer/syncsentry/report).
type JobStatus struct {
	AssetID   string    `json:"asset_id"`
	State     string    `json:"state"` // queued | running | done | failed
	UpdatedAt time.Time `json:"updated_at"`
}

func healthzHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":  "ok",
		"service": "syncsentry-orchestrator",
	})
}

func main() {
	addr := os.Getenv("SYNCSENTRY_ORCHESTRATOR_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", healthzHandler)

	log.Printf("syncsentry-orchestrator listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}
