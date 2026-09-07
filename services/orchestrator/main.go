// Command orchestrator is the SyncSentry Go service: catalog batch runner +
// REST API over the analyzer's detect/fix pipeline (see docs/ROADMAP.md,
// M4).
//
// Job lifecycle: POST /jobs enqueues a video (by path) into Postgres;
// one or more worker goroutines (internal/worker) poll for queued jobs and
// call the analyzer's HTTP API (internal/pyclient) to actually process
// them; GET /jobs/{id} and GET /jobs report status/results. All
// detection/fixing logic stays in Python -- this service is purely queueing,
// persistence, and orchestration, so a catalog-scale batch run can process
// many assets concurrently across worker goroutines/processes without the
// Python side needing to know anything about scheduling.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/arjunt/syncsentry-orchestrator/internal/pyclient"
	"github.com/arjunt/syncsentry-orchestrator/internal/store"
	"github.com/arjunt/syncsentry-orchestrator/internal/worker"
)

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func main() {
	addr := getenv("SYNCSENTRY_ORCHESTRATOR_ADDR", ":8080")
	dsn := getenv("SYNCSENTRY_DATABASE_URL", "postgres://localhost:5432/syncsentry?sslmode=disable")
	pythonAPIURL := getenv("SYNCSENTRY_PYTHON_API_URL", "http://localhost:8000")
	workerCount := getenvInt("SYNCSENTRY_WORKER_COUNT", 2)
	pollInterval := time.Duration(getenvInt("SYNCSENTRY_POLL_INTERVAL_MS", 2000)) * time.Millisecond

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	pg, err := store.Open(ctx, dsn)
	if err != nil {
		log.Fatalf("connect to postgres: %v", err)
	}
	defer pg.Close()
	log.Printf("connected to postgres, schema applied")

	pc := pyclient.New(pythonAPIURL)

	for i := 0; i < workerCount; i++ {
		w := worker.New(pg, pc, pollInterval)
		go w.Run(ctx)
	}
	log.Printf("started %d worker(s), polling every %s, analyzer API at %s", workerCount, pollInterval, pythonAPIURL)

	srv := &Server{Store: pg}
	mux := http.NewServeMux()
	srv.routes(mux)

	httpSrv := &http.Server{Addr: addr, Handler: mux}
	go func() {
		<-ctx.Done()
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		_ = httpSrv.Shutdown(shutdownCtx)
	}()

	log.Printf("syncsentry-orchestrator listening on %s", addr)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
