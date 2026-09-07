-- Minimal schema, applied idempotently on startup (Postgres.Migrate) rather
-- than via a migration framework -- appropriate for this project's single
-- table; a real multi-table production service would use golang-migrate
-- or goose instead. gen_random_uuid() is a Postgres core builtin since v13,
-- no extension needed.
CREATE TABLE IF NOT EXISTS jobs (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_name              TEXT NOT NULL,
    video_path              TEXT NOT NULL,
    captions_path           TEXT,
    use_syncnet             BOOLEAN NOT NULL DEFAULT FALSE,
    syncnet_min_confidence  DOUBLE PRECISION NOT NULL DEFAULT 3.0,
    status                  TEXT NOT NULL DEFAULT 'queued',
    result                  JSONB,
    error                   TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Serves both ClaimNextQueuedJob (status = 'queued' ORDER BY created_at)
-- and ListJobs's optional status filter.
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at);
