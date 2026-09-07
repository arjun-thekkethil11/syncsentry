package store

import (
	"context"
	"database/sql"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"

	_ "github.com/jackc/pgx/v5/stdlib" // registers the "pgx" sql.DB driver
)

//go:embed schema.sql
var schemaSQL string

// Postgres is the real Store implementation, used by main.go. Concurrency
// safety for job claiming comes from Postgres itself (SELECT ... FOR
// UPDATE SKIP LOCKED in ClaimNextQueuedJob) -- multiple orchestrator
// processes/workers can run against the same database without a
// distributed lock of their own.
type Postgres struct {
	db *sql.DB
}

// Open connects to Postgres (dsn e.g.
// "postgres://user:pass@localhost:5432/syncsentry?sslmode=disable") and
// applies the schema. Safe to call on every startup -- schema.sql is all
// `CREATE ... IF NOT EXISTS`.
func Open(ctx context.Context, dsn string) (*Postgres, error) {
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, fmt.Errorf("open postgres: %w", err)
	}
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("ping postgres: %w", err)
	}
	if _, err := db.ExecContext(ctx, schemaSQL); err != nil {
		return nil, fmt.Errorf("apply schema: %w", err)
	}
	return &Postgres{db: db}, nil
}

func (p *Postgres) Close() error { return p.db.Close() }

const jobColumns = `id, asset_name, video_path, captions_path, use_syncnet, syncnet_min_confidence, status, result, error, created_at, updated_at`

func scanJob(row interface{ Scan(...any) error }) (*Job, error) {
	var j Job
	var captionsPath sql.NullString
	var result []byte
	var errMsg sql.NullString
	if err := row.Scan(&j.ID, &j.AssetName, &j.VideoPath, &captionsPath, &j.UseSyncnet,
		&j.SyncnetMinConfidence, &j.Status, &result, &errMsg, &j.CreatedAt, &j.UpdatedAt); err != nil {
		return nil, err
	}
	if captionsPath.Valid {
		j.CaptionsPath = &captionsPath.String
	}
	if len(result) > 0 {
		j.Result = json.RawMessage(result)
	}
	if errMsg.Valid {
		j.Error = &errMsg.String
	}
	return &j, nil
}

func (p *Postgres) CreateJob(ctx context.Context, j *Job) error {
	if j.Status == "" {
		j.Status = StatusQueued
	}
	row := p.db.QueryRowContext(ctx, `
		INSERT INTO jobs (asset_name, video_path, captions_path, use_syncnet, syncnet_min_confidence, status)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING `+jobColumns,
		j.AssetName, j.VideoPath, j.CaptionsPath, j.UseSyncnet, j.SyncnetMinConfidence, j.Status,
	)
	created, err := scanJob(row)
	if err != nil {
		return fmt.Errorf("create job: %w", err)
	}
	*j = *created
	return nil
}

func (p *Postgres) GetJob(ctx context.Context, id string) (*Job, error) {
	row := p.db.QueryRowContext(ctx, `SELECT `+jobColumns+` FROM jobs WHERE id = $1`, id)
	j, err := scanJob(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("get job: %w", err)
	}
	return j, nil
}

func (p *Postgres) ListJobs(ctx context.Context, status *Status, limit int) ([]*Job, error) {
	if limit <= 0 {
		limit = 100
	}
	var rows *sql.Rows
	var err error
	if status != nil {
		rows, err = p.db.QueryContext(ctx,
			`SELECT `+jobColumns+` FROM jobs WHERE status = $1 ORDER BY created_at DESC LIMIT $2`,
			string(*status), limit)
	} else {
		rows, err = p.db.QueryContext(ctx,
			`SELECT `+jobColumns+` FROM jobs ORDER BY created_at DESC LIMIT $1`, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("list jobs: %w", err)
	}
	defer rows.Close()

	var out []*Job
	for rows.Next() {
		j, err := scanJob(rows)
		if err != nil {
			return nil, fmt.Errorf("scan job: %w", err)
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

// ClaimNextQueuedJob atomically picks the oldest queued job and marks it
// running in one statement -- the subquery's `FOR UPDATE SKIP LOCKED` means
// concurrent callers (multiple worker goroutines, or multiple orchestrator
// processes) each get a different job rather than racing on the same one,
// with no explicit BEGIN/COMMIT needed since a single statement is already
// its own transaction.
func (p *Postgres) ClaimNextQueuedJob(ctx context.Context) (*Job, error) {
	row := p.db.QueryRowContext(ctx, `
		UPDATE jobs
		SET status = 'running', updated_at = now()
		WHERE id = (
			SELECT id FROM jobs
			WHERE status = 'queued'
			ORDER BY created_at
			FOR UPDATE SKIP LOCKED
			LIMIT 1
		)
		RETURNING `+jobColumns,
	)
	j, err := scanJob(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil // no queued work -- not an error
	}
	if err != nil {
		return nil, fmt.Errorf("claim job: %w", err)
	}
	return j, nil
}

func (p *Postgres) CompleteJob(ctx context.Context, id string, result json.RawMessage) error {
	res, err := p.db.ExecContext(ctx,
		`UPDATE jobs SET status = 'done', result = $2, updated_at = now() WHERE id = $1`, id, result)
	return checkRowsAffected(res, err)
}

func (p *Postgres) FailJob(ctx context.Context, id string, errMsg string) error {
	res, err := p.db.ExecContext(ctx,
		`UPDATE jobs SET status = 'failed', error = $2, updated_at = now() WHERE id = $1`, id, errMsg)
	return checkRowsAffected(res, err)
}

func checkRowsAffected(res sql.Result, err error) error {
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return ErrNotFound
	}
	return nil
}
