// Package pyclient calls the analyzer's FastAPI HTTP API (see
// analyzer/syncsentry/api.py) -- the same /v1/fix endpoint a human would
// hit with `curl -F "video=@asset.mkv"`. This is the seam between the two
// languages: all detection/fixing logic stays in Python (see
// docs/ROADMAP.md M4 for why the Go side doesn't reimplement any of it),
// the Go side is purely "read a file off disk, forward it over HTTP, keep
// track of the result."
package pyclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Client talks to one analyzer API instance (uvicorn syncsentry.api:app).
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// New builds a Client with a generous default timeout -- SyncNet runs
// (docs/RESEARCH.md section 1e) take minutes, not seconds, per the API's
// own docstring, so a short default timeout would break the one feature
// this exists to support.
func New(baseURL string) *Client {
	return &Client{
		BaseURL:    baseURL,
		HTTPClient: &http.Client{Timeout: 10 * time.Minute},
	}
}

// FixOptions mirrors /v1/fix's query parameters.
type FixOptions struct {
	AVMinConfidence      float64
	UseSyncnet           bool
	SyncnetMinConfidence float64
}

// FixResult mirrors the JSON body /v1/fix returns -- see
// analyzer/syncsentry/report/summary.py's FixSummary.to_dict(), plus
// job_id/download_urls added in api.py itself.
type FixResult struct {
	JobID         string            `json:"job_id"`
	AssetName     string            `json:"asset_name"`
	OverallStatus string            `json:"overall_status"` // "resolved" | "attention_needed"
	Issues        []json.RawMessage `json:"issues"`
	DownloadURLs  map[string]string `json:"download_urls"`
}

// Fix uploads videoPath (and, if non-empty, captionsPath) to the
// analyzer's /v1/fix and returns the parsed report. On a non-2xx response
// it returns an error including the response body, since api.py puts the
// real failure reason in the JSON `detail` field (FastAPI's
// HTTPException), not just the status code.
func (c *Client) Fix(ctx context.Context, videoPath, captionsPath string, opts FixOptions) (*FixResult, error) {
	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)

	if err := attachFile(w, "video", videoPath); err != nil {
		return nil, fmt.Errorf("attach video: %w", err)
	}
	if captionsPath != "" {
		if err := attachFile(w, "captions", captionsPath); err != nil {
			return nil, fmt.Errorf("attach captions: %w", err)
		}
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("close multipart writer: %w", err)
	}

	url := fmt.Sprintf("%s/v1/fix?av_min_confidence=%g&use_syncnet=%t&syncnet_min_confidence=%g",
		c.BaseURL, opts.AVMinConfidence, opts.UseSyncnet, opts.SyncnetMinConfidence)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call analyzer API: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("analyzer API returned %d: %s", resp.StatusCode, respBody)
	}

	var result FixResult
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}

func attachFile(w *multipart.Writer, field, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	part, err := w.CreateFormFile(field, filepath.Base(path))
	if err != nil {
		return err
	}
	_, err = io.Copy(part, f)
	return err
}
