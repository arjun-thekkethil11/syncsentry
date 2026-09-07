package pyclient

import (
	"context"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// fakeAnalyzerAPI stands in for the real uvicorn process -- it asserts the
// request actually looks like what api.py expects (multipart with a
// "video" field, the right query params) and returns a canned response,
// so this test exercises the real HTTP/multipart wire format without
// needing Python or ffmpeg installed in CI.
func fakeAnalyzerAPI(t *testing.T, wantUseSyncnet bool) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/fix" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if got := r.URL.Query().Get("use_syncnet"); got != fmt.Sprintf("%t", wantUseSyncnet) {
			t.Errorf("expected use_syncnet=%t, got %q", wantUseSyncnet, got)
		}

		mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if err != nil || mediaType != "multipart/form-data" {
			t.Fatalf("expected multipart/form-data, got %q (err %v)", r.Header.Get("Content-Type"), err)
		}
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			t.Fatalf("ParseMultipartForm: %v", err)
		}
		_ = params
		if _, ok := r.MultipartForm.File["video"]; !ok {
			t.Fatalf("expected a 'video' file field, got %+v", r.MultipartForm.File)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(FixResult{
			JobID:         "abc123",
			AssetName:     "asset.mkv",
			OverallStatus: "resolved",
			DownloadURLs:  map[string]string{"corrected_video": "/v1/jobs/abc123/files/asset.corrected.mkv"},
		})
	}))
}

func writeTempFile(t *testing.T, name string) string {
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("not a real video, just bytes for the wire format test"), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return path
}

func TestClientFixSendsMultipartAndParsesResponse(t *testing.T) {
	srv := fakeAnalyzerAPI(t, true)
	defer srv.Close()

	c := New(srv.URL)
	videoPath := writeTempFile(t, "asset.mkv")

	result, err := c.Fix(context.Background(), videoPath, "", FixOptions{
		AVMinConfidence: 0.3, UseSyncnet: true, SyncnetMinConfidence: 3.0,
	})
	if err != nil {
		t.Fatalf("Fix: %v", err)
	}
	if result.JobID != "abc123" || result.OverallStatus != "resolved" {
		t.Fatalf("unexpected result: %+v", result)
	}
	if result.DownloadURLs["corrected_video"] == "" {
		t.Fatalf("expected a corrected_video download URL, got %+v", result.DownloadURLs)
	}
}

func TestClientFixReturnsErrorOnNonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail":"Processing failed: could not decode video"}`))
	}))
	defer srv.Close()

	c := New(srv.URL)
	videoPath := writeTempFile(t, "asset.mkv")

	_, err := c.Fix(context.Background(), videoPath, "", FixOptions{})
	if err == nil {
		t.Fatal("expected an error for a 422 response, got nil")
	}
}

func TestClientFixErrorsOnMissingVideoFile(t *testing.T) {
	c := New("http://unused.invalid")
	_, err := c.Fix(context.Background(), "/no/such/file.mkv", "", FixOptions{})
	if err == nil {
		t.Fatal("expected an error for a missing video file, got nil")
	}
}
