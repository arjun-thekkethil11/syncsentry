import type { FixOptions, FixResponse } from "./types";

// The analyzer's FastAPI server, run separately via
// `uvicorn syncsentry.api:app`. Overridable via VITE_API_BASE_URL.
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${API_BASE_URL}/healthz`, { signal: AbortSignal.timeout(4000) });
    return resp.ok;
  } catch {
    return false;
  }
}

/** Upload a video (or a .zip bundle) plus optional captions and run the
 * detect and fix pipeline. This is a single HTTP call that can take from
 * about a second to several minutes depending on the detector used. */
export async function runFix(
  video: File,
  captions: File | null,
  options: FixOptions,
  signal?: AbortSignal,
): Promise<FixResponse> {
  const form = new FormData();
  form.append("video", video);
  if (captions) form.append("captions", captions);

  const params = new URLSearchParams({
    av_min_confidence: String(options.avMinConfidence),
    use_syncnet: String(options.useSyncnet),
    syncnet_min_confidence: String(options.syncnetMinConfidence),
  });

  const resp = await fetch(`${API_BASE_URL}/v1/fix?${params.toString()}`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!resp.ok) {
    let detail = `Request failed with status ${resp.status}`;
    try {
      const body = await resp.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response was not JSON, keep the generic message
    }
    throw new ApiError(detail, resp.status);
  }

  return (await resp.json()) as FixResponse;
}

export function downloadUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}
