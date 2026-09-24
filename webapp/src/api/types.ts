// Mirrors syncsentry.report.summary.FixSummary.to_dict() plus the extra
// fields the API layer adds for the webapp (job_id, input_*,
// processing_time_s, used_syncnet, download_urls).

export interface IssueReport {
  name: string; // "A/V sync" | "Captions"
  // Authoritative outcome set by the backend. Always use this rather
  // than re-deriving a verdict client-side. "undetermined" covers
  // several different backend reasons (low confidence, no trackable
  // face, no matching caption/speech onset pairs), all of which render
  // the same way here: not as "in sync".
  status: "in_sync" | "fixed" | "not_fixed" | "undetermined";
  had_issue: boolean;
  detected_offset_ms: number | null;
  fixed: boolean;
  residual_offset_ms: number | null;
  // Confidence of the residual re-measurement itself, distinct from
  // `confidence` below, which is the original detection's confidence.
  residual_confidence: number | null;
  note: string | null;
  // "A/V sync": confidence/min_confidence from the coarse or SyncNet
  // detector, method is "coarse" | "syncnet".
  // "Captions": confidence is matched_count / (matched_count +
  // unmatched_count), method is null.
  confidence: number | null;
  min_confidence: number | null;
  // "piecewise": the video does not fit one global offset. Different
  // regions were independently detected and corrected. See `segments`
  // for the per-region breakdown.
  method: "coarse" | "syncnet" | "mtdvocalist" | "piecewise" | null;
  matched_count: number | null;
  unmatched_count: number | null;
  // Only present when method === "piecewise": one entry per detected
  // region of the video, in chronological order.
  segments: IssueSegment[] | null;
}

export interface IssueSegment {
  start_s: number;
  end_s: number;
  offset_ms: number | null;
  status: "trusted" | "undetermined";
}

export interface FixResponse {
  asset_name: string;
  overall_status: "resolved" | "attention_needed";
  issues: IssueReport[];
  job_id: string;
  input_filename: string;
  input_duration_s: number | null;
  processing_time_s: number;
  used_syncnet: boolean;
  download_urls: {
    corrected_video?: string;
    corrected_captions?: string;
    // Only present when an "A/V sync" issue came back `status:
    // "undetermined"` with a real numeric `detected_offset_ms` (low
    // detector confidence, not "no face/no evidence at all"): a candidate
    // correction at that offset, rendered even though it wasn't confident
    // enough to apply automatically. Exists so a human can actually watch
    // it and decide, instead of guessing a lower `syncnetMinConfidence`
    // and re-running the full (slow) detection pass hoping it looks right.
    av_sync_preview?: string;
  };
}

export interface FixOptions {
  useSyncnet: boolean;
  avMinConfidence: number;
  syncnetMinConfidence: number;
}

export interface ApiErrorShape {
  detail: string;
}
