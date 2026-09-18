// Mirrors syncsentry.report.summary.FixSummary.to_dict() plus the extra
// fields the API layer (analyzer/syncsentry/api.py) adds for the webapp
// (job_id, input_*, processing_time_s, used_syncnet, download_urls).

export interface IssueReport {
  name: string; // "A/V sync" | "Captions"
  had_issue: boolean;
  detected_offset_ms: number | null;
  fixed: boolean;
  residual_offset_ms: number | null;
  note: string | null;
  // "A/V sync": confidence/min_confidence from the coarse or SyncNet
  // detector, method is "coarse" | "syncnet".
  // "Captions": confidence is matched_count / (matched_count +
  // unmatched_count), method is null.
  confidence: number | null;
  min_confidence: number | null;
  method: "coarse" | "syncnet" | null;
  matched_count: number | null;
  unmatched_count: number | null;
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
