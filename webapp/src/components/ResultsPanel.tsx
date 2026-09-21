import { useState } from "react";
import type { FixResponse } from "../api/types";
import { downloadUrl } from "../api/client";
import { IssueCard } from "./IssueCard";
import { StatusBadge } from "./StatusBadge";

interface ResultsPanelProps {
  result: FixResponse;
  onReset: () => void;
}

// Mirrors `summarizeIssue` at the whole-asset level. "resolved" only
// means the backend's own `overall_status === "resolved"`; an
// undetermined issue always makes this "needs attention".
function overallHeadline(result: FixResponse): { text: string; tone: "good" | "warn" } {
  if (result.overall_status === "resolved") {
    const anyFixed = result.issues.some((i) => i.status === "fixed");
    return {
      text: anyFixed ? "Sync issues found and corrected" : "Everything checks out",
      tone: "good",
    };
  }
  return { text: "Needs your attention", tone: "warn" };
}

export function ResultsPanel({ result, onReset }: ResultsPanelProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const headline = overallHeadline(result);
  const hasDownload = Boolean(result.download_urls.corrected_video || result.download_urls.corrected_captions);

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-sm text-slate-500 mb-1 truncate max-w-sm">{result.asset_name}</p>
            <h2 className="text-xl font-semibold text-slate-100">{headline.text}</h2>
          </div>
          <StatusBadge tone={headline.tone}>
            {headline.tone === "good" ? "resolved" : "attention needed"}
          </StatusBadge>
        </div>

        {hasDownload && (
          <div className="flex flex-wrap items-center gap-3 mt-5">
            {result.download_urls.corrected_video && (
              <a
                href={downloadUrl(result.download_urls.corrected_video)}
                download
                className="rounded-lg bg-brand-500 hover:bg-brand-400 text-white text-sm font-medium px-5 py-2.5 transition-colors"
              >
                ⬇ Download corrected video
              </a>
            )}
            {result.download_urls.corrected_captions && (
              <a
                href={downloadUrl(result.download_urls.corrected_captions)}
                download
                className="rounded-lg border border-brand-500/40 bg-brand-500/10 hover:bg-brand-500/20 text-brand-200 text-sm font-medium px-4 py-2.5 transition-colors"
              >
                ⬇ Corrected captions
              </a>
            )}
          </div>
        )}
        {!hasDownload && (
          <p className="text-sm text-slate-400 mt-4">
            No file was changed. See below for why.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {result.issues.map((issue) => (
          <IssueCard key={issue.name} issue={issue} />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => setDetailsOpen((v) => !v)}
          className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          {detailsOpen ? "Hide run details" : "Show run details"}
        </button>
        <button
          onClick={onReset}
          className="ml-auto rounded-lg border border-white/10 hover:border-white/20 text-slate-300 text-sm font-medium px-4 py-2 transition-colors"
        >
          Analyze another asset
        </button>
      </div>

      {detailsOpen && (
        <div className="flex flex-wrap items-center gap-6 text-sm rounded-xl border border-white/10 bg-white/[0.02] p-5">
          <div>
            <p className="text-slate-500 text-xs mb-1">Duration</p>
            <p className="text-slate-200 tabular-nums">
              {result.input_duration_s !== null ? `${result.input_duration_s.toFixed(1)}s` : "N/A"}
            </p>
          </div>
          <div>
            <p className="text-slate-500 text-xs mb-1">Processing time</p>
            <p className="text-slate-200 tabular-nums">{result.processing_time_s.toFixed(1)}s</p>
          </div>
          <div>
            <p className="text-slate-500 text-xs mb-1">Analysis used</p>
            <p className="text-slate-200">
              {result.used_syncnet ? "SyncNet lip-motion analysis" : "Basic audio/brightness detector"}
            </p>
          </div>
          <div>
            <p className="text-slate-500 text-xs mb-1">Job ID</p>
            <p className="text-slate-200 font-mono text-xs">{result.job_id}</p>
          </div>
        </div>
      )}
    </div>
  );
}
