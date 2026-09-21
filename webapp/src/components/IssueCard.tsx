import { useState } from "react";
import type { IssueReport } from "../api/types";
import { OffsetBar } from "./OffsetBar";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { StatusBadge } from "./StatusBadge";
import { summarizeIssue } from "../lib/summarize";

interface IssueCardProps {
  issue: IssueReport;
}

const badgeTone = { good: "good", warn: "warn", neutral: "neutral" } as const;

export function IssueCard({ issue }: IssueCardProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  // Always defer to the backend's own `status` field rather than
  // re-deriving a verdict here. The backend has the full context
  // (corroboration counts, etc.) needed to make this call correctly.
  const summary = summarizeIssue(issue);

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div>
          <p className="text-xs text-slate-500 mb-1">{issue.name}</p>
          <h3 className="font-semibold text-slate-100 leading-snug">{summary.headline}</h3>
        </div>
        <StatusBadge tone={badgeTone[summary.tone]}>
          {issue.status === "in_sync"
            ? "synchronized"
            : issue.status === "fixed"
              ? "corrected"
              : issue.status === "not_fixed"
                ? "not fixed"
                : "unverified"}
        </StatusBadge>
      </div>
      <p className="text-sm text-slate-400 leading-relaxed">{summary.detail}</p>
      {summary.analysisNote && (
        <p className="mt-2 text-xs text-slate-500 leading-relaxed">{summary.analysisNote}</p>
      )}

      {/* Per-region breakdown for piecewise results. Shown by default,
          not behind the technical-details toggle, since which regions
          were fixed vs. left undetermined is the result here. */}
      {issue.method === "piecewise" && issue.segments && issue.segments.length > 0 && (
        <ul className="mt-3 space-y-1">
          {issue.segments.map((seg, i) => (
            <li
              key={i}
              className="flex items-center justify-between gap-3 text-xs rounded-lg bg-white/[0.03] border border-white/5 px-3 py-1.5"
            >
              <span className="text-slate-400 tabular-nums">
                {seg.start_s.toFixed(1)}s to {seg.end_s.toFixed(1)}s
              </span>
              <span className={seg.status === "trusted" ? "text-emerald-400" : "text-amber-400"}>
                {seg.status === "trusted"
                  ? seg.offset_ms !== null && Math.abs(seg.offset_ms) > 0
                    ? `corrected ${Math.round(seg.offset_ms)}ms`
                    : "already in sync"
                  : "unverified"}
              </span>
            </li>
          ))}
        </ul>
      )}

      <button
        type="button"
        onClick={() => setDetailsOpen((v) => !v)}
        className="mt-4 text-xs text-slate-500 hover:text-slate-300 transition-colors"
      >
        {detailsOpen ? "Hide technical details" : "Show technical details"}
      </button>

      {detailsOpen && (
        <div className="mt-4 pt-4 border-t border-white/5 space-y-4">
          {issue.method && (
            <p className="text-[10px] uppercase tracking-wide text-slate-500 border border-white/10 rounded px-1.5 py-0.5 inline-block">
              method: {issue.method}
            </p>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto] gap-6 items-center">
            <div className="space-y-4">
              <OffsetBar label="Detected offset" offsetMs={issue.detected_offset_ms} />
              {(issue.status === "fixed" || issue.status === "not_fixed") && (
                <OffsetBar
                  label={issue.status === "fixed" ? "Residual offset (after fix)" : "Residual offset"}
                  offsetMs={issue.residual_offset_ms}
                  scaleMs={100}
                />
              )}
            </div>
            {issue.confidence !== null && issue.min_confidence !== null && (
              <ConfidenceGauge confidence={issue.confidence} threshold={issue.min_confidence} />
            )}
          </div>

          {issue.matched_count !== null && issue.unmatched_count !== null && (
            <p className="text-xs text-slate-500">
              {issue.matched_count} of {issue.matched_count + issue.unmatched_count} caption cues had
              a matching speech onset nearby.
            </p>
          )}

          {issue.note && (
            <p className="text-xs text-slate-400 bg-white/[0.03] border border-white/5 rounded-lg px-3 py-2 leading-relaxed">
              {issue.note}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
