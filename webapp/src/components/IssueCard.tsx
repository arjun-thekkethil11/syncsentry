import type { IssueReport } from "../api/types";
import { OffsetBar } from "./OffsetBar";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { StatusBadge } from "./StatusBadge";

interface IssueCardProps {
  issue: IssueReport;
}

export function IssueCard({ issue }: IssueCardProps) {
  // Below-threshold confidence means "we genuinely can't tell", not
  // "confirmed in sync" -- these are different claims and the badge
  // shouldn't blur them together (see docs/RESEARCH.md on why this
  // distinction exists at all).
  const isUndetermined =
    !issue.had_issue &&
    issue.confidence !== null &&
    issue.min_confidence !== null &&
    issue.confidence < issue.min_confidence;

  const badge = isUndetermined ? (
    <StatusBadge tone="neutral">undetermined</StatusBadge>
  ) : !issue.had_issue ? (
    <StatusBadge tone="good">confirmed in sync</StatusBadge>
  ) : issue.fixed ? (
    <StatusBadge tone="good">fixed</StatusBadge>
  ) : (
    <StatusBadge tone="warn">not fixed</StatusBadge>
  );

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-slate-100">{issue.name}</h3>
          {issue.method && (
            <span className="text-[10px] uppercase tracking-wide text-slate-500 border border-white/10 rounded px-1.5 py-0.5">
              {issue.method}
            </span>
          )}
        </div>
        {badge}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto] gap-6 items-center">
        <div className="space-y-4">
          <OffsetBar label="Detected offset" offsetMs={issue.detected_offset_ms} />
          {issue.had_issue && (
            <OffsetBar
              label={issue.fixed ? "Residual offset (after fix)" : "Residual offset"}
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
        <p className="mt-4 text-xs text-slate-500">
          {issue.matched_count} of {issue.matched_count + issue.unmatched_count} caption cues had a
          matching speech onset nearby.
        </p>
      )}

      {issue.note && (
        <p className="mt-4 text-xs text-slate-400 bg-white/[0.03] border border-white/5 rounded-lg px-3 py-2 leading-relaxed">
          {issue.note}
        </p>
      )}
    </div>
  );
}
