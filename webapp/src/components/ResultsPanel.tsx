import type { FixResponse } from "../api/types";
import { downloadUrl } from "../api/client";
import { IssueCard } from "./IssueCard";
import { StatusBadge } from "./StatusBadge";

interface ResultsPanelProps {
  result: FixResponse;
  onReset: () => void;
}

export function ResultsPanel({ result, onReset }: ResultsPanelProps) {
  const resolved = result.overall_status === "resolved";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/[0.02] p-5">
        <div>
          <p className="text-sm text-slate-500 mb-1">Asset</p>
          <p className="font-medium text-slate-100">{result.asset_name}</p>
        </div>
        <div className="flex items-center gap-6 text-sm">
          <div>
            <p className="text-slate-500 text-xs mb-1">Duration</p>
            <p className="text-slate-200 tabular-nums">
              {result.input_duration_s !== null ? `${result.input_duration_s.toFixed(1)}s` : "—"}
            </p>
          </div>
          <div>
            <p className="text-slate-500 text-xs mb-1">Processing time</p>
            <p className="text-slate-200 tabular-nums">{result.processing_time_s.toFixed(1)}s</p>
          </div>
          <div>
            <p className="text-slate-500 text-xs mb-1">Detector</p>
            <p className="text-slate-200">{result.used_syncnet ? "SyncNet" : "coarse"}</p>
          </div>
          <StatusBadge tone={resolved ? "good" : "warn"}>
            {resolved ? "all issues resolved" : "attention needed"}
          </StatusBadge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {result.issues.map((issue) => (
          <IssueCard key={issue.name} issue={issue} />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-white/10 bg-white/[0.02] p-5">
        <span className="text-sm text-slate-400 mr-2">Downloads:</span>
        {result.download_urls.corrected_video && (
          <a
            href={downloadUrl(result.download_urls.corrected_video)}
            download
            className="rounded-lg bg-brand-500/15 hover:bg-brand-500/25 text-brand-200 text-sm font-medium px-4 py-2 transition-colors border border-brand-500/30"
          >
            ⬇ Corrected video
          </a>
        )}
        {result.download_urls.corrected_captions && (
          <a
            href={downloadUrl(result.download_urls.corrected_captions)}
            download
            className="rounded-lg bg-brand-500/15 hover:bg-brand-500/25 text-brand-200 text-sm font-medium px-4 py-2 transition-colors border border-brand-500/30"
          >
            ⬇ Corrected captions
          </a>
        )}
        <button
          onClick={onReset}
          className="ml-auto rounded-lg border border-white/10 hover:border-white/20 text-slate-300 text-sm font-medium px-4 py-2 transition-colors"
        >
          Analyze another asset
        </button>
      </div>
    </div>
  );
}
