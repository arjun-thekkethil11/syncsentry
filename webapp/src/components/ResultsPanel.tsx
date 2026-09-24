import { useEffect, useState } from "react";
import type { FixResponse } from "../api/types";
import { downloadUrl, triggerDownload } from "../api/client";
import { IssueCard } from "./IssueCard";
import { StatusBadge } from "./StatusBadge";

interface ResultsPanelProps {
  result: FixResponse;
  // The original `File` the user uploaded, kept around by `Analyze` (not
  // re-fetched from the server) purely so this panel can render an
  // in-browser preview of it without a round trip. `null` if for some
  // reason the caller doesn't have it; the "before" side of the
  // comparison degrades gracefully in that case.
  inputVideoFile: File | null;
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

function filenameFromPath(path: string): string {
  return path.split("/").pop() || "download";
}

// Cross-origin links silently ignore the `download` attribute, so a
// plain click would navigate the whole tab to the raw file instead of
// downloading it. Fetch it as a blob and download that instead; fall
// back to the plain link (opens in a new tab) if the fetch fails.
function handleDownloadClick(path: string) {
  return async (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    try {
      await triggerDownload(path, filenameFromPath(path));
    } catch {
      window.open(downloadUrl(path), "_blank", "noopener,noreferrer");
    }
  };
}

function VideoSlot({ label, tone, url }: { label: string; tone: "neutral" | "good" | "warn"; url: string | null }) {
  const labelClass =
    tone === "good" ? "text-emerald-400" : tone === "warn" ? "text-amber-400" : "text-slate-500";
  return (
    <div>
      <p className={`text-xs mb-2 ${labelClass}`}>{label}</p>
      {url ? (
        <video src={url} controls className="w-full aspect-video rounded-lg border border-white/10 bg-black" />
      ) : (
        <div className="aspect-video rounded-lg border border-white/10 bg-black/40 flex items-center justify-center text-xs text-slate-600 px-4 text-center">
          Not available
        </div>
      )}
    </div>
  );
}

export function ResultsPanel({ result, inputVideoFile, onReset }: ResultsPanelProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string | null>(null);

  // Object URLs are cheap (no re-encoding, just a reference into the
  // already-in-memory File) but must be revoked or they leak for the
  // life of the tab; tying creation/cleanup to `inputVideoFile` identity
  // keeps exactly one alive at a time.
  useEffect(() => {
    if (!inputVideoFile) {
      setInputPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(inputVideoFile);
    setInputPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [inputVideoFile]);

  const headline = overallHeadline(result);
  const correctedUrl = result.download_urls.corrected_video;
  const previewUrl = result.download_urls.av_sync_preview;
  const fixedApplied = result.issues.some((i) => i.status === "fixed");
  const hasDownload = Boolean(correctedUrl || result.download_urls.corrected_captions);

  // The issue this preview's offset/direction actually came from, so the
  // copy below can show real numbers instead of a bare video.
  const previewIssue = result.issues.find((i) => i.name === "A/V sync" && i.status === "undetermined");

  // What to show as the "second" video in the before/after comparison.
  // Never the same clip twice: when nothing changed and there's no
  // low-confidence candidate either, the right slot stays empty rather
  // than showing an identical copy of the original.
  const secondVideo: { url: string; label: string; tone: "good" | "warn" } | null = previewUrl
    ? { url: downloadUrl(previewUrl), label: "Candidate preview (unverified)", tone: "warn" }
    : fixedApplied && correctedUrl
      ? { url: downloadUrl(correctedUrl), label: "Corrected", tone: "good" }
      : null;

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
            {correctedUrl && (
              <a
                href={downloadUrl(correctedUrl)}
                download
                onClick={handleDownloadClick(correctedUrl)}
                className="rounded-lg bg-brand-500 hover:bg-brand-400 text-white text-sm font-medium px-5 py-2.5 transition-colors"
              >
                ⬇ Download corrected video
              </a>
            )}
            {result.download_urls.corrected_captions && (
              <a
                href={downloadUrl(result.download_urls.corrected_captions)}
                download
                onClick={handleDownloadClick(result.download_urls.corrected_captions)}
                className="rounded-lg border border-brand-500/40 bg-brand-500/10 hover:bg-brand-500/20 text-brand-200 text-sm font-medium px-4 py-2.5 transition-colors"
              >
                ⬇ Corrected captions
              </a>
            )}
          </div>
        )}
        {!hasDownload && !previewUrl && (
          <p className="text-sm text-slate-400 mt-4">No file was changed. See below for why.</p>
        )}
      </div>

      {(inputPreviewUrl || secondVideo) && (
        <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6">
          <p className="text-sm font-medium text-slate-200 mb-4">Before / After</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <VideoSlot label="Original" tone="neutral" url={inputPreviewUrl} />
            <VideoSlot
              label={secondVideo?.label ?? "Already in sync"}
              tone={secondVideo?.tone ?? "good"}
              url={secondVideo?.url ?? null}
            />
          </div>

          {previewUrl && (
            <>
              <p className="text-xs text-amber-300/80 mt-4 leading-relaxed">
                {previewIssue?.detected_offset_ms != null
                  ? `The detector's best guess is a ${previewIssue.detected_offset_ms > 0 ? "+" : ""}${Math.round(
                      previewIssue.detected_offset_ms,
                    )}ms offset, but its own confidence (${previewIssue.confidence?.toFixed(2)}) was below the ` +
                    `trust threshold (${previewIssue.min_confidence?.toFixed(2)}), so nothing was applied ` +
                    `automatically. `
                  : ""}
                Watch/listen to both above and decide for yourself whether the candidate looks
                right &mdash; no confidence number, from any detector, is a substitute for actually
                checking.
              </p>
              <div className="flex flex-wrap items-center gap-3 mt-4">
                <a
                  href={downloadUrl(previewUrl)}
                  download
                  onClick={handleDownloadClick(previewUrl)}
                  className="rounded-lg bg-amber-500/90 hover:bg-amber-400 text-slate-950 text-sm font-medium px-5 py-2.5 transition-colors"
                >
                  ⬇ Download this candidate correction
                </a>
                <span className="text-xs text-slate-500">Only keep it if it looks right to you</span>
              </div>
            </>
          )}
        </div>
      )}

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
