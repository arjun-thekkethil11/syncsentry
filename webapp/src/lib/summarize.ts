import type { IssueReport } from "../api/types";

export type Tone = "good" | "warn" | "neutral";

export interface PlainSummary {
  /** Short, plain-language headline. Never a bare status string. */
  headline: string;
  /** One-sentence explanation of what that means for this asset. */
  detail: string;
  tone: Tone;
  /** Whether learned lip-motion analysis actually ran, or the pipeline
   * fell back to the weaker signal-only detector. Null when not
   * applicable (e.g. the caption-sync issue). */
  analysisNote: string | null;
}

const LEARNED_METHODS = new Set(["syncnet", "mtdvocalist"]);

/**
 * Turns the backend's `status` field into a plain-language result.
 * Never says "in sync" or "synchronized" for anything other than
 * `status === "in_sync"`.
 *
 * `hasPreview` should be true when a renderable preview candidate exists
 * for this issue (currently only possible for a low-confidence A/V sync
 * result). It changes the `undetermined` copy from a generic "couldn't
 * verify" warning to a calmer "here's our best guess, go take a look"
 * message — the former reads as alarming even when we actually have a
 * good, checkable answer sitting right there for the user to confirm.
 */
export function summarizeIssue(issue: IssueReport, hasPreview = false): PlainSummary {
  const isCaptions = issue.name.toLowerCase().includes("caption");
  const analysisNote = isCaptions
    ? null
    : issue.method === null
      ? null
      : LEARNED_METHODS.has(issue.method)
        ? "Checked using learned lip-motion analysis."
        : issue.method === "piecewise"
          ? null // handled by the piecewise branch below
          : "Fell back to the basic audio/brightness detector. No reliable face or lip signal was found, so this result is less certain than a lip-motion check.";

  // A piecewise result means the video was split into independently
  // offset regions, some corrected, some possibly left undetermined.
  // This doesn't fit the single-offset language below.
  if (issue.method === "piecewise" && issue.segments) {
    const total = issue.segments.length;
    const fixedCount = issue.segments.filter(
      (s) => s.status === "trusted" && s.offset_ms !== null && Math.abs(s.offset_ms) > 0,
    ).length;
    const undeterminedCount = issue.segments.filter((s) => s.status === "undetermined").length;
    const allResolved = undeterminedCount === 0;
    return {
      headline: allResolved
        ? `Corrected sync independently across ${total} region${total === 1 ? "" : "s"} of the video`
        : `Partially corrected sync across the video`,
      detail: allResolved
        ? `This video did not have one single sync offset. ${fixedCount} of ${total} regions had their own offset, each detected and corrected separately.`
        : `This video did not have one single sync offset. ${fixedCount} of ${total} regions were corrected, but ${undeterminedCount} region${undeterminedCount === 1 ? "" : "s"} could not be reliably verified and were left unmodified. Treat this as a partial fix. See the region breakdown below.`,
      tone: allResolved ? "good" : "warn",
      analysisNote: "Checked using learned lip-motion analysis, per region.",
    };
  }

  switch (issue.status) {
    case "in_sync":
      return {
        headline: isCaptions ? "Captions are synchronized" : "Audio and video are synchronized",
        detail: "No correction was needed.",
        tone: "good",
        analysisNote,
      };
    case "fixed": {
      const ms = issue.detected_offset_ms !== null ? Math.abs(Math.round(issue.detected_offset_ms)) : null;
      return {
        headline: isCaptions
          ? ms !== null ? `Corrected a ${ms}ms caption offset` : "Corrected the caption timing"
          : ms !== null ? `Corrected a ${ms}ms sync offset` : "Corrected the sync offset",
        detail: "The fix was applied and re-checked afterward. It verified as aligned.",
        tone: "good",
        analysisNote,
      };
    }
    case "not_fixed":
      return {
        headline: isCaptions ? "Found a caption offset, but could not safely fix it" : "Found an offset, but could not safely fix it",
        detail: "A correction was attempted but did not verify afterward, so the original file was kept. Manual review recommended.",
        tone: "warn",
        analysisNote,
      };
    case "undetermined":
    default:
      if (hasPreview) {
        const ms = issue.detected_offset_ms !== null ? Math.abs(Math.round(issue.detected_offset_ms)) : null;
        return {
          headline: ms !== null ? `Found a likely ${ms}ms offset \u2014 please confirm` : "Found a likely fix \u2014 please confirm",
          detail: "The detector's confidence was below the trust threshold, so this wasn't applied automatically. A preview is ready below \u2014 take a quick look and keep it if it looks right.",
          tone: "neutral",
          analysisNote,
        };
      }
      return {
        headline: "Inconclusive",
        detail: "There wasn't enough reliable evidence either way. This is common on content a lip-motion model can't get a clear read on, and doesn't necessarily mean anything is wrong.",
        tone: "neutral",
        analysisNote,
      };
  }
}
