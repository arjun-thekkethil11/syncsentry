// Regression coverage for a real bug class (see docs/EXPERIMENT_LOG_M5.md
// entry 4, and analyzer/syncsentry/pipeline.py's `status` field): this
// component used to re-derive its "confirmed in sync" badge client-side
// from `!had_issue && confidence < min_confidence`, which missed the
// specific case where a SyncNet result clears the confidence bar but
// rests on a single *uncorroborated* whole-track reading (no independent
// windows agreed). The backend now computes an authoritative `status`
// field precisely so the frontend never needs to (and can't incorrectly)
// re-derive this call itself. These tests assert the component honors
// that field, especially the safety-critical property: nothing except
// `status === "in_sync"` may ever render as synchronized, in either the
// default plain-language view or the badge.
//
// Redesign note (M5, web-app-redesign step): the default view now leads
// with a plain-language headline/detail (`summarizeIssue`,
// `src/lib/summarize.ts`) instead of a bare status badge + raw metrics
// grid; the raw offset/confidence numbers are behind a "Show technical
// details" disclosure. Tests below assert on the always-visible headline
// text for the safety-critical invariant (never call uncertain media
// synchronized), and open the disclosure explicitly for tests that check
// the underlying numeric detail.
import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

import { IssueCard } from "./IssueCard";
import type { IssueReport } from "../api/types";

function makeIssue(overrides: Partial<IssueReport>): IssueReport {
  return {
    name: "A/V sync",
    status: "in_sync",
    had_issue: false,
    detected_offset_ms: 0,
    fixed: false,
    residual_offset_ms: null,
    residual_confidence: null,
    note: null,
    confidence: null,
    min_confidence: null,
    method: null,
    matched_count: null,
    unmatched_count: null,
    segments: null,
    ...overrides,
  };
}

function openTechnicalDetails() {
  fireEvent.click(screen.getByText(/Show technical details/));
}

describe("IssueCard plain-language result + badge", () => {
  it("only status=in_sync ever renders as synchronized (badge and headline)", () => {
    render(<IssueCard issue={makeIssue({ status: "in_sync", detected_offset_ms: 10 })} />);
    expect(screen.getByText("synchronized")).toBeInTheDocument(); // badge
    expect(screen.getByText("Audio and video are synchronized")).toBeInTheDocument(); // headline
  });

  it("never renders as synchronized for an uncorroborated near-zero SyncNet reading " +
    "even though confidence clears the trust threshold (the real bug this guards against)", () => {
    const issue = makeIssue({
      status: "undetermined",
      had_issue: false,
      detected_offset_ms: -0,
      confidence: 3.4, // clears min_confidence
      min_confidence: 3.0,
      method: "syncnet",
      note: "uncorroborated near-zero detection [syncnet]: raw estimate -0ms, confidence 3.40 clears "
        + "the trust threshold but only one whole-track reading supports it",
    });
    render(<IssueCard issue={issue} />);
    expect(screen.queryByText("synchronized")).not.toBeInTheDocument();
    expect(screen.getByText("unverified")).toBeInTheDocument();
    // Plain-language headline must be visible by default, not buried
    // behind the technical details disclosure.
    expect(screen.getByText("Unable to verify sync safely")).toBeInTheDocument();
    expect(screen.getByText(/treat it as unverified/i)).toBeInTheDocument();

    openTechnicalDetails();
    expect(screen.getByText(/uncorroborated near-zero detection/)).toBeInTheDocument();
  });

  it("shows a plain-language 'corrected' headline with a residual offset behind details for status=fixed", () => {
    const issue = makeIssue({
      status: "fixed",
      had_issue: true,
      fixed: true,
      detected_offset_ms: 1800,
      residual_offset_ms: 0,
      residual_confidence: 9.07,
      confidence: 9.07,
      min_confidence: 3.0,
      method: "syncnet",
    });
    render(<IssueCard issue={issue} />);
    expect(screen.getByText("corrected")).toBeInTheDocument();
    expect(screen.getByText(/Corrected a 1800ms sync offset/)).toBeInTheDocument();
    expect(screen.getByText(/learned lip-motion/)).toBeInTheDocument();

    openTechnicalDetails();
    expect(screen.getByText("Residual offset (after fix)")).toBeInTheDocument();
  });

  it("never renders as synchronized when verification rejects the correction (status=not_fixed)", () => {
    const issue = makeIssue({
      status: "not_fixed",
      had_issue: true,
      fixed: false,
      detected_offset_ms: 1640,
      residual_offset_ms: 2240,
      residual_confidence: 1.07,
      confidence: 4.3,
      min_confidence: 3.0,
      method: "syncnet",
      note: "residual re-check was itself low-confidence",
    });
    render(<IssueCard issue={issue} />);
    expect(screen.getByText("not fixed")).toBeInTheDocument();
    expect(screen.queryByText("synchronized")).not.toBeInTheDocument();
    expect(screen.getByText(/could not safely fix it/)).toBeInTheDocument();
  });

  it("shows 'unverified' (never synchronized) when no matching caption/speech onsets were found", () => {
    const issue = makeIssue({
      name: "Captions",
      status: "undetermined",
      had_issue: false,
      detected_offset_ms: null,
      note: "no matching speech onsets found",
      matched_count: 0,
      unmatched_count: 5,
    });
    render(<IssueCard issue={issue} />);
    expect(screen.getByText("unverified")).toBeInTheDocument();
    expect(screen.queryByText("synchronized")).not.toBeInTheDocument();

    openTechnicalDetails();
    expect(screen.getByText(/caption cues had a matching speech onset nearby/)).toBeInTheDocument();
  });

  it("surfaces a fallback warning in plain language when the coarse (non-learned) detector was used", () => {
    const issue = makeIssue({
      status: "undetermined",
      method: "coarse",
      confidence: 0.08,
      min_confidence: 0.3,
      detected_offset_ms: 120,
      note: "low-confidence detection",
    });
    render(<IssueCard issue={issue} />);
    expect(screen.getByText(/Fell back to the basic audio\/brightness detector/)).toBeInTheDocument();
  });

  // Piecewise coverage (M5 entries 9-10, syncsentry/lipsync/piecewise_offset.py):
  // a video that doesn't fit one global offset gets a per-region
  // breakdown instead. The safety-critical property here is the same
  // "never overclaim" one as above, just at the aggregate level: any
  // undetermined region must keep the overall headline/tone honest, not
  // just be a footnote in the (hidden) note.
  it("shows a fully-resolved piecewise headline plus a visible per-region breakdown when every region was corrected", () => {
    const issue = makeIssue({
      status: "fixed",
      had_issue: true,
      fixed: true,
      method: "piecewise",
      detected_offset_ms: null,
      note: "content doesn't fit a single global A/V offset: found 2 region(s)...",
      segments: [
        { start_s: 0, end_s: 6, offset_ms: 0, status: "trusted" },
        { start_s: 6, end_s: 12, offset_ms: 300, status: "trusted" },
      ],
    });
    render(<IssueCard issue={issue} />);
    expect(screen.getByText("corrected")).toBeInTheDocument();
    expect(screen.getByText(/Corrected sync independently across 2 regions/)).toBeInTheDocument();
    expect(screen.getByText("0.0s to 6.0s")).toBeInTheDocument();
    expect(screen.getByText("already in sync")).toBeInTheDocument();
    expect(screen.getByText("corrected 300ms")).toBeInTheDocument();
  });

  it("never claims full resolution for a partially-undetermined piecewise result", () => {
    const issue = makeIssue({
      status: "not_fixed",
      had_issue: true,
      fixed: false,
      method: "piecewise",
      detected_offset_ms: null,
      note: "content doesn't fit a single global A/V offset...; 83% of the video's duration could not be "
        + "independently confirmed either way and is left unmodified",
      segments: [
        { start_s: 0, end_s: 6, offset_ms: 0, status: "trusted" },
        { start_s: 6, end_s: 66, offset_ms: null, status: "undetermined" },
        { start_s: 66, end_s: 72, offset_ms: 300, status: "trusted" },
      ],
    });
    render(<IssueCard issue={issue} />);
    expect(screen.queryByText("synchronized")).not.toBeInTheDocument();
    expect(screen.queryByText("corrected")).not.toBeInTheDocument();
    expect(screen.getByText("not fixed")).toBeInTheDocument();
    expect(screen.getByText(/Partially corrected sync/)).toBeInTheDocument();
    expect(screen.getByText(/1 region could not be reliably verified/)).toBeInTheDocument();
    // The breakdown must be visible without opening the technical
    // details disclosure. It is the result, not a footnote.
    expect(screen.getAllByText("unverified").length).toBeGreaterThan(0);
  });
});
