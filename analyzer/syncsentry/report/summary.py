"""The small, human-readable report SyncSentry produces for one asset.

Deliberately not a dump of every internal measurement, just the handful of
numbers a content-ops person or an engineer skimming a CI log actually
needs: what was wrong, what we did about it, and proof it worked (the
residual after fixing). Per-cue detail, confidence scores, etc. still live
in the JSON detection/fix results for anyone who needs to dig deeper, but
they aren't in this summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IssueSummary:
    name: str  # "A/V sync" | "Captions"
    had_issue: bool
    detected_offset_ms: float | None
    fixed: bool
    residual_offset_ms: float | None
    # Explicit, caller-set status: one of "in_sync" | "fixed" |
    # "not_fixed" | "undetermined". Exists because `had_issue` alone
    # conflates two very different situations under `had_issue=False`:
    # "we positively confirmed this is in sync" and "we genuinely don't
    # have enough evidence to say either way" (low detector confidence,
    # an uncorroborated near-zero reading, no trackable face, no matching
    # caption/speech-onset pairs, etc). Treating both as "resolved" in
    # `SyncFixSummary.all_resolved` would let a low-confidence detection
    # report `overall_status: "resolved"` even while its own note says
    # "too low to trust". `all_resolved` keys off this field instead:
    # only "in_sync" and "fixed" count as resolved; "undetermined" and
    # "not_fixed" both surface as `attention_needed`, regardless of
    # `had_issue`.
    #
    # Deliberately has no default: this is the one field that exists
    # specifically to stop a false "in sync" claim, so a constructor call
    # that forgets to set it must fail loudly (`TypeError` at call time)
    # rather than silently defaulting to the single most dangerous value.
    # Must therefore stay ordered before any field below that does have a
    # default (dataclass field-ordering rule); moved above `note` for
    # exactly this reason.
    status: str
    note: str = ""
    # Confidence of the *residual* re-measurement itself (distinct from
    # `confidence` below, which is the original detection's confidence,
    # the one that decided whether to apply a correction in the first
    # place). An un-gated residual number with no confidence attached
    # reads as authoritative ("the fix left a -2240ms residual") when it
    # may just be a low-confidence, discardable re-measurement. `None`
    # for "A/V sync" issues that were never fixed (no residual was
    # computed) and for "Captions" issues (which don't have this field).
    residual_confidence: float | None = None
    # Structured detector metadata, mainly for the webapp's visual gauges
    # (`note` stays the human-readable prose version of the same numbers
    # for CLI/report.txt).
    # "A/V sync": confidence/min_confidence come from the coarse or SyncNet
    # detector; method is "coarse" | "syncnet".
    # "Captions": confidence is matched_count / (matched_count +
    # unmatched_count), the fraction of cues that had a nearby speech
    # onset to compare against, not a model score; method is None.
    confidence: float | None = None
    min_confidence: float | None = None
    method: str | None = None
    matched_count: int | None = None
    unmatched_count: int | None = None
    # Populated only for "A/V sync" issues resolved via the piecewise path
    # (`syncsentry.lipsync.piecewise_offset`, `method="piecewise"`):
    # content that doesn't fit the single-global-offset assumption every
    # other path here makes (a video assembled from multiple
    # independently-offset sources). Each dict has `start_s`, `end_s`,
    # `offset_ms` (`None` for an undetermined span), `status` ("trusted"
    # | "undetermined"). `detected_offset_ms` / `residual_offset_ms` on
    # the issue itself don't mean much for a piecewise result (there is
    # no one number) and are left `None`; `note` carries the
    # human-readable per-segment breakdown instead.
    segments: list[dict] | None = None


@dataclass
class SyncFixSummary:
    asset_name: str
    issues: list[IssueSummary] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)

    @property
    def all_resolved(self) -> bool:
        return all(i.status in ("in_sync", "fixed") for i in self.issues)

    def to_text(self) -> str:
        width = 60
        lines = [
            "SyncSentry Report",
            f"Asset: {self.asset_name}",
            "-" * width,
        ]
        for issue in self.issues:
            if issue.segments is not None:
                n_trusted = sum(1 for s in issue.segments if s["status"] == "trusted")
                lines.append(f"{issue.name:<12}: PIECEWISE, {len(issue.segments)} region(s), "
                             f"{n_trusted} independently corrected")
                for s in issue.segments:
                    off = f"{s['offset_ms']:+.0f}ms" if s["offset_ms"] is not None else "undetermined"
                    lines.append(f"  [{s['start_s']:6.1f}s - {s['end_s']:6.1f}s]  {off}  ({s['status']})")
                if issue.note:
                    lines.append(f"  [{issue.note}]")
                continue
            if issue.status == "undetermined":
                # Genuinely can't tell: never phrase this as "in sync".
                # Guard against `detected_offset_ms=None` (the "no
                # matching speech onsets" case), which would otherwise
                # crash here on NoneType.__format__.
                lines.append(f"{issue.name:<12}: UNDETERMINED: {issue.note or 'insufficient evidence'}")
                continue
            if issue.status == "in_sync":
                if issue.detected_offset_ms is not None:
                    lines.append(f"{issue.name:<12}: in sync ({issue.detected_offset_ms:+.0f}ms, below threshold)")
                else:
                    lines.append(f"{issue.name:<12}: in sync")
                continue

            status = "FIXED" if issue.status == "fixed" else "NOT FIXED"
            line = f"{issue.name:<12}: detected {issue.detected_offset_ms:+.0f}ms -> {status}"
            if issue.residual_offset_ms is not None:
                conf_str = (f", confidence {issue.residual_confidence:.2f}"
                            if issue.residual_confidence is not None else "")
                line += f" (residual {issue.residual_offset_ms:+.1f}ms{conf_str})"
            if issue.note:
                line += f"  [{issue.note}]"
            lines.append(line)

        lines.append("-" * width)
        overall = "\u2705 all issues resolved" if self.all_resolved else "\u26a0 some issues remain, see above"
        lines.append(f"Overall: {overall}")
        if self.output_files:
            lines.append("Output files:")
            for label, path in self.output_files.items():
                lines.append(f"  {label}: {path}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "asset_name": self.asset_name,
            "overall_status": "resolved" if self.all_resolved else "attention_needed",
            "issues": [
                {
                    "name": i.name,
                    "status": i.status,
                    "had_issue": i.had_issue,
                    "detected_offset_ms": i.detected_offset_ms,
                    "fixed": i.fixed,
                    "residual_offset_ms": i.residual_offset_ms,
                    "residual_confidence": i.residual_confidence,
                    "note": i.note,
                    "confidence": i.confidence,
                    "min_confidence": i.min_confidence,
                    "method": i.method,
                    "matched_count": i.matched_count,
                    "unmatched_count": i.unmatched_count,
                    "segments": i.segments,
                }
                for i in self.issues
            ],
            "output_files": self.output_files,
        }
