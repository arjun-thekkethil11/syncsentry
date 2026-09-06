"""The small, human-readable report SyncSentry produces for one asset.

Deliberately NOT a dump of every internal measurement -- just the handful of
numbers a content-ops person or an engineer skimming a CI log actually needs:
what was wrong, what we did about it, and proof it worked (the residual after
fixing). Per-cue detail, confidence scores, etc. still live in the JSON
detection/fix results for anyone who needs to dig deeper, but they aren't
in this summary.
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
    note: str = ""


@dataclass
class SyncFixSummary:
    asset_name: str
    issues: list[IssueSummary] = field(default_factory=list)
    output_files: dict[str, str] = field(default_factory=dict)

    @property
    def all_resolved(self) -> bool:
        return all(
            (not i.had_issue) or (i.residual_offset_ms is not None and abs(i.residual_offset_ms) < 20)
            for i in self.issues
        )

    def to_text(self) -> str:
        width = 60
        lines = [
            "SyncSentry Report",
            f"Asset: {self.asset_name}",
            "-" * width,
        ]
        for issue in self.issues:
            if not issue.had_issue:
                lines.append(f"{issue.name:<12}: in sync ({issue.detected_offset_ms:+.0f}ms, below threshold)")
                continue

            status = "FIXED" if issue.fixed else "NOT FIXED"
            line = f"{issue.name:<12}: detected {issue.detected_offset_ms:+.0f}ms -> {status}"
            if issue.fixed and issue.residual_offset_ms is not None:
                line += f" (residual {issue.residual_offset_ms:+.1f}ms)"
            if issue.note:
                line += f"  [{issue.note}]"
            lines.append(line)

        lines.append("-" * width)
        overall = "\u2705 all issues resolved" if self.all_resolved else "\u26a0 some issues remain -- see above"
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
                    "had_issue": i.had_issue,
                    "detected_offset_ms": i.detected_offset_ms,
                    "fixed": i.fixed,
                    "residual_offset_ms": i.residual_offset_ms,
                    "note": i.note,
                }
                for i in self.issues
            ],
            "output_files": self.output_files,
        }
