import type { ReactNode } from "react";

interface StatusBadgeProps {
  tone: "good" | "warn" | "bad" | "neutral";
  children: ReactNode;
}

const toneClasses: Record<StatusBadgeProps["tone"], string> = {
  good: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  warn: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  bad: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  neutral: "bg-slate-500/15 text-slate-300 border-slate-500/30",
};

export function StatusBadge({ tone, children }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${toneClasses[tone]}`}
    >
      {children}
    </span>
  );
}
