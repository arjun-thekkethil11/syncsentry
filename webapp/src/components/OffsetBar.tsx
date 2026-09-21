interface OffsetBarProps {
  label: string;
  offsetMs: number | null;
  /** Full-scale range the bar represents, in ms either side of 0. */
  scaleMs?: number;
}

/** A horizontal offset bar, centered at 0ms, showing direction and
 * magnitude at a glance. */
export function OffsetBar({ label, offsetMs, scaleMs = 500 }: OffsetBarProps) {
  const clamped = offsetMs === null ? 0 : Math.max(-scaleMs, Math.min(scaleMs, offsetMs));
  const pct = (clamped / scaleMs) * 50; // -50..50, relative to center
  const isUnknown = offsetMs === null;
  const magnitude = offsetMs === null ? 0 : Math.abs(offsetMs);
  const barColor = isUnknown ? "bg-slate-500" : magnitude < 40 ? "bg-emerald-400" : magnitude < 150 ? "bg-amber-400" : "bg-rose-400";

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-sm text-slate-400">{label}</span>
        <span className="text-sm font-mono tabular-nums text-slate-200">
          {isUnknown ? "n/a" : `${offsetMs > 0 ? "+" : ""}${offsetMs.toFixed(0)} ms`}
        </span>
      </div>
      <div className="relative h-3 rounded-full bg-slate-800 overflow-hidden">
        {/* center tick = perfectly in sync */}
        <div className="absolute left-1/2 top-0 h-full w-px bg-slate-600 z-10" />
        <div
          className={`absolute top-0 h-full rounded-full transition-all duration-700 ease-out ${barColor}`}
          style={
            pct >= 0
              ? { left: "50%", width: `${pct}%` }
              : { right: "50%", width: `${-pct}%` }
          }
        />
      </div>
      <div className="flex justify-between mt-1 text-[10px] uppercase tracking-wide text-slate-600">
        <span>audio leads</span>
        <span>in sync</span>
        <span>video leads</span>
      </div>
    </div>
  );
}
