interface ConfidenceGaugeProps {
  confidence: number | null;
  threshold: number;
  size?: number;
}

/** Radial gauge showing detector confidence against the threshold used
 * to decide whether to trust it. */
export function ConfidenceGauge({ confidence, threshold, size = 120 }: ConfidenceGaugeProps) {
  const radius = size / 2 - 10;
  const circumference = 2 * Math.PI * radius;
  const maxScale = Math.max(threshold * 2, confidence ?? 0, 1);
  const frac = confidence === null ? 0 : Math.min(1, confidence / maxScale);
  const dash = frac * circumference;
  const passed = confidence !== null && confidence >= threshold;
  const color = confidence === null ? "#64748b" : passed ? "#34d399" : "#f59e0b";
  const thresholdFrac = Math.min(1, threshold / maxScale);
  const thresholdAngle = thresholdFrac * 360;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#1e2740"
            strokeWidth={9}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={9}
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference}`}
            className="transition-all duration-700 ease-out"
          />
          {/* threshold tick mark */}
          <line
            x1={size / 2 + radius}
            y1={size / 2}
            x2={size / 2 + radius - 12}
            y2={size / 2}
            stroke="#94a3b8"
            strokeWidth={2}
            transform={`rotate(${thresholdAngle} ${size / 2} ${size / 2})`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-lg font-semibold tabular-nums text-slate-100">
            {confidence === null ? "N/A" : confidence.toFixed(2)}
          </span>
          <span className="text-[10px] text-slate-500">confidence</span>
        </div>
      </div>
      <span className={`text-xs font-medium ${passed ? "text-emerald-400" : "text-amber-400"}`}>
        {confidence === null ? "no signal" : passed ? "above threshold" : "below threshold"} (min {threshold.toFixed(2)})
      </span>
    </div>
  );
}
