import { useEffect, useState } from "react";

interface ProcessingViewProps {
  useSyncnet: boolean;
  onCancel: () => void;
}

/** The pipeline runs as one blocking request, so there is no real progress
 * percentage. This shows an elapsed-time counter and the current stage
 * instead of a fake progress bar. */
export function ProcessingView({ useSyncnet, onCancel }: ProcessingViewProps) {
  const [elapsedS, setElapsedS] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => setElapsedS(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);

  const stage = !useSyncnet
    ? "Extracting audio and video signals..."
    : elapsedS < 15
      ? "Extracting audio and video signals..."
      : elapsedS < 90
        ? "Tracking faces and running SyncNet..."
        : "Verifying the fix...";

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-10 text-center">
      <div className="mx-auto mb-6 h-12 w-12">
        <svg className="animate-spin h-12 w-12 text-brand-400" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
          <path
            d="M12 2a10 10 0 0 1 10 10"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
          />
        </svg>
      </div>
      <p className="text-slate-200 font-medium mb-1">Processing...</p>
      <p className="text-sm text-slate-500 mb-6">{stage}</p>
      <p className="text-2xl font-mono tabular-nums text-brand-300 mb-1">
        {String(Math.floor(elapsedS / 60)).padStart(2, "0")}:{String(elapsedS % 60).padStart(2, "0")}
      </p>
      <p className="text-xs text-slate-600 mb-6">
        {useSyncnet ? "Typically 30 to 120 seconds" : "Usually a few seconds"}
      </p>
      <button
        onClick={onCancel}
        className="text-sm text-slate-500 hover:text-rose-400 transition-colors"
      >
        Cancel
      </button>
    </div>
  );
}
