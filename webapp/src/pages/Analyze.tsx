import { useRef, useState } from "react";
import { UploadForm } from "../components/UploadForm";
import { ProcessingView } from "../components/ProcessingView";
import { ResultsPanel } from "../components/ResultsPanel";
import { ApiError, runFix } from "../api/client";
import type { FixOptions, FixResponse } from "../api/types";

type ViewState =
  | { kind: "idle" }
  | { kind: "processing"; useSyncnet: boolean }
  | { kind: "done"; result: FixResponse }
  | { kind: "error"; message: string };

export function Analyze() {
  const [state, setState] = useState<ViewState>({ kind: "idle" });
  const abortRef = useRef<AbortController | null>(null);

  const handleSubmit = async (video: File, captions: File | null, options: FixOptions) => {
    const controller = new AbortController();
    abortRef.current = controller;
    setState({ kind: "processing", useSyncnet: options.useSyncnet });
    try {
      const result = await runFix(video, captions, options, controller.signal);
      setState({ kind: "done", result });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setState({ kind: "idle" });
        return;
      }
      const message = err instanceof ApiError ? err.message : "Couldn't reach the SyncSentry API. Is it running?";
      setState({ kind: "error", message });
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
  };

  const handleReset = () => setState({ kind: "idle" });

  return (
    <div className="mx-auto max-w-4xl px-6 py-16">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-100 mb-2">Analyze a video</h1>
        <p className="text-slate-400 text-sm leading-relaxed">
          Upload a video, or a .zip with the video and captions inside. SyncSentry detects any
          A/V or caption sync offset, fixes it, and shows you what it found before and after.
        </p>
      </div>

      {state.kind === "idle" && <UploadForm onSubmit={handleSubmit} />}

      {state.kind === "processing" && (
        <ProcessingView useSyncnet={state.useSyncnet} onCancel={handleCancel} />
      )}

      {state.kind === "error" && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/[0.05] p-6 text-center">
          <p className="text-rose-300 font-medium mb-2">Something went wrong</p>
          <p className="text-sm text-slate-400 mb-5">{state.message}</p>
          <button
            onClick={handleReset}
            className="rounded-lg border border-white/10 hover:border-white/20 text-slate-300 text-sm font-medium px-4 py-2 transition-colors"
          >
            Try again
          </button>
        </div>
      )}

      {state.kind === "done" && <ResultsPanel result={state.result} onReset={handleReset} />}
    </div>
  );
}
