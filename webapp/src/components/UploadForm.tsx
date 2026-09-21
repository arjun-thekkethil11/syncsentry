import { useState } from "react";
import { FileDrop } from "./FileDrop";
import type { FixOptions } from "../api/types";

interface UploadFormProps {
  onSubmit: (video: File, captions: File | null, options: FixOptions) => void;
  disabled?: boolean;
}

// SyncNet is more accurate on real talking-head content, so it is on
// by default. The plain cross-correlation detector is faster but less
// reliable, so it is opt-out rather than the default.
const DEFAULT_OPTIONS: FixOptions = {
  useSyncnet: true,
  avMinConfidence: 0.3,
  syncnetMinConfidence: 3.0,
};

export function UploadForm({ onSubmit, disabled }: UploadFormProps) {
  const [video, setVideo] = useState<File | null>(null);
  const [captions, setCaptions] = useState<File | null>(null);
  const [options, setOptions] = useState<FixOptions>(DEFAULT_OPTIONS);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const isZip = video?.name.toLowerCase().endsWith(".zip");
  const canSubmit = video !== null && !disabled;

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (video) onSubmit(video, captions, options);
      }}
      className="space-y-6"
    >
      <FileDrop
        label="Video (or a .zip bundle)"
        hint="MP4, MKV, MOV, WebM, or a .zip containing the video and optional captions"
        accept="video/*,.zip,.mkv"
        file={video}
        onChange={setVideo}
      />

      {isZip && (
        <p className="text-xs text-brand-300 bg-brand-500/10 border border-brand-500/20 rounded-lg px-3 py-2">
          Zip detected. SyncSentry will extract it and look for a video and optional captions
          inside.
        </p>
      )}

      {!isZip && (
        <FileDrop
          label="Captions"
          hint="WebVTT (.vtt) or SubRip (.srt)"
          accept=".vtt,.srt"
          file={captions}
          onChange={setCaptions}
          optional
        />
      )}

      <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={options.useSyncnet}
            onChange={(e) => setOptions((o) => ({ ...o, useSyncnet: e.target.checked }))}
            className="mt-1 h-4 w-4 rounded border-white/20 bg-white/5 text-brand-500 focus:ring-brand-400"
          />
          <span>
            <span className="block text-sm font-medium text-slate-200">
              Use SyncNet lip-motion analysis
            </span>
            <span className="block text-xs text-slate-500 mt-0.5">
              More accurate on talking-head video, slower (30 to 120 seconds). Uncheck for a
              faster, less reliable estimate.
            </span>
          </span>
        </label>

        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          className="mt-3 text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          {advancedOpen ? "Hide advanced options" : "Advanced options"}
        </button>

        {advancedOpen && (
          <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-white/5">
            <label className="block">
              <span className="text-xs text-slate-400">
                Coarse detector min. confidence ({options.avMinConfidence.toFixed(2)})
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={options.avMinConfidence}
                onChange={(e) => setOptions((o) => ({ ...o, avMinConfidence: Number(e.target.value) }))}
                className="w-full mt-1 accent-brand-500"
              />
            </label>
            <label className="block">
              <span className="text-xs text-slate-400">
                SyncNet min. confidence ({options.syncnetMinConfidence.toFixed(1)})
              </span>
              <input
                type="range"
                min={0}
                max={6}
                step={0.1}
                value={options.syncnetMinConfidence}
                onChange={(e) =>
                  setOptions((o) => ({ ...o, syncnetMinConfidence: Number(e.target.value) }))
                }
                disabled={!options.useSyncnet}
                className="w-full mt-1 accent-brand-500 disabled:opacity-40"
              />
            </label>
            <p className="text-[11px] text-slate-500 sm:col-span-2">
              Lower thresholds fix more offsets automatically but risk more false positives.
            </p>
          </div>
        )}
      </div>

      <button
        type="submit"
        disabled={!canSubmit}
        className="w-full rounded-lg bg-brand-500 hover:bg-brand-400 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-medium py-3 transition-colors"
      >
        {disabled ? "Processing..." : "Detect and fix sync issues"}
      </button>
    </form>
  );
}
