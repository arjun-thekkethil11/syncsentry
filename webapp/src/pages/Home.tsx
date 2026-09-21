import { Link } from "react-router-dom";

const steps = [
  {
    title: "Detect",
    body: "Tracks faces and lip movement with SyncNet to find the true audio/video offset. Falls back to audio/video envelope cross-correlation when there is no usable face signal, and checks caption drift if you upload captions.",
  },
  {
    title: "Fix",
    body: "Re-encodes the audio track and shifts caption timestamps to match. No metadata-only remux.",
  },
  {
    title: "Verify",
    body: "Re-measures the corrected output before calling it fixed. If the residual offset is still above threshold, it says so instead of reporting a false pass.",
  },
];

export function Home() {
  return (
    <div className="mx-auto max-w-6xl px-6">
      {/* Hero */}
      <section className="pt-20 pb-16 text-center">
        <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400 mb-6">
          Open source
        </span>
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-50 mb-5">
          Find and fix A/V &amp; caption
          <br className="hidden sm:block" /> sync drift in video files.
        </h1>
        <p className="max-w-2xl mx-auto text-slate-400 text-lg leading-relaxed mb-8">
          Upload a video and optional captions. SyncSentry detects the real offset,
          corrects it, and re-measures the result to confirm the fix worked.
        </p>
        <div className="flex items-center justify-center gap-3">
          <Link
            to="/analyze"
            className="rounded-lg bg-brand-500 hover:bg-brand-400 text-white font-medium px-6 py-3 transition-colors shadow-lg shadow-brand-500/20"
          >
            Try it on your video
          </Link>
          <a
            href="https://github.com/arjun-thekkethil11/syncsentry"
            target="_blank"
            rel="noreferrer"
            className="rounded-lg border border-white/10 hover:border-white/20 text-slate-300 font-medium px-6 py-3 transition-colors"
          >
            View source
          </a>
        </div>
      </section>

      {/* What it does */}
      <section className="pb-16">
        <h2 className="text-xl font-semibold text-slate-100 mb-2">What this does</h2>
        <p className="text-slate-400 mb-8 max-w-3xl">
          Most sync tools assume one global offset for the whole file. Real content drifts in
          different ways: a constant offset, drift that worsens over time, or a different
          offset in each clip after an edit. SyncSentry detects and corrects all of these,
          and reports low-confidence cases as undetermined instead of guessing.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
          {steps.map((s, i) => (
            <div key={s.title} className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brand-500/20 text-brand-300 text-xs font-semibold">
                  {i + 1}
                </span>
                <h3 className="font-semibold text-slate-100">{s.title}</h3>
              </div>
              <p className="text-sm text-slate-400 leading-relaxed">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Detectors */}
      <section className="pb-20">
        <h2 className="text-xl font-semibold text-slate-100 mb-2">Two detectors</h2>
        <p className="text-slate-400 mb-8 max-w-3xl">
          SyncNet runs by default for accuracy. The coarse detector is available as a faster,
          less reliable alternative.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <div className="rounded-xl border border-brand-500/30 bg-brand-500/[0.04] p-6">
            <h3 className="font-semibold text-slate-100 mb-2">SyncNet lip-motion (default)</h3>
            <p className="text-sm text-slate-400 leading-relaxed mb-3">
              Tracks faces and mouth motion, and checks multiple windows before trusting an
              offset. A wide-range pass extends this beyond SyncNet's native 600ms window to
              recover offsets of several seconds.
            </p>
            <span className="text-xs text-slate-500">30 to 120 seconds per video</span>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6">
            <h3 className="font-semibold text-slate-100 mb-2">Coarse (fast alternative)</h3>
            <p className="text-sm text-slate-400 leading-relaxed mb-3">
              Cross-correlates audio and video envelopes. Works well for hard cuts and
              flash-and-beep content, less reliable on continuous dialogue.
            </p>
            <span className="text-xs text-slate-500">1 to 5 seconds per video</span>
          </div>
        </div>
      </section>
    </div>
  );
}
