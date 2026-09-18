import { Link } from "react-router-dom";

const steps = [
  {
    title: "Detect",
    body: "Cross-correlates the audio envelope against the video brightness envelope (and, optionally, a pretrained SyncNet lip-sync model) to find the true A/V offset — plus caption-vs-speech drift if you upload captions.",
  },
  {
    title: "Fix",
    body: "Re-encodes the audio track (real sample trim/pad, not a metadata-only remux) and shifts caption cue timestamps to match.",
  },
  {
    title: "Verify",
    body: "Re-measures the corrected output before calling it fixed — if the residual offset is still above threshold, it says so instead of reporting a false pass.",
  },
];

const facts = [
  { k: "0.00ms", v: "mean A/V offset error across a ±900ms synthetic sweep" },
  { k: "±1 frame", v: "SyncNet accuracy on real talking-head content" },
  { k: "2 tiers", v: "fast coarse detector (seconds) + opt-in learned model (accurate, minutes)" },
];

export function Home() {
  return (
    <div className="mx-auto max-w-6xl px-6">
      {/* Hero */}
      <section className="pt-20 pb-16 text-center">
        <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400 mb-6">
          Research-grounded · open source
        </span>
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-50 mb-5">
          Find and fix A/V &amp; caption
          <br className="hidden sm:block" /> sync drift in video assets.
        </h1>
        <p className="max-w-2xl mx-auto text-slate-400 text-lg leading-relaxed mb-8">
          Upload a video (and optional captions). SyncSentry detects the real offset,
          corrects it, and re-measures the result to prove the fix actually worked —
          instead of guessing from a fixed tolerance the way most caption/VOD QC does.
        </p>
        <div className="flex items-center justify-center gap-3">
          <Link
            to="/analyze"
            className="rounded-lg bg-brand-500 hover:bg-brand-400 text-white font-medium px-6 py-3 transition-colors shadow-lg shadow-brand-500/20"
          >
            Try it on your video →
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

      {/* Stats */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4 pb-16">
        {facts.map((f) => (
          <div key={f.k} className="rounded-xl border border-white/10 bg-white/[0.02] p-5 text-center">
            <div className="text-2xl font-bold text-brand-300 mb-1">{f.k}</div>
            <div className="text-sm text-slate-500">{f.v}</div>
          </div>
        ))}
      </section>

      {/* What it does */}
      <section className="pb-16">
        <h2 className="text-xl font-semibold text-slate-100 mb-2">What this actually does</h2>
        <p className="text-slate-400 mb-8 max-w-3xl">
          Most caption/VOD sync QC treats every drift as one global offset caught by a fixed
          tolerance. Real content drifts in different ways — constant offset, drift that
          worsens over a title's runtime, or intermittent per-scene issues from edits/cuts.
          SyncSentry detects and corrects the common case (a global A/V or caption offset) end
          to end, and is honest — via a confidence gate — when it can't reliably tell rather
          than "fixing" noise.
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
        <h2 className="text-xl font-semibold text-slate-100 mb-2">Two detector tiers</h2>
        <p className="text-slate-400 mb-8 max-w-3xl">
          Pick the trade-off that fits your use case — both are wired into the same pipeline
          below.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6">
            <h3 className="font-semibold text-slate-100 mb-2">Coarse (default)</h3>
            <p className="text-sm text-slate-400 leading-relaxed mb-3">
              Audio RMS envelope × video brightness envelope, cross-correlated. Seconds per
              video. Great for hard cuts / flash-and-beep style content; can be less reliable
              on continuous talking-head dialogue.
            </p>
            <span className="text-xs text-slate-500">~1–5 seconds per video</span>
          </div>
          <div className="rounded-xl border border-brand-500/30 bg-brand-500/[0.04] p-6">
            <h3 className="font-semibold text-slate-100 mb-2">SyncNet (opt-in)</h3>
            <p className="text-sm text-slate-400 leading-relaxed mb-3">
              A pretrained lip-sync model: tracks faces, gates on active-speaker mouth motion,
              and cross-corroborates windows before trusting an offset. Validated to recover
              known offsets within ~1 frame on real talking-head footage.
            </p>
            <span className="text-xs text-slate-500">~1–5 minutes per video (CPU)</span>
          </div>
        </div>
      </section>
    </div>
  );
}
