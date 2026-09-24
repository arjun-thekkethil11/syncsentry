import { useEffect, useState } from "react";

interface FactsCarouselProps {
  facts: string[];
  /** Floor and ceiling on how long a single fact stays on screen. Actual
   * time scales with the fact's word count (see `readingDurationMs`) so
   * a short fact isn't stuck lingering and a long one isn't yanked away
   * before it can be read. */
  minMs?: number;
  maxMs?: number;
}

const FADE_MS = 350;

// Slower than average adult silent-reading speed (~200-250 wpm, i.e.
// ~240-300ms/word) on purpose: this is a passive, secondary UI element
// competing with a mildly stressful "please wait" moment, not a page of
// focused text, so it gets extra margin to actually be read rather than
// just technically displayed for "long enough".
const MS_PER_WORD = 340;

function readingDurationMs(fact: string, minMs: number, maxMs: number): number {
  const words = fact.trim().split(/\s+/).filter(Boolean).length;
  return Math.min(maxMs, Math.max(minMs, words * MS_PER_WORD));
}

/** Rotates through `facts`, holding each on screen for a duration scaled
 * to its length, with a small card treatment (icon + progress bar +
 * position dots) instead of a bare paragraph. Purely to give the user
 * something interesting to read during a request that can take anywhere
 * from a few seconds to a couple of minutes (see `ProcessingView`); this
 * unmounts (and so stops) the moment the result comes back, since it's
 * only ever rendered by that view. */
export function FactsCarousel({ facts, minMs = 7000, maxMs = 13000 }: FactsCarouselProps) {
  const [index, setIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  // Guards against `facts` changing to a shorter array between renders
  // (e.g. once asset metrics finish loading) leaving a stale, now
  // out-of-bounds `index` from a previous, longer playlist.
  const currentFact = facts.length > 0 ? facts[index % facts.length] : null;
  const durationMs = currentFact ? readingDurationMs(currentFact, minMs, maxMs) : minMs;

  useEffect(() => {
    if (facts.length <= 1) return;
    let fadeId: ReturnType<typeof setTimeout> | undefined;
    const holdId = setTimeout(() => {
      setVisible(false);
      fadeId = setTimeout(() => {
        setIndex((i) => (i + 1) % facts.length);
        setVisible(true);
      }, FADE_MS);
    }, durationMs);
    return () => {
      clearTimeout(holdId);
      if (fadeId) clearTimeout(fadeId);
    };
    // Re-arms whenever the visible fact (or its computed duration)
    // changes - i.e. once per rotation, not on every unrelated re-render.
  }, [index, facts.length, durationMs]);

  if (!currentFact) return null;

  // Capped so a long asset+general facts playlist doesn't turn into a
  // wall of tiny dots; the progress bar alone still communicates pacing.
  const showDots = facts.length > 1 && facts.length <= 9;

  return (
    <div className="mt-6 mx-auto max-w-md">
      <div className="relative overflow-hidden rounded-xl border border-brand-500/25 bg-gradient-to-br from-brand-500/[0.08] via-white/[0.02] to-transparent px-5 py-4 shadow-[0_0_30px_-15px_rgba(56,102,240,0.6)]">
        <div
          className={`flex items-start gap-3 transition-all duration-300 ${
            visible ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-1"
          }`}
        >
          <span aria-hidden className="text-lg leading-none mt-0.5 shrink-0">
            ✨
          </span>
          <p className="text-sm text-slate-300 text-left leading-relaxed">
            <span className="text-brand-300 font-medium">Did you know? </span>
            {currentFact}
          </p>
        </div>

        {/* Time-remaining bar for the current fact. Re-keyed on `index`
            so the fill animation restarts from empty every rotation. */}
        <div className="mt-3 h-1 w-full rounded-full bg-white/5 overflow-hidden">
          <div
            key={index}
            className="h-full rounded-full bg-gradient-to-r from-brand-500 to-brand-300"
            style={{ animation: `fact-progress ${durationMs}ms linear forwards` }}
          />
        </div>
      </div>

      {showDots && (
        <div className="flex items-center justify-center gap-1.5 mt-3">
          {facts.map((_, i) => (
            <span
              key={i}
              aria-hidden
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === index % facts.length ? "w-4 bg-brand-400" : "w-1.5 bg-white/15"
              }`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
