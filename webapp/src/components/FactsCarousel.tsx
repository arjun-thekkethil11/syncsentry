import { useEffect, useState } from "react";

interface FactsCarouselProps {
  facts: string[];
  intervalMs?: number;
}

/** Rotates through `facts` with a brief fade between each. Purely to give
 * the user something interesting to read during a request that can take
 * anywhere from a few seconds to a couple of minutes (see
 * `ProcessingView`); this unmounts (and so stops) the moment the result
 * comes back, since it's only ever rendered by that view. */
export function FactsCarousel({ facts, intervalMs = 5500 }: FactsCarouselProps) {
  const [index, setIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (facts.length <= 1) return;
    const id = setInterval(() => {
      setVisible(false);
      const fadeOutMs = 300;
      setTimeout(() => {
        setIndex((i) => (i + 1) % facts.length);
        setVisible(true);
      }, fadeOutMs);
    }, intervalMs);
    return () => clearInterval(id);
  }, [facts, intervalMs]);

  if (facts.length === 0) return null;
  // Guards against `facts` changing to a shorter array between renders
  // (e.g. once asset metrics finish loading) leaving a stale, now
  // out-of-bounds `index` from a previous, longer playlist.
  const currentFact = facts[index % facts.length];

  return (
    <div className="mt-6 min-h-[4.5rem] flex items-center justify-center px-2">
      <p
        className={`text-sm text-slate-400 text-center leading-relaxed max-w-md transition-opacity duration-300 ${
          visible ? "opacity-100" : "opacity-0"
        }`}
      >
        <span className="text-slate-500">Did you know? </span>
        {currentFact}
      </p>
    </div>
  );
}
