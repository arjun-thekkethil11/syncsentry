// Facts shown in the `FactsCarousel` while a request is processing (see
// `ProcessingView`). Two sources, deliberately kept separate then
// interleaved: facts actually computed from *this* asset (so "the asset
// the user gave" is literal, not just flavor text), and general A/V-sync
// trivia as filler so the carousel still has enough variety on a short
// clip or before metadata has loaded.

export interface AssetMetrics {
  durationS: number | null;
  width: number | null;
  height: number | null;
}

/** Reads duration/resolution off a hidden `<video>` element rather than
 * shelling out to ffprobe or pulling in a parsing library: this only
 * needs to feed a "fun fact", not drive any real detection logic, so the
 * browser's own (cheap, always-available) metadata is enough. Resolves
 * with all-`null` fields on any failure rather than rejecting, since a
 * missing fact is fine but an unhandled rejection breaking the whole
 * processing view over flavor text is not. */
export function readVideoMetrics(file: File): Promise<AssetMetrics> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const videoEl = document.createElement("video");
    videoEl.preload = "metadata";
    videoEl.muted = true;
    videoEl.src = url;

    const finish = (metrics: AssetMetrics) => {
      URL.revokeObjectURL(url);
      resolve(metrics);
    };

    videoEl.onloadedmetadata = () => {
      finish({
        durationS: Number.isFinite(videoEl.duration) ? videoEl.duration : null,
        width: videoEl.videoWidth || null,
        height: videoEl.videoHeight || null,
      });
    };
    videoEl.onerror = () => finish({ durationS: null, width: null, height: null });
  });
}

function formatDuration(totalS: number): string {
  const m = Math.floor(totalS / 60);
  const s = Math.round(totalS % 60);
  if (m <= 0) return `${s}s`;
  return `${m}m ${s}s`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

function aspectRatioLabel(width: number, height: number): string {
  const ratio = width / height;
  // Named common ratios within a small tolerance read better than a
  // literal reduced fraction (e.g. 1920x1080 reducing to "16:9" is
  // recognizable; most real footage's reduced fraction is not).
  const named: [number, string][] = [
    [16 / 9, "16:9"],
    [4 / 3, "4:3"],
    [1, "1:1 (square)"],
    [9 / 16, "9:16 (vertical)"],
    [3 / 4, "3:4 (vertical)"],
  ];
  for (const [target, label] of named) {
    if (Math.abs(ratio - target) < 0.02) return label;
  }
  const d = gcd(width, height);
  return `${width / d}:${height / d}`;
}

/** Facts computed directly from this specific upload: what the user
 * quite literally gave the tool. Only includes a fact when the
 * underlying number was actually readable, so a video whose metadata
 * failed to load still gets *something* (at minimum, file size, which
 * comes straight from the `File` object and can't fail). */
export function buildAssetFacts(file: File, metrics: AssetMetrics): string[] {
  const facts: string[] = [`This file is ${formatBytes(file.size)}.`];

  if (metrics.durationS && metrics.durationS > 0) {
    facts.push(`Running time: ${formatDuration(metrics.durationS)}.`);

    const totalFrames25fps = Math.round(metrics.durationS * 25);
    facts.push(
      `At a typical 25fps, that's about ${totalFrames25fps.toLocaleString()} frames \u2014 ` +
        `any single dropped or duplicated one among them can throw sync off by a frame's worth of time.`,
    );

    const mbps = (file.size * 8) / metrics.durationS / 1_000_000;
    if (Number.isFinite(mbps) && mbps > 0) {
      facts.push(`Average bitrate: roughly ${mbps.toFixed(1)} Mbps.`);
    }
  }

  if (metrics.width && metrics.height) {
    facts.push(`Resolution: ${metrics.width}\u00d7${metrics.height} (${aspectRatioLabel(metrics.width, metrics.height)}).`);
  }

  return facts;
}

/** General A/V-sync trivia, unrelated to any specific asset. Exists so
 * the carousel has real variety even on a short clip (few asset facts)
 * or before metadata has finished loading (zero asset facts), and
 * because it's genuinely relevant context for what the tool is doing
 * while the user waits. */
export const GENERAL_SYNC_FACTS: string[] = [
  "Most viewers start noticing audio lagging behind video once the gap passes about 45ms \u2014 roughly one frame at 24fps.",
  "Audio arriving early is generally noticed later, and tolerated more, than audio arriving late by the same amount.",
  "Broadcast delivery standards (e.g. EBU R37, ATSC) define acceptable A/V sync error in single-digit milliseconds \u2014 far tighter than most viewers can consciously detect.",
  "SyncNet, one of the models this tool can use, learns to match lip movement to speech without ever being told what a phoneme is.",
  "A/V drift often creeps in gradually during editing: one dropped or duplicated frame early in a timeline can throw everything after it off by tens of milliseconds.",
  "A single frame at 30fps lasts about 33 milliseconds \u2014 many real-world sync issues are smaller than that, which is why eyeballing a waveform often isn't enough.",
  "Some sync issues aren't constant: a video assembled from multiple sources can have a different offset in different regions of the same file.",
  "Lip-sync detection models work by comparing short-window mouth motion to the speech audio in that same window, then checking how many independent windows agree.",
];

/** Interleaves asset-specific facts (shown first, since they're the more
 * literal answer to "facts about the asset the user gave") with a
 * shuffled sample of general trivia, for the carousel to rotate through. */
export function buildFactsPlaylist(assetFacts: string[]): string[] {
  const shuffled = [...GENERAL_SYNC_FACTS].sort(() => Math.random() - 0.5);
  return [...assetFacts, ...shuffled];
}
