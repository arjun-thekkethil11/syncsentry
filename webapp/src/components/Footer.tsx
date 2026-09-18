export function Footer() {
  return (
    <footer className="border-t border-white/5 mt-auto">
      <div className="mx-auto max-w-6xl px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
        <p>SyncSentry — open-source A/V &amp; caption sync detection for OTT/streaming assets.</p>
        <div className="flex items-center gap-4">
          <a
            className="hover:text-slate-300 transition-colors"
            href="https://github.com/arjun-thekkethil11/syncsentry"
            target="_blank"
            rel="noreferrer"
          >
            Source on GitHub
          </a>
          <a
            className="hover:text-slate-300 transition-colors"
            href="https://github.com/arjun-thekkethil11/syncsentry/blob/main/docs/RESEARCH.md"
            target="_blank"
            rel="noreferrer"
          >
            Research notes
          </a>
        </div>
      </div>
    </footer>
  );
}
