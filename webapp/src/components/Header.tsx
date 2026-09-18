import { NavLink } from "react-router-dom";
import { useApiHealth } from "../hooks/useApiHealth";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
    isActive ? "bg-brand-500/20 text-brand-200" : "text-slate-400 hover:text-slate-100 hover:bg-white/5"
  }`;

export function Header() {
  const { status } = useApiHealth();

  return (
    <header className="sticky top-0 z-40 border-b border-white/5 bg-[#0a0e1a]/85 backdrop-blur-md">
      <div className="mx-auto max-w-6xl flex items-center justify-between px-6 py-3.5">
        <NavLink to="/" className="flex items-center gap-2 shrink-0">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-400 to-brand-700 text-white font-bold text-sm">
            S
          </span>
          <span className="font-semibold text-slate-100 tracking-tight">SyncSentry</span>
        </NavLink>

        <nav className="flex items-center gap-1">
          <NavLink to="/" className={navLinkClass} end>
            About
          </NavLink>
          <NavLink to="/analyze" className={navLinkClass}>
            Analyze
          </NavLink>
          <a
            href="https://github.com/arjun-thekkethil11/syncsentry"
            target="_blank"
            rel="noreferrer"
            className="px-3 py-1.5 rounded-full text-sm font-medium text-slate-400 hover:text-slate-100 hover:bg-white/5 transition-colors"
          >
            GitHub
          </a>
        </nav>

        <div className="hidden sm:flex items-center gap-2 shrink-0 text-xs text-slate-500">
          <span
            className={`h-2 w-2 rounded-full ${
              status === "up" ? "bg-emerald-400" : status === "down" ? "bg-rose-400" : "bg-slate-500 animate-pulse"
            }`}
          />
          {status === "up" ? "API connected" : status === "down" ? "API unreachable" : "Checking API…"}
        </div>
      </div>
    </header>
  );
}
