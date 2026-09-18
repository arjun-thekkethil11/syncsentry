import { useEffect, useState } from "react";
import { checkHealth } from "../api/client";

type Status = "checking" | "up" | "down";

/** Polls the analyzer's /healthz so the UI can honestly say whether the
 * backend is actually reachable, instead of only discovering that the hard
 * way when a multi-minute SyncNet run's upload immediately fails. */
export function useApiHealth(intervalMs = 15000) {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const ok = await checkHealth();
      if (!cancelled) setStatus(ok ? "up" : "down");
    };
    check();
    const id = setInterval(check, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return { status };
}
