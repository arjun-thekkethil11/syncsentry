import { useEffect, useState } from "react";
import { checkHealth } from "../api/client";

type Status = "checking" | "up" | "down";

/** Polls the analyzer's /healthz endpoint so the UI can show whether the
 * backend is reachable. */
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
