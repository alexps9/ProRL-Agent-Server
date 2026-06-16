// Shared display/formatting helpers used across the dashboard UI.

/** Truncate a long id/string for table cells, keeping the head + an ellipsis. */
export function shortId(value: string | null | undefined, max = 12): string {
  if (!value) return "—";
  return value.length <= max ? value : `${value.slice(0, max - 1)}…`;
}

/** Mean/per-session reward as a fixed-precision string; em dash when absent. */
export function formatReward(reward: number | null | undefined): string {
  if (reward == null || Number.isNaN(reward)) return "—";
  return reward.toFixed(3);
}

/** Human-readable duration from milliseconds; em dash when absent. */
export function formatMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rem = Math.round(seconds % 60);
  return `${minutes}m ${rem}s`;
}

/** "5m ago" style label from an epoch-seconds timestamp (matches `updated_at`). */
export function relativeTime(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "—";
  const seconds = Math.round((Date.now() - epochSeconds * 1000) / 1000);
  if (seconds < 0) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/** Tailwind color classes for a status pill, grouped by terminal/active/failed. */
export function statusClass(status: string | null | undefined): string {
  const s = (status ?? "").toLowerCase();
  if (/\b(completed?|success(ful)?|succeeded|resolved|ready|ok|up|passed?|healthy)\b/.test(s))
    return "bg-emerald-100 text-emerald-700";
  if (/\b(failed?|error(ed)?|down|timeout|timed.?out|cancell?ed|aborted?)\b/.test(s))
    return "bg-rose-100 text-rose-700";
  if (/\b(running|run|initializing|init|pending|queued?|in.?progress|registered|dispatched?|starting)\b/.test(s))
    return "bg-blue-100 text-blue-700";
  return "bg-slate-100 text-slate-700";
}

/** Copy text to the clipboard, with a fallback for non-secure contexts. */
export async function copyToClipboard(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
    } finally {
      document.body.removeChild(ta);
    }
  }
}
