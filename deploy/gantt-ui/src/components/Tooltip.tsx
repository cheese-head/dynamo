import { useGanttStore } from "../store";
import { spanColor } from "../lib/colors";
import { fmtDuration, fmtTime } from "../lib/time";

export function Tooltip() {
  const hover = useGanttStore((s) => s.hoverData);
  if (!hover) return null;

  const { span, row, x, y } = hover;
  const color = spanColor(span.name);
  const requestId = row.attributes.request_id || row.label;

  return (
    <div
      className="pointer-events-none fixed z-50 rounded-lg border border-border bg-surface px-3 py-2 shadow-xl"
      style={{
        left: x + 14,
        top: y - 10,
        backdropFilter: "blur(8px)",
        maxWidth: 320,
      }}
    >
      <div className="mb-1 flex items-center gap-2">
        <span
          className="inline-block h-2.5 w-2.5 rounded-sm"
          style={{ backgroundColor: color }}
        />
        <span className="text-xs font-semibold text-text-primary">
          {span.name}
        </span>
      </div>
      <div className="flex flex-col gap-0.5 text-[11px] text-text-secondary">
        <span>Start: <span className="font-mono text-text-primary">{fmtTime(span.startNs)}</span></span>
        <span>Duration: <span className="text-text-primary">{fmtDuration(span.durMs)}</span></span>
        <span>Request ID: <span className="font-mono text-text-primary">{requestId}</span></span>
      </div>
    </div>
  );
}
