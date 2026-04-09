import clsx from "clsx";
import type { AgroDetection } from "@/lib/types";

interface Props {
  detection: AgroDetection;
}

const QUALITY_TONE: Record<string, string> = {
  good: "text-[var(--color-ok)]",
  defective: "text-[var(--color-fault)]",
  unripe: "text-[var(--color-warn)]",
};

export function FruitCard({ detection }: Props) {
  const tone = QUALITY_TONE[detection.quality ?? "good"] ?? "text-[var(--color-text)]";
  return (
    <div className="panel p-3">
      <div className="flex items-center justify-between">
        <div className="value text-[14px] font-medium">{detection.fruit_class}</div>
        <span className={clsx("font-mono text-[10px] uppercase tracking-wider", tone)}>
          {detection.quality ?? "n/a"}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] text-[var(--color-text-dim)]">
        <div>
          <div className="label">conf</div>
          <div className="value text-[12px] text-[var(--color-text)]">{(detection.confidence * 100).toFixed(0)}%</div>
        </div>
        <div>
          <div className="label">⌀ mm</div>
          <div className="value text-[12px] text-[var(--color-text)]">{detection.diameter_mm.toFixed(0)}</div>
        </div>
        <div>
          <div className="label">bbox</div>
          <div className="value text-[10px] text-[var(--color-text-dim)]">
            {detection.bbox_x2 - detection.bbox_x1}×{detection.bbox_y2 - detection.bbox_y1}
          </div>
        </div>
      </div>
    </div>
  );
}
