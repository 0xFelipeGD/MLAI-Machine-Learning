// Optional canvas-based heatmap overlay. The engine already returns
// heatmap-fused frames so this component is mostly a visual decoration
// (corner crosshair + frame counter overlay).

interface Props {
  score?: number;
}

export function HeatmapOverlay({ score = 0 }: Props) {
  const tone = score >= 0.6 ? "var(--color-fault)" : score >= 0.4 ? "var(--color-warn)" : "var(--color-ok)";
  return (
    <div className="pointer-events-none absolute inset-0">
      {/* corner brackets */}
      {[
        "top-2 left-2 border-t-2 border-l-2",
        "top-2 right-2 border-t-2 border-r-2",
        "bottom-2 left-2 border-b-2 border-l-2",
        "bottom-2 right-2 border-b-2 border-r-2",
      ].map((cls, i) => (
        <span
          key={i}
          className={"absolute h-4 w-4 " + cls}
          style={{ borderColor: tone }}
        />
      ))}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 font-mono text-[10px] uppercase tracking-widest"
           style={{ color: tone }}>
        anomaly • {(score * 100).toFixed(0)}%
      </div>
    </div>
  );
}
