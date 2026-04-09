import clsx from "clsx";

interface Props {
  label: string;
  value: number;
  unit?: string;
  min?: number;
  max?: number;
  thresholds?: { warn: number; fault: number };
  decimals?: number;
}

export function GaugeWidget({ label, value, unit, min = 0, max = 100, thresholds, decimals = 0 }: Props) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const tone = thresholds
    ? value >= thresholds.fault
      ? "fault"
      : value >= thresholds.warn
        ? "warn"
        : "ok"
    : "info";

  const colors: Record<string, string> = {
    ok: "var(--color-ok)",
    warn: "var(--color-warn)",
    fault: "var(--color-fault)",
    info: "var(--color-accent)",
  };
  const color = colors[tone];

  // SVG arc parameters
  const radius = 54;
  const stroke = 8;
  const c = 2 * Math.PI * radius;
  const offset = c * (1 - pct * 0.75); // 270° arc
  const rotation = 135;

  return (
    <div className="panel p-4 flex flex-col items-center justify-between min-h-[170px]">
      <div className="label">{label}</div>
      <div className="relative">
        <svg width="140" height="140" viewBox="0 0 140 140">
          <g transform={`rotate(${rotation} 70 70)`}>
            <circle
              cx="70"
              cy="70"
              r={radius}
              fill="none"
              stroke="var(--color-border-strong)"
              strokeWidth={stroke}
              strokeDasharray={c}
              strokeDashoffset={c * 0.25}
              strokeLinecap="butt"
            />
            <circle
              cx="70"
              cy="70"
              r={radius}
              fill="none"
              stroke={color}
              strokeWidth={stroke}
              strokeDasharray={c}
              strokeDashoffset={offset}
              strokeLinecap="butt"
              style={{ transition: "stroke-dashoffset 0.4s ease, stroke 0.4s ease" }}
            />
          </g>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-3">
          <div className={clsx("value text-[28px] leading-none")} style={{ color }}>
            {value.toFixed(decimals)}
          </div>
          {unit && <div className="label mt-1">{unit}</div>}
        </div>
      </div>
      <div className="flex w-full justify-between text-[10px] text-[var(--color-text-mute)] font-mono mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
