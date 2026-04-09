interface Props {
  label: string;
  value: number | string | null | undefined;
  unit?: string;
  delta?: string;
}

export function MeasurementCard({ label, value, unit, delta }: Props) {
  const display = value === null || value === undefined ? "—" : typeof value === "number" ? value.toFixed(2) : value;
  return (
    <div className="panel p-3">
      <div className="label">{label}</div>
      <div className="flex items-baseline gap-1 mt-2">
        <div className="value text-[22px] text-[var(--color-text)]">{display}</div>
        {unit && <div className="label">{unit}</div>}
      </div>
      {delta && <div className="value text-[10px] text-[var(--color-text-dim)] mt-0.5">{delta}</div>}
    </div>
  );
}
