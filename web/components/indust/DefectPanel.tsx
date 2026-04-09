interface Props {
  verdict: string;
  score: number;
}

export function DefectPanel({ verdict, score }: Props) {
  const reasons: string[] = [];
  if (verdict === "FAIL") {
    if (score > 0.85) reasons.push("Severe anomaly cluster detected");
    else reasons.push("Anomaly score above threshold");
  } else if (verdict === "WARN") {
    reasons.push("Borderline anomaly — re-inspect recommended");
  } else if (verdict === "PASS") {
    reasons.push("No significant defects");
  }
  return (
    <div className="panel p-4">
      <div className="label">Defect Analysis</div>
      <ul className="mt-2 space-y-1 text-[12px]">
        {reasons.map((r, i) => (
          <li key={i} className="flex gap-2 text-[var(--color-text-dim)]">
            <span className="text-[var(--color-accent)]">›</span>
            {r}
          </li>
        ))}
      </ul>
    </div>
  );
}
