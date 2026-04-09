import { ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: string;
  tag?: string;
  right?: ReactNode;
}

export function PageHeader({ title, subtitle, tag, right }: Props) {
  return (
    <div className="flex items-end justify-between mb-5">
      <div>
        {tag && <div className="label mb-1">{tag}</div>}
        <h1 className="text-[22px] font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="text-[12px] text-[var(--color-text-dim)] mt-1">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}
