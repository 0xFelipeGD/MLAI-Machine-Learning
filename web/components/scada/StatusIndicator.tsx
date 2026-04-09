import clsx from "clsx";

type Tone = "ok" | "warn" | "fault" | "info" | "idle";

const TONE_CLASS: Record<Tone, string> = {
  ok: "text-[var(--color-ok)]",
  warn: "text-[var(--color-warn)]",
  fault: "text-[var(--color-fault)]",
  info: "text-[var(--color-info)]",
  idle: "text-[var(--color-text-mute)]",
};

export function StatusIndicator({
  tone = "idle",
  label,
  pulse = true,
}: {
  tone?: Tone;
  label?: string;
  pulse?: boolean;
}) {
  return (
    <div className="inline-flex items-center gap-2">
      <span className={clsx("dot", pulse && "dot-pulse", TONE_CLASS[tone])} />
      {label && (
        <span className={clsx("font-mono text-[11px] uppercase tracking-wider", TONE_CLASS[tone])}>
          {label}
        </span>
      )}
    </div>
  );
}
