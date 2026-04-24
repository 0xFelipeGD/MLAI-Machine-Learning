"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import { api } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

function Pill({ label, value, tone }: { label: string; value: string; tone?: "ok" | "warn" | "fault" | "info" }) {
  const colors: Record<string, string> = {
    ok: "text-[var(--color-ok)]",
    warn: "text-[var(--color-warn)]",
    fault: "text-[var(--color-fault)]",
    info: "text-[var(--color-info)]",
  };
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 panel">
      <span className="label">{label}</span>
      <span className={clsx("value text-[12px] font-medium", tone && colors[tone])}>{value}</span>
    </div>
  );
}

export function Topbar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [now, setNow] = useState<string>("");

  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const h = await api.health();
        if (mounted) setHealth(h);
      } catch {
        if (mounted) setHealth(null);
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    const id2 = setInterval(() => setNow(new Date().toISOString().replace("T", " ").slice(0, 19) + "Z"), 1000);
    return () => {
      mounted = false;
      clearInterval(id);
      clearInterval(id2);
    };
  }, []);

  const cpuTone = !health ? "fault" : health.cpu_percent < 60 ? "ok" : health.cpu_percent < 85 ? "warn" : "fault";
  const ramTone = !health ? "fault" : health.ram_percent < 70 ? "ok" : health.ram_percent < 90 ? "warn" : "fault";

  return (
    <header className="flex items-center justify-between gap-4 border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-5 h-14">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              "dot dot-pulse",
              health ? "text-[var(--color-ok)]" : "text-[var(--color-fault)]"
            )}
          />
          <span className="label">{health ? "Online" : "Offline"}</span>
        </div>
        <div className="ml-2 panel px-3 py-1">
          <span className="text-[11px] font-mono uppercase tracking-wider text-[var(--color-accent)]">
            AGRO
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Pill label="CPU" value={`${health?.cpu_percent.toFixed(0) ?? "--"}%`} tone={cpuTone} />
        <Pill label="RAM" value={`${health?.ram_percent.toFixed(0) ?? "--"}%`} tone={ramTone} />
        <Pill
          label="Temp"
          value={health?.temperature_c != null ? `${health.temperature_c}°C` : "--"}
          tone="info"
        />
        <Pill label="FPS" value={`${health?.fps?.toFixed(1) ?? "0.0"}`} tone="ok" />
        <Pill label="UTC" value={now || "--"} />
      </div>
    </header>
  );
}
