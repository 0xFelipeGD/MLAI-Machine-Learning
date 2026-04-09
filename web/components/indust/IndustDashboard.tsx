"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import { LiveFeed } from "@/components/scada/LiveFeed";
import { GaugeWidget } from "@/components/scada/GaugeWidget";
import { MeasurementCard } from "@/components/scada/MeasurementCard";
import { StatusIndicator } from "@/components/scada/StatusIndicator";
import { HeatmapOverlay } from "@/components/scada/HeatmapOverlay";
import { DefectPanel } from "./DefectPanel";
import { api } from "@/lib/api";
import type { IndustStatus, WSMessage } from "@/lib/types";

export function IndustDashboard() {
  const [status, setStatus] = useState<IndustStatus | null>(null);
  const [live, setLive] = useState<{ score: number; verdict: string; ms: number; w?: number; h?: number; a?: number }>({
    score: 0,
    verdict: "—",
    ms: 0,
  });

  useEffect(() => {
    const tick = async () => {
      try {
        setStatus(await api.indust.status());
      } catch {}
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, []);

  const onResult = (msg: WSMessage) => {
    if (msg.type !== "indust_result") return;
    setLive({
      score: msg.anomaly_score,
      verdict: msg.verdict,
      ms: msg.inference_ms,
      w: msg.measurements?.width_mm,
      h: msg.measurements?.height_mm,
      a: msg.measurements?.area_mm2,
    });
  };

  const verdictTone =
    live.verdict === "PASS" ? "ok" : live.verdict === "WARN" ? "warn" : live.verdict === "FAIL" ? "fault" : "idle";

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 xl:col-span-8">
        <div className="relative">
          <LiveFeed module="INDUST" onResult={onResult} />
          <HeatmapOverlay score={live.score} />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <MeasurementCard label="Width" value={live.w ?? null} unit="mm" />
          <MeasurementCard label="Height" value={live.h ?? null} unit="mm" />
          <MeasurementCard label="Area" value={live.a ?? null} unit="mm²" />
          <MeasurementCard label="Inference" value={live.ms} unit="ms" />
        </div>
      </div>

      <div className="col-span-12 xl:col-span-4 flex flex-col gap-4">
        <div className="panel p-4">
          <div className="flex items-center justify-between">
            <div className="label">Verdict</div>
            <StatusIndicator tone={verdictTone as any} label={live.verdict} />
          </div>
          <div
            className={clsx(
              "value text-[44px] mt-2",
              live.verdict === "PASS" && "text-[var(--color-ok)]",
              live.verdict === "WARN" && "text-[var(--color-warn)]",
              live.verdict === "FAIL" && "text-[var(--color-fault)]"
            )}
          >
            {live.verdict}
          </div>
          <div className="mt-1 value text-[11px] text-[var(--color-text-dim)]">
            {status?.active_category ?? "—"} • threshold {(status?.threshold ?? 0).toFixed(2)}
          </div>
        </div>

        <GaugeWidget
          label="Anomaly Score"
          value={live.score * 100}
          unit="%"
          decimals={1}
          thresholds={{ warn: 40, fault: 60 }}
        />

        <DefectPanel verdict={live.verdict} score={live.score} />
      </div>
    </div>
  );
}
