"use client";

import { useEffect, useState } from "react";
import { LiveFeed } from "@/components/scada/LiveFeed";
import { MeasurementCard } from "@/components/scada/MeasurementCard";
import { CameraTuner } from "./CameraTuner";
import { FruitCard } from "./FruitCard";
import { SizeHistogram } from "./SizeHistogram";
import { api } from "@/lib/api";
import type { AgroDetection, AgroStats, AgroStatus, WSMessage } from "@/lib/types";

export function AgroDashboard() {
  const [status, setStatus] = useState<AgroStatus | null>(null);
  const [stats, setStats] = useState<AgroStats | null>(null);
  const [live, setLive] = useState<{ detections: AgroDetection[]; total: number; ms: number }>({
    detections: [],
    total: 0,
    ms: 0,
  });

  useEffect(() => {
    const tick = async () => {
      try {
        const [s, st] = await Promise.all([api.agro.status(), api.agro.stats()]);
        setStatus(s);
        setStats(st);
      } catch {}
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, []);

  const onResult = (msg: WSMessage) => {
    if (msg.type !== "agro_result") return;
    setLive({ detections: msg.detections, total: msg.total_count, ms: msg.inference_ms });
  };

  const avgDiameter =
    live.detections.length > 0
      ? live.detections.reduce((s, d) => s + (d.diameter_mm || 0), 0) / live.detections.length
      : 0;

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 xl:col-span-8">
        <LiveFeed onResult={onResult} />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <MeasurementCard label="Detections" value={live.total} />
          <MeasurementCard label="Avg Diameter" value={avgDiameter} unit="mm" />
          <MeasurementCard label="Inference" value={live.ms} unit="ms" />
          <MeasurementCard label="Threshold" value={status?.detection_threshold ?? 0} />
        </div>
        <div className="mt-4">
          <SizeHistogram bins={stats?.size_histogram ?? []} />
        </div>
      </div>

      <div className="col-span-12 xl:col-span-4 flex flex-col gap-3">
        <CameraTuner />
        <div className="label px-1">Live Detections</div>
        <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
          {live.detections.length === 0 ? (
            <div className="panel p-6 text-center text-[var(--color-text-mute)] font-mono text-[11px] uppercase">
              waiting for fruits…
            </div>
          ) : (
            live.detections.map((d, i) => <FruitCard key={i} detection={d} />)
          )}
        </div>
      </div>
    </div>
  );
}
