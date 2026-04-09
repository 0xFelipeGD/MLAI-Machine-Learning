"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "@/components/scada/PageHeader";
import { GaugeWidget } from "@/components/scada/GaugeWidget";
import { MeasurementCard } from "@/components/scada/MeasurementCard";
import { StatusIndicator } from "@/components/scada/StatusIndicator";
import { api } from "@/lib/api";
import type { CameraConfig, HealthResponse } from "@/lib/types";

export default function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [camera, setCamera] = useState<CameraConfig | null>(null);

  useEffect(() => {
    const tick = async () => {
      try {
        const [h, c] = await Promise.all([api.health(), api.cameraConfig()]);
        setHealth(h);
        setCamera(c);
      } catch {}
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => clearInterval(id);
  }, []);

  return (
    <div>
      <PageHeader tag="SYSTEM" title="Health" subtitle="Pi 4 telemetry, services, and camera configuration" />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <GaugeWidget label="CPU" value={health?.cpu_percent ?? 0} unit="%" thresholds={{ warn: 60, fault: 85 }} />
        <GaugeWidget label="RAM" value={health?.ram_percent ?? 0} unit="%" thresholds={{ warn: 70, fault: 90 }} />
        <GaugeWidget
          label="Temperature"
          value={health?.temperature_c ?? 0}
          unit="°C"
          max={90}
          thresholds={{ warn: 65, fault: 80 }}
          decimals={1}
        />
        <GaugeWidget label="FPS" value={health?.fps ?? 0} unit="fps" max={15} decimals={1} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel p-4">
          <div className="label mb-3">Services</div>
          <ul className="space-y-2 text-[12px]">
            {[
              { name: "mlai-engine", role: "Camera + Inference", ok: health != null },
              { name: "mlai-api", role: "REST + WebSocket", ok: health != null },
              { name: "mlai-web", role: "Frontend (this UI)", ok: true },
            ].map((s) => (
              <li key={s.name} className="flex items-center justify-between border-b border-[var(--color-border)] pb-2 last:border-0">
                <div>
                  <div className="value">{s.name}</div>
                  <div className="label mt-0.5">{s.role}</div>
                </div>
                <StatusIndicator tone={s.ok ? "ok" : "fault"} label={s.ok ? "RUNNING" : "DOWN"} />
              </li>
            ))}
          </ul>
        </div>

        <div className="panel p-4">
          <div className="label mb-3">Camera</div>
          <div className="grid grid-cols-2 gap-3">
            <MeasurementCard label="Resolution" value={camera ? `${camera.width}×${camera.height}` : "—"} />
            <MeasurementCard label="Target FPS" value={camera?.fps ?? null} />
            <MeasurementCard label="Source" value={camera?.source ?? "—"} />
            <MeasurementCard
              label="Calibrated"
              value={camera ? (camera.calibrated ? "yes" : "no") : "—"}
              unit={camera?.calibrated ? `${camera.px_per_mm?.toFixed(2) ?? ""} px/mm` : undefined}
            />
          </div>
        </div>
      </div>

      <div className="mt-4 panel p-4">
        <div className="label mb-2">Uptime</div>
        <div className="value text-[24px]">
          {health
            ? `${Math.floor(health.uptime_seconds / 3600)}h ${Math.floor((health.uptime_seconds % 3600) / 60)}m`
            : "—"}
        </div>
      </div>
    </div>
  );
}
