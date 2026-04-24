"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { GaugeWidget } from "@/components/scada/GaugeWidget";
import { PageHeader } from "@/components/scada/PageHeader";
import { StatusIndicator } from "@/components/scada/StatusIndicator";
import { LiveFeed } from "@/components/scada/LiveFeed";
import { api } from "@/lib/api";
import type { HealthResponse, AgroStatus } from "@/lib/types";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [agro, setAgro] = useState<AgroStatus | null>(null);

  useEffect(() => {
    const tick = async () => {
      try {
        const [h, a] = await Promise.all([api.health(), api.agro.status()]);
        setHealth(h);
        setAgro(a);
      } catch {}
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div>
      <PageHeader
        tag="Overview"
        title="System Dashboard"
        subtitle="Edge AI fruit inspection — live status"
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <GaugeWidget
          label="CPU Load"
          value={health?.cpu_percent ?? 0}
          unit="%"
          thresholds={{ warn: 60, fault: 85 }}
        />
        <GaugeWidget
          label="RAM"
          value={health?.ram_percent ?? 0}
          unit="%"
          thresholds={{ warn: 70, fault: 90 }}
        />
        <GaugeWidget
          label="Temperature"
          value={health?.temperature_c ?? 0}
          unit="°C"
          max={90}
          thresholds={{ warn: 65, fault: 80 }}
          decimals={1}
        />
        <GaugeWidget label="Live FPS" value={health?.fps ?? 0} unit="fps" max={15} decimals={1} />
      </div>

      <div className="grid grid-cols-1 gap-4">
        <Link href="/agro" className="block">
          <div className="panel p-4 hover:border-[var(--color-accent)] transition-colors h-full">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="label">Module — AGRO</div>
                <h3 className="text-[18px] font-semibold mt-1">Fruit Detection &amp; Grading</h3>
              </div>
              <StatusIndicator
                tone={agro?.running ? "ok" : "idle"}
                label={agro?.running ? "ACTIVE" : "STANDBY"}
              />
            </div>
            <div className="grid grid-cols-3 gap-3 text-[12px] text-[var(--color-text-dim)]">
              <div>
                <div className="label">Classes</div>
                <div className="value text-[14px] text-[var(--color-text)] mt-1">
                  {agro?.fruit_classes?.length ?? 0}
                </div>
              </div>
              <div>
                <div className="label">Threshold</div>
                <div className="value text-[14px] text-[var(--color-text)] mt-1">
                  {agro?.detection_threshold?.toFixed(2) ?? "—"}
                </div>
              </div>
              <div>
                <div className="label">Last Count</div>
                <div className="value text-[14px] text-[var(--color-text)] mt-1">
                  {agro?.last_result?.total_detections ?? 0}
                </div>
              </div>
            </div>
          </div>
        </Link>
      </div>

      <div className="mt-6">
        <LiveFeed />
      </div>
    </div>
  );
}
