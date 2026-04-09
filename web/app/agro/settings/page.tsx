"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "@/components/scada/PageHeader";
import { api } from "@/lib/api";
import type { AgroStatus } from "@/lib/types";

export default function AgroSettingsPage() {
  const [status, setStatus] = useState<AgroStatus | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [classes, setClasses] = useState("apple,orange,tomato");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.agro
      .status()
      .then((s) => {
        setStatus(s);
        setThreshold(s.detection_threshold);
        setClasses(s.fruit_classes.join(","));
      })
      .catch(() => {});
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const s = await api.agro.config({
        detection_threshold: threshold,
        fruit_classes: classes.split(",").map((c) => c.trim()).filter(Boolean),
      });
      setStatus(s);
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <PageHeader tag="AGRO" title="Settings" subtitle="Detection threshold and active fruit classes" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel p-4">
          <div className="label mb-3">Detection Threshold</div>
          <div className="value text-[36px]">{threshold.toFixed(2)}</div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full mt-3 accent-[var(--color-accent)]"
          />
          <div className="flex justify-between text-[10px] text-[var(--color-text-mute)] font-mono mt-1">
            <span>0.00 keep all</span>
            <span>1.00 strict</span>
          </div>
        </div>

        <div className="panel p-4">
          <div className="label mb-3">Active Fruit Classes</div>
          <input
            value={classes}
            onChange={(e) => setClasses(e.target.value)}
            placeholder="apple,orange,tomato"
            className="w-full bg-[var(--color-bg)] border border-[var(--color-border-strong)] px-3 py-2 font-mono text-[12px] focus:border-[var(--color-accent)] focus:outline-none"
          />
          <div className="text-[10px] text-[var(--color-text-mute)] mt-2">
            Comma-separated. Must match the labels the detector was trained on.
          </div>
        </div>
      </div>

      <button
        onClick={save}
        disabled={saving}
        className="mt-4 panel-elevated px-4 py-2 text-[11px] uppercase tracking-widest hover:border-[var(--color-accent)]"
      >
        {saved ? "✓ saved" : saving ? "saving…" : "Apply"}
      </button>
    </div>
  );
}
