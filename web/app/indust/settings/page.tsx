"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import { PageHeader } from "@/components/scada/PageHeader";
import { api } from "@/lib/api";
import type { IndustCategory, IndustStatus } from "@/lib/types";

export default function IndustSettingsPage() {
  const [cats, setCats] = useState<IndustCategory[]>([]);
  const [status, setStatus] = useState<IndustStatus | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    Promise.all([api.indust.categories(), api.indust.status()])
      .then(([c, s]) => {
        setCats(c);
        setStatus(s);
        setThreshold(s.threshold);
      })
      .catch(() => {});
  }, []);

  const select = async (category: string) => {
    setSaving(true);
    try {
      const s = await api.indust.config({ category });
      setStatus(s);
      setThreshold(s.threshold);
    } finally {
      setSaving(false);
    }
  };

  const saveThreshold = async () => {
    setSaving(true);
    try {
      const s = await api.indust.config({ threshold });
      setStatus(s);
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <PageHeader tag="INDUST" title="Settings" subtitle="Pick a category and tune the verdict threshold" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel p-4">
          <div className="label mb-3">Category</div>
          <div className="space-y-2">
            {cats.map((c) => {
              const active = status?.active_category === c.name;
              return (
                <button
                  key={c.name}
                  onClick={() => select(c.name)}
                  disabled={saving || !c.has_model}
                  className={clsx(
                    "w-full text-left p-3 panel transition-colors",
                    active && "border-[var(--color-accent)] bg-[var(--color-bg-elevated)]",
                    !c.has_model && "opacity-40 cursor-not-allowed"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="value text-[14px]">{c.name}</div>
                    <div className="value text-[11px] text-[var(--color-text-dim)]">
                      {c.has_model ? "model ready" : "no model"}
                    </div>
                  </div>
                  {c.description && (
                    <div className="text-[11px] text-[var(--color-text-dim)] mt-1">{c.description}</div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        <div className="panel p-4">
          <div className="label mb-3">Anomaly Threshold</div>
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
            <span>0.00 strict</span>
            <span>1.00 lenient</span>
          </div>
          <button
            onClick={saveThreshold}
            disabled={saving}
            className="mt-4 panel-elevated px-4 py-2 text-[11px] uppercase tracking-widest hover:border-[var(--color-accent)]"
          >
            {saved ? "✓ saved" : saving ? "saving…" : "Apply"}
          </button>
        </div>
      </div>
    </div>
  );
}
