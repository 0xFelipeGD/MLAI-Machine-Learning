"use client";

import { useEffect, useState } from "react";
import { PageHeader } from "@/components/scada/PageHeader";
import { HistoryTable, type Column } from "@/components/scada/HistoryTable";
import { api } from "@/lib/api";
import type { AgroResult } from "@/lib/types";

export default function AgroHistoryPage() {
  const [rows, setRows] = useState<AgroResult[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const limit = 50;

  useEffect(() => {
    api.agro
      .history(limit, offset)
      .then((p) => {
        setRows(p.items);
        setTotal(p.total);
      })
      .catch(() => {});
  }, [offset]);

  const cols: Column<AgroResult>[] = [
    { key: "id", header: "ID", className: "w-12 text-[var(--color-text-dim)]" },
    { key: "timestamp", header: "Timestamp", render: (r) => r.timestamp.replace("T", " ").slice(0, 19) },
    { key: "total_detections", header: "Count" },
    { key: "avg_diameter_mm", header: "Avg ⌀ mm", render: (r) => r.avg_diameter_mm.toFixed(1) },
    {
      key: "detections",
      header: "Classes",
      render: (r) => Array.from(new Set(r.detections.map((d) => d.fruit_class))).join(", ") || "—",
    },
    { key: "inference_ms", header: "Inf ms" },
  ];

  return (
    <div>
      <PageHeader tag="AGRO" title="History" subtitle={`${total.toLocaleString()} records`} />
      <HistoryTable rows={rows} columns={cols} />
      <div className="flex items-center justify-end gap-2 mt-4 text-[11px]">
        <button
          className="panel px-3 py-1.5 disabled:opacity-30"
          onClick={() => setOffset(Math.max(0, offset - limit))}
          disabled={offset === 0}
        >
          ‹ Prev
        </button>
        <span className="value text-[var(--color-text-dim)]">
          {offset + 1}–{Math.min(offset + limit, total)} / {total}
        </span>
        <button
          className="panel px-3 py-1.5 disabled:opacity-30"
          onClick={() => setOffset(offset + limit)}
          disabled={offset + limit >= total}
        >
          Next ›
        </button>
      </div>
    </div>
  );
}
