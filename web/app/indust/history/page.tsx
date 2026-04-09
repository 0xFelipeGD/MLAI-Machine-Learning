"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import { PageHeader } from "@/components/scada/PageHeader";
import { HistoryTable, type Column } from "@/components/scada/HistoryTable";
import { api } from "@/lib/api";
import type { IndustResult } from "@/lib/types";

const VERDICT_TONE: Record<string, string> = {
  PASS: "text-[var(--color-ok)]",
  WARN: "text-[var(--color-warn)]",
  FAIL: "text-[var(--color-fault)]",
};

export default function IndustHistoryPage() {
  const [rows, setRows] = useState<IndustResult[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const limit = 50;

  useEffect(() => {
    api.indust
      .history(limit, offset)
      .then((p) => {
        setRows(p.items);
        setTotal(p.total);
      })
      .catch(() => {});
  }, [offset]);

  const cols: Column<IndustResult>[] = [
    { key: "id", header: "ID", className: "w-12 text-[var(--color-text-dim)]" },
    {
      key: "timestamp",
      header: "Timestamp",
      render: (r) => <span>{r.timestamp.replace("T", " ").slice(0, 19)}</span>,
    },
    { key: "category", header: "Category" },
    {
      key: "anomaly_score",
      header: "Score",
      render: (r) => <span>{r.anomaly_score.toFixed(3)}</span>,
    },
    {
      key: "verdict",
      header: "Verdict",
      render: (r) => <span className={clsx("font-mono", VERDICT_TONE[r.verdict])}>{r.verdict}</span>,
    },
    {
      key: "width_mm",
      header: "W × H mm",
      render: (r) =>
        r.width_mm != null && r.height_mm != null ? `${r.width_mm.toFixed(1)} × ${r.height_mm.toFixed(1)}` : "—",
    },
    {
      key: "inference_ms",
      header: "Inf ms",
      render: (r) => `${r.inference_ms}`,
    },
  ];

  return (
    <div>
      <PageHeader
        tag="INDUST"
        title="Inspection History"
        subtitle={`${total.toLocaleString()} records`}
        right={
          <a
            href="/api/indust/history/export"
            className="panel px-3 py-2 text-[11px] uppercase tracking-wider hover:border-[var(--color-accent)]"
          >
            Export CSV
          </a>
        }
      />
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
