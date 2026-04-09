"use client";

import clsx from "clsx";
import { ReactNode } from "react";

export interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (row: T) => ReactNode;
  className?: string;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  onRowClick?: (row: T) => void;
  emptyMessage?: string;
}

export function HistoryTable<T extends { id?: number | null }>({
  rows,
  columns,
  onRowClick,
  emptyMessage = "No records yet.",
}: Props<T>) {
  return (
    <div className="panel overflow-hidden">
      <table className="w-full text-[12px]">
        <thead>
          <tr className="bg-[var(--color-bg-elevated)] text-left">
            {columns.map((c) => (
              <th
                key={String(c.key)}
                className={clsx("label py-2.5 px-3 border-b border-[var(--color-border)]", c.className)}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="text-center py-10 text-[var(--color-text-mute)] font-mono">
                {emptyMessage}
              </td>
            </tr>
          )}
          {rows.map((row, i) => (
            <tr
              key={row.id ?? i}
              onClick={() => onRowClick?.(row)}
              className={clsx(
                "border-b border-[var(--color-border)] transition-colors",
                onRowClick && "cursor-pointer hover:bg-[var(--color-bg-elevated)]"
              )}
            >
              {columns.map((c) => (
                <td
                  key={String(c.key)}
                  className={clsx("py-2 px-3 value text-[var(--color-text)]", c.className)}
                >
                  {c.render ? c.render(row) : ((row as any)[c.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
