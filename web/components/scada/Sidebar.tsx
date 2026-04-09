"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Apple, Cog, Cpu, Factory, Gauge, History, Settings, Sliders } from "lucide-react";

const NAV: { group: string; items: { href: string; label: string; Icon: typeof Gauge }[] }[] = [
  {
    group: "Overview",
    items: [{ href: "/", label: "Dashboard", Icon: Gauge }],
  },
  {
    group: "Indust",
    items: [
      { href: "/indust", label: "Live Inspection", Icon: Factory },
      { href: "/indust/history", label: "History", Icon: History },
      { href: "/indust/settings", label: "Settings", Icon: Sliders },
    ],
  },
  {
    group: "Agro",
    items: [
      { href: "/agro", label: "Live Inspection", Icon: Apple },
      { href: "/agro/history", label: "History", Icon: History },
      { href: "/agro/settings", label: "Settings", Icon: Sliders },
    ],
  },
  {
    group: "System",
    items: [
      { href: "/system", label: "Health", Icon: Cpu },
      { href: "/system/calibration", label: "Calibration", Icon: Cog },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden lg:flex h-screen w-60 flex-col border-r border-[var(--color-border)] bg-[var(--color-bg-elevated)]">
      <div className="px-5 pt-5 pb-4 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-2">
          <Activity size={18} className="text-[var(--color-accent)]" />
          <span className="font-mono text-[15px] font-semibold tracking-tight">MLAI</span>
        </div>
        <div className="label mt-1">v1.0 • Edge Inspection</div>
      </div>
      <nav className="flex-1 overflow-y-auto py-3">
        {NAV.map((g) => (
          <div key={g.group} className="mb-4">
            <div className="label px-5 mb-1">{g.group}</div>
            {g.items.map(({ href, label, Icon }) => {
              const active = pathname === href || (href !== "/" && pathname.startsWith(href));
              return (
                <Link
                  key={href}
                  href={href}
                  className={
                    "flex items-center gap-3 px-5 py-2 text-[13px] transition-colors " +
                    (active
                      ? "bg-[var(--color-bg-panel)] text-[var(--color-text)] border-l-2 border-[var(--color-accent)]"
                      : "text-[var(--color-text-dim)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-panel)] border-l-2 border-transparent")
                  }
                >
                  <Icon size={14} />
                  {label}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>
      <div className="border-t border-[var(--color-border)] px-5 py-3">
        <div className="label">Build</div>
        <div className="value text-[11px] mt-0.5">2026.04.08 • PI4-8GB</div>
      </div>
    </aside>
  );
}
