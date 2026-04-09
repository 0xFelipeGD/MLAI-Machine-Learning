"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface Props {
  bins: { range_mm: string; count: number }[];
}

export function SizeHistogram({ bins }: Props) {
  return (
    <div className="panel p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="label">Size Distribution</div>
        <div className="value text-[10px] text-[var(--color-text-dim)]">millimetres</div>
      </div>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="#262b3a" strokeDasharray="2 2" vertical={false} />
            <XAxis
              dataKey="range_mm"
              tick={{ fill: "#5a607a", fontSize: 10, fontFamily: "JetBrains Mono" }}
              tickLine={false}
              axisLine={{ stroke: "#262b3a" }}
            />
            <YAxis
              tick={{ fill: "#5a607a", fontSize: 10, fontFamily: "JetBrains Mono" }}
              tickLine={false}
              axisLine={{ stroke: "#262b3a" }}
            />
            <Tooltip
              cursor={{ fill: "rgba(0, 212, 255, 0.05)" }}
              contentStyle={{
                background: "#161922",
                border: "1px solid #3a4055",
                fontFamily: "JetBrains Mono",
                fontSize: "11px",
              }}
            />
            <Bar dataKey="count" fill="#00d4ff" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
