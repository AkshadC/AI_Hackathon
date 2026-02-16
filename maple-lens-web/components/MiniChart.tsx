"use client";

import { ResponsiveContainer, AreaChart, Area, XAxis, Tooltip } from "recharts";

type Point = { date: string; count: number };

export default function MiniChart({ data }: { data: Point[] }) {
  if (!data || data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={60}>
      <AreaChart data={data}>
        <Area
          type="monotone"
          dataKey="count"
          stroke="#C45D1A"
          fill="#C45D1A"
          fillOpacity={0.2}
        />
        <XAxis dataKey="date" hide />
        <Tooltip
          contentStyle={{
            fontSize: 11,
            borderRadius: 8,
            background: "rgba(255,255,255,0.9)",
            border: "none",
            boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
          }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
