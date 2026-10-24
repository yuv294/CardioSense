import { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { loadRecords, removeRecord, clearRecords } from "../../lib/storage.js";
import { useAuth } from "../../lib/useAuth.jsx";
import { BAND, CALCULATORS } from "../../lib/ui.js";

export default function PatientHistory() {
  const { user } = useAuth();
  const [items, setItems] = useState(() => loadRecords(user));
  const byModel = (m) => [...items].reverse().filter((r) => (r.model || "framingham") === m).map((r) => ({
    day: new Date(r.at).toLocaleDateString(undefined, { day: "numeric", month: "short" }), risk: r.result.risk_percent,
  }));

  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between">
        <h1 className="font-display text-4xl font-extrabold tracking-tight">Your history</h1>
        {items.length > 0 && (
          <button onClick={() => { clearRecords(user); setItems([]); }} className="text-sm font-medium text-high hover:underline">Clear all</button>
        )}
      </div>

      {items.length === 0 ? (
        <div className="rounded-2xl border-2 border-dashed border-fg/15 p-12 text-center text-fg/55">
          Nothing here yet. Check your risk and it will be saved here, on this device only.
        </div>
      ) : (
        <>
          <div className="grid gap-6 md:grid-cols-2">
            {Object.keys(CALCULATORS).map((m) => {
              const trend = byModel(m);
              if (trend.length < 2) return null;
              return (
                <section key={m} className="rounded-2xl border border-fg/10 bg-surface p-6">
                  <h2 className="mb-4 font-display text-lg font-bold tracking-tight">{CALCULATORS[m].short}: risk over time</h2>
                  <div className="h-48">
                    <ResponsiveContainer>
                      <LineChart data={trend} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                        <CartesianGrid stroke="rgb(var(--fg) / 0.08)" vertical={false} />
                        <XAxis dataKey="day" stroke="rgb(var(--fg) / 0.5)" fontSize={12} tickLine={false} />
                        <YAxis stroke="rgb(var(--fg) / 0.5)" fontSize={12} tickLine={false} unit="%" />
                        <Tooltip formatter={(v) => `${v}%`} />
                        <Line type="monotone" dataKey="risk" stroke="rgb(var(--accent))" strokeWidth={2.5} dot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              );
            })}
          </div>
          <ul className="divide-y divide-fg/10 rounded-2xl border border-fg/10 bg-surface">
            {items.map((r) => {
              const b = BAND[r.result.band] || BAND.intermediate;
              return (
                <li key={r.id ?? r.at} className="flex items-center gap-5 px-6 py-5">
                  <span className={`w-24 font-display text-4xl font-extrabold tabular-nums ${b.text}`}>{r.result.risk_percent}%</span>
                  <div className="min-w-0 flex-1">
                    <div className={`font-semibold ${b.text}`}>{r.result.band_label || r.result.band}
                      <span className="ml-2 text-sm font-normal text-fg/55">{r.result.outcome || "10-year CHD event"}</span></div>
                    <div className="truncate text-sm text-fg/60">
                      {new Date(r.at).toLocaleString()} · {r.inputs.age} yrs · BP {r.inputs.sysBP}/{r.inputs.diaBP}
                    </div>
                  </div>
                  <button onClick={() => setItems(removeRecord(user, r.id))} className="text-sm text-fg/50 hover:text-high hover:underline">Delete</button>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}
