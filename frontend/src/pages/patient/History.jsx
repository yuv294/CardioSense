import { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from "recharts";
import { loadRecords, removeRecord, clearRecords } from "../../lib/storage.js";
import { useAuth } from "../../lib/useAuth.jsx";
import { BAND } from "../../lib/ui.js";

export default function PatientHistory() {
  const { user } = useAuth();
  const [items, setItems] = useState(() => loadRecords(user));

  const trend = [...items].reverse().map((r) => ({
    day: new Date(r.at).toLocaleDateString(undefined, { day: "numeric", month: "short" }),
    risk: r.result.risk_percent,
  }));

  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between">
        <h1 className="font-display text-4xl font-extrabold tracking-tight">Your history</h1>
        {items.length > 0 && (
          <button onClick={() => { clearRecords(user); setItems([]); }}
                  className="text-sm font-medium text-high hover:underline">Clear all</button>
        )}
      </div>

      {items.length === 0 ? (
        <div className="rounded-2xl border-2 border-dashed border-fg/15 p-12 text-center text-fg/55">
          Nothing here yet. Check your risk and it will be saved here.
        </div>
      ) : (
        <>
          {items.length > 1 && (
            <section className="rounded-2xl bg-surface p-6">
              <h2 className="mb-4 font-display text-xl font-bold tracking-tight">Your risk over time</h2>
              <div className="h-56">
                <ResponsiveContainer>
                  <LineChart data={trend} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                    <CartesianGrid stroke="#0b1a2b14" vertical={false} />
                    <XAxis dataKey="day" stroke="#5b6b7c" fontSize={12} tickLine={false} />
                    <YAxis stroke="#5b6b7c" fontSize={12} tickLine={false} unit="%" />
                    <ReferenceLine y={10} stroke="#B07008" strokeDasharray="4 4" />
                    <ReferenceLine y={20} stroke="#D02C40" strokeDasharray="4 4" />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Line type="monotone" dataKey="risk" stroke="#0E7C86" strokeWidth={2.5} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          <ul className="divide-y divide-fg/10 rounded-2xl bg-surface">
            {items.map((r) => {
              const b = BAND[r.result.band];
              return (
                <li key={r.id ?? r.at} className="flex items-center gap-5 px-6 py-5">
                  <span className={`w-24 font-display text-4xl font-extrabold tabular-nums ${b.text}`}>
                    {r.result.risk_percent}%
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className={`font-semibold ${b.text}`}>{b.label}</div>
                    <div className="truncate text-sm text-fg/60">
                      {new Date(r.at).toLocaleString()} · {r.inputs.age} yrs · {r.inputs.male ? "male" : "female"} ·
                      BP {r.inputs.sysBP}/{r.inputs.diaBP}
                    </div>
                  </div>
                  <button onClick={() => setItems(removeRecord(user, r.id))}
                          className="text-sm text-fg/50 hover:text-high hover:underline">Delete</button>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}
