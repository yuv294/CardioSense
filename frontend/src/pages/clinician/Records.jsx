import { useMemo, useState } from "react";
import { loadRecords, removeRecord } from "../../lib/storage.js";
import { useAuth } from "../../lib/useAuth.jsx";
import { BAND } from "../../lib/ui.js";

function toCsv(rows) {
  const head = ["date", "patient", "age", "sex", "systolic", "diastolic", "risk_percent", "band", "flagged"];
  const lines = rows.map((r) => [
    r.at, `"${(r.patient || "").replace(/"/g, '""')}"`, r.inputs.age, r.inputs.male ? "M" : "F",
    r.inputs.sysBP, r.inputs.diaBP, r.result.risk_percent, r.result.band, r.result.flagged_high_risk ? "yes" : "no",
  ].join(","));
  return [head.join(","), ...lines].join("\n");
}

export default function Records() {
  const { user } = useAuth();
  const [rows, setRows] = useState(() => loadRecords(user));
  const [q, setQ] = useState("");

  // change vs the same patient's previous assessment (rows are newest first)
  const withChange = useMemo(() => rows.map((r, i) => {
    const prev = rows.slice(i + 1).find((x) => x.patient === r.patient);
    return { ...r, change: prev ? +(r.result.risk_percent - prev.result.risk_percent).toFixed(1) : null };
  }), [rows]);

  const shown = withChange.filter((r) => (r.patient || "").toLowerCase().includes(q.toLowerCase()));

  function exportCsv() {
    const url = URL.createObjectURL(new Blob([toCsv(shown)], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url; a.download = "cardiosense-records.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <h1 className="font-display text-3xl font-extrabold tracking-tight">Patient records</h1>
        {rows.length > 0 && (
          <div className="flex items-center gap-3">
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search patient"
                   aria-label="Search patient"
                   className="w-56 rounded-lg border border-fg/15 bg-surface px-3 py-2 text-sm placeholder:text-fg/40 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30" />
            <button onClick={exportCsv}
                    className="rounded-lg border border-accent px-4 py-2 text-sm font-semibold text-accent hover:bg-accent/10">
              Export CSV
            </button>
          </div>
        )}
      </div>

      {rows.length === 0 ? (
        <div className="rounded-xl border border-dashed border-fg/20 p-12 text-center text-fg/55">
          No assessments yet. Run one from Assess patient and it is saved here.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-fg/10 bg-surface">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-fg/10 text-fg/55">
              <tr>
                {["Date", "Patient", "Age / sex", "BP", "Risk", "Band", "Screen", "Change", ""].map((h) => (
                  <th key={h} className="px-4 py-3 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-fg/10">
              {shown.map((r) => {
                const b = BAND[r.result.band];
                return (
                  <tr key={r.id} className="hover:bg-fg/[0.03]">
                    <td className="whitespace-nowrap px-4 py-3 text-fg/70">{new Date(r.at).toLocaleDateString()}</td>
                    <td className="px-4 py-3 font-medium">{r.patient}</td>
                    <td className="px-4 py-3">{r.inputs.age} / {r.inputs.male ? "M" : "F"}</td>
                    <td className="px-4 py-3 tabular-nums">{r.inputs.sysBP}/{r.inputs.diaBP}</td>
                    <td className={`px-4 py-3 font-display text-lg font-bold tabular-nums ${b.text}`}>{r.result.risk_percent}%</td>
                    <td className={`px-4 py-3 ${b.text}`}>{b.label}</td>
                    <td className="px-4 py-3">{r.result.flagged_high_risk ? "Flagged" : "Clear"}</td>
                    <td className={`px-4 py-3 tabular-nums ${r.change == null ? "text-fg/35" : r.change > 0 ? "text-high" : r.change < 0 ? "text-low" : "text-fg/60"}`}>
                      {r.change == null ? "–" : `${r.change > 0 ? "+" : ""}${r.change} pt`}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button onClick={() => setRows(removeRecord(user, r.id))}
                              className="text-fg/45 hover:text-high hover:underline">Delete</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
