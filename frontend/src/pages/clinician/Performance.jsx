import { useEffect, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  LineChart, Line, ReferenceLine, Cell,
} from "recharts";
import { api } from "../../lib/api.js";

const C = { accent: "#62E2D0", grid: "#e2eaf21f", text: "#9db0c4", base: "#6b7d92", high: "#FF6E7C", low: "#50D098", mid: "#F2BC4A" };
const NAMES = {
  framingham_baseline: "Framingham score", logreg: "Logistic regression", stack: "Stacked ensemble",
  mlp: "Neural network", rf: "Random forest", hgb: "Gradient boosting",
};

function Panel({ title, className = "", children }) {
  return (
    <section className={`rounded-xl border border-fg/10 bg-surface p-6 ${className}`}>
      <h2 className="mb-5 font-display text-lg font-bold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-xl border border-fg/10 bg-surface p-5">
      <div className="font-display text-4xl font-extrabold tabular-nums">{value}</div>
      <div className="mt-1 text-sm text-fg/60">{label}</div>
    </div>
  );
}

export default function Performance() {
  const [m, setM] = useState(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => { api.metrics().then(setM).catch(() => setFailed(true)); }, []);

  if (failed) return <p className="text-fg/60">Metrics could not be loaded. Start the backend and reload.</p>;
  if (!m) return <p className="text-fg/60">Loading…</p>;

  // the table row for the served model shows the calibrated model that the app actually serves
  const rows = m.metrics.map((r) => (r.model === m.served_model.base && m.served_metrics ? { ...r, ...m.served_metrics } : r))
    .sort((a, b) => b.auroc - a.auroc);
  const chart = rows.map((r) => ({ name: NAMES[r.model] || r.model, auroc: +r.auroc.toFixed(3), key: r.model }));
  const calib = m.calibration_curve.mean_predicted.map((p, i) => ({
    predicted: +(p * 100).toFixed(1), observed: +(m.calibration_curve.fraction_positive[i] * 100).toFixed(1),
  }));
  const { tn, fp, fn, tp } = m.confusion_at_threshold;
  const base = m.metrics.find((x) => x.model === "framingham_baseline");
  const served = rows.find((x) => x.model === m.served_model.base);
  const cell = "flex flex-col items-center justify-center rounded-lg py-5";

  return (
    <div className="space-y-8">
      <h1 className="font-display text-3xl font-extrabold tracking-tight">Model performance</h1>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="AUROC, served model" value={served.auroc.toFixed(2)} />
        <Stat label="AUROC, Framingham score" value={base.auroc.toFixed(2)} />
        <Stat label="Sensitivity at threshold" value={`${(m.test_sensitivity * 100).toFixed(0)}%`} />
        <Stat label="Specificity at threshold" value={`${(m.test_specificity * 100).toFixed(0)}%`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="AUROC on held-out patients">
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={chart} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 30 }}>
                <CartesianGrid stroke={C.grid} horizontal={false} />
                <XAxis type="number" domain={[0.5, 0.75]} stroke={C.text} fontSize={12} tickLine={false} />
                <YAxis type="category" dataKey="name" width={140} stroke={C.text} fontSize={12} tickLine={false} />
                <Tooltip cursor={{ fill: "#ffffff10" }} contentStyle={{ background: "#111c2a", border: "1px solid #ffffff22", borderRadius: 8 }} />
                <Bar dataKey="auroc" radius={[0, 4, 4, 0]}>
                  {chart.map((d) => (
                    <Cell key={d.key} fill={d.key === "framingham_baseline" ? C.mid : d.key === m.served_model.base ? C.accent : C.base} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Calibration">
          <div className="h-72">
            <ResponsiveContainer>
              <LineChart data={calib} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis type="number" dataKey="predicted" domain={[0, 45]} ticks={[0, 10, 20, 30, 40]} stroke={C.text} fontSize={12} tickLine={false} unit="%"
                       label={{ value: "Predicted risk", position: "insideBottom", offset: -10, fill: C.text, fontSize: 12 }} />
                <YAxis type="number" domain={[0, 45]} ticks={[0, 10, 20, 30, 40]} stroke={C.text} fontSize={12} tickLine={false} unit="%"
                       label={{ value: "Observed", angle: -90, position: "insideLeft", fill: C.text, fontSize: 12 }} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 45, y: 45 }]} stroke={C.text} strokeDasharray="4 4" />
                <Tooltip formatter={(v) => `${v}%`} contentStyle={{ background: "#111c2a", border: "1px solid #ffffff22", borderRadius: 8 }} />
                <Line type="linear" dataKey="observed" stroke={C.accent} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title={`Outcomes at the ${(m.operating_threshold * 100).toFixed(0)}% threshold`}>
          <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-2 text-center text-sm">
            <span />
            <span className="text-fg/60">Predicted clear</span>
            <span className="text-fg/60">Predicted flagged</span>
            <span className="pr-2 text-right text-fg/60">No event</span>
            <div className={`${cell} bg-low/15`}><b className="font-display text-3xl tabular-nums text-low">{tn}</b><span className="text-fg/60">correctly cleared</span></div>
            <div className={`${cell} bg-mid/15`}><b className="font-display text-3xl tabular-nums text-mid">{fp}</b><span className="text-fg/60">false alarms</span></div>
            <span className="pr-2 text-right text-fg/60">Had event</span>
            <div className={`${cell} bg-high/15`}><b className="font-display text-3xl tabular-nums text-high">{fn}</b><span className="text-fg/60">missed</span></div>
            <div className={`${cell} bg-low/15`}><b className="font-display text-3xl tabular-nums text-low">{tp}</b><span className="text-fg/60">correctly flagged</span></div>
          </div>
        </Panel>

        <Panel title="All models">
          <table className="w-full text-sm">
            <thead className="text-left text-fg/55">
              <tr><th className="pb-2 font-medium">Model</th><th className="pb-2 font-medium">AUROC (95% CI)</th><th className="pb-2 text-right font-medium">AUPRC</th></tr>
            </thead>
            <tbody className="divide-y divide-fg/10">
              {rows.map((r) => (
                <tr key={r.model}>
                  <td className="py-2.5">{NAMES[r.model] || r.model}{r.model === m.served_model.base && <span className="ml-2 text-xs text-accent">served</span>}</td>
                  <td className="py-2.5 tabular-nums">{r.auroc.toFixed(3)} <span className="text-fg/50">({r.auroc_ci_low.toFixed(2)}–{r.auroc_ci_high.toFixed(2)})</span></td>
                  <td className="py-2.5 text-right tabular-nums">{r.auprc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
    </div>
  );
}
