import { useEffect, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Legend,
  BarChart, Bar, Cell, ErrorBar,
} from "recharts";
import { api } from "../lib/api.js";
import { MODEL_NAMES, f3 } from "../lib/ui.js";

const C = { accent: "rgb(var(--accent))", base: "rgb(var(--mid))", recal: "rgb(var(--high))", muted: "rgb(var(--fg) / 0.35)",
            grid: "rgb(var(--fg) / 0.08)", text: "rgb(var(--fg) / 0.6)" };
const tip = { contentStyle: { background: "rgb(var(--surface))", border: "1px solid rgb(var(--fg) / 0.15)", borderRadius: 8, fontSize: 12 } };
const name = (m) => MODEL_NAMES[m] || m;

function Card({ title, sub, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-fg/10 bg-surface p-6 ${className}`}>
      <h3 className="font-display text-lg font-bold tracking-tight">{title}</h3>
      {sub && <p className="mt-1 text-sm leading-relaxed text-fg/55">{sub}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

function Delta({ d, label }) {
  const sig = d.d_auroc_ci[0] > 0 || d.d_auroc_ci[1] < 0;
  return (
    <div className="rounded-xl border border-fg/10 p-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-fg/50">{label}</div>
      <div className={`mt-1 font-display text-3xl font-extrabold tabular-nums ${sig && d.d_auroc > 0 ? "text-low" : sig ? "text-high" : ""}`}>
        {d.d_auroc > 0 ? "+" : ""}{d.d_auroc.toFixed(3)}
      </div>
      <div className="text-xs tabular-nums text-fg/60">
        95% CI {d.d_auroc_ci[0].toFixed(3)} to {d.d_auroc_ci[1].toFixed(3)} · p = {d.p_auroc < 0.001 ? "<0.001" : d.p_auroc.toFixed(3)}
      </div>
    </div>
  );
}

export function EvidenceView({ model }) {
  const [e, setE] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { setE(null); setErr(null); api.evidence(model).then(setE).catch((x) => setErr(x.message)); }, [model]);
  if (err) return <p className="text-fg/60">Evidence could not be loaded ({err}).</p>;
  if (!e) return <div className="h-96 animate-pulse rounded-2xl bg-fg/5" />;

  const ev = e.evaluation, nested = e.nested_cv, ds = e.data_summary, abl = e.ablation;
  const served = ev.served_model, base = ev.primary_baseline;
  const main = ev.paired[`${served}-vs-${base}`];
  const vsRecal = ev.paired[`${served}-vs-${base}_recal`];
  const vsRefit = ev.paired[`${served}-vs-refit`];
  const nMain = nested?.paired[`served-vs-${base}`];

  const order = ["ensemble", "gbm", "logreg", "rf", "mlp", "refit", base, `${base}_recal`,
                 ...Object.keys(ev.metrics).filter((m) => m.startsWith("frs") && m !== base && !m.endsWith("_recal"))];
  const rows = order.filter((m, i) => ev.metrics[m] && order.indexOf(m) === i).map((m) => ({ key: m, ...ev.metrics[m] }));
  const bars = rows.filter((r) => !r.key.endsWith("_recal")).map((r) => ({
    name: name(r.key), key: r.key, auroc: +r.auroc.toFixed(3),
    err: [+(r.auroc - r.auroc_ci[0]).toFixed(3), +(r.auroc_ci[1] - r.auroc).toFixed(3)],
  }));
  const lo = Math.floor(Math.min(...rows.map((r) => r.auroc_ci[0])) * 20) / 20;

  const calKeys = [served, base, `${base}_recal`];
  const calColors = [C.accent, C.base, C.recal];
  const calMax = Math.max(...calKeys.flatMap((k) => ev.calibration_curves[k].map((p) => Math.max(p.predicted, p.observed)))) * 100;
  const dca = ev.decision_curve.thresholds.map((t, i) => ({
    t: +(t * 100).toFixed(1), all: ev.decision_curve.treat_all[i], none: 0,
    ...Object.fromEntries(calKeys.map((k) => [k, ev.decision_curve[k][i]])),
  }));
  const dcaMin = -0.2 * Math.max(...dca.map((d) => d[served]));

  const ablRows = abl ? Object.entries(abl.gbm).filter(([k]) => k !== "all inputs")
    .map(([k, v]) => ({ name: k.replace("without ", "− ").replace("with ", "+ "), d: +(v.d_auroc * 1000).toFixed(1) }))
    .sort((a, b) => a.d - b.d) : [];

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-fg/10 bg-surface p-6">
        <div className="text-xs font-semibold uppercase tracking-wide text-accent">{ev.title}</div>
        <p className="mt-2 max-w-3xl text-[15px] leading-relaxed text-fg/80">
          <b>Outcome:</b> {ev.outcome}. <b>Data:</b> {ev.source}. {ds.all.rows.toLocaleString()} rows from{" "}
          {ds.all.participants.toLocaleString()} people, {ds.all.events.toLocaleString()} events. A locked test set of{" "}
          {ev.participants_test.toLocaleString()} people ({ev.events_test} events) was opened once, after every modelling decision.
        </p>
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <Delta d={main} label={`${name(served)} vs ${name(base)}: test AUROC`} />
          {nMain && <Delta d={nMain} label="Same comparison, nested CV (all events)" />}
          <Delta d={vsRefit} label={`vs the published formula refitted: test AUROC`} />
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Discrimination on held-out people" sub="AUROC with 95% participant-level bootstrap intervals.">
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={bars} layout="vertical" margin={{ top: 0, right: 30, bottom: 0, left: 10 }}>
                <CartesianGrid stroke={C.grid} horizontal={false} />
                <XAxis type="number" domain={[lo, "auto"]} tickCount={6} stroke={C.text} fontSize={12} tickLine={false} />
                <YAxis type="category" dataKey="name" width={170} stroke={C.text} fontSize={12} tickLine={false} />
                <Tooltip {...tip} cursor={{ fill: "rgb(var(--fg) / 0.05)" }} />
                <Bar dataKey="auroc" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {bars.map((d) => <Cell key={d.key} fill={d.key === served ? C.accent : d.key === base || d.key.startsWith("frs") || d.key === "pce" ? C.base : C.muted} />)}
                  <ErrorBar dataKey="err" width={4} stroke="rgb(var(--fg) / 0.7)" direction="x" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Calibration" sub="Do predicted risks match what happened? Deciles of predicted risk; the dashed line is perfect agreement.">
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis type="number" dataKey="x" domain={[0, Math.ceil(calMax)]} stroke={C.text} fontSize={12} unit="%"
                       label={{ value: "Predicted", position: "insideBottom", offset: -10, fill: C.text, fontSize: 12 }} />
                <YAxis type="number" domain={[0, Math.ceil(calMax)]} stroke={C.text} fontSize={12} unit="%"
                       label={{ value: "Observed", angle: -90, position: "insideLeft", fill: C.text, fontSize: 12 }} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: Math.ceil(calMax), y: Math.ceil(calMax) }]} stroke={C.text} strokeDasharray="4 4" />
                <Tooltip {...tip} formatter={(v) => `${v}%`} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                {calKeys.map((k, i) => (
                  <Line key={k} name={name(k)} type="linear" stroke={calColors[i]} strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false}
                        data={ev.calibration_curves[k].map((p) => ({ x: +(p.predicted * 100).toFixed(2), y: +(p.observed * 100).toFixed(2) }))}
                        dataKey="y" />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Clinical usefulness (decision curve)"
              sub="Net benefit: true positives found per person, minus false alarms weighted by the treatment threshold. Higher is better; above 'treat all' and 0 means useful.">
          <div className="h-80">
            <ResponsiveContainer>
              <LineChart data={dca} margin={{ top: 8, right: 16, bottom: 18, left: 8 }}>
                <CartesianGrid stroke={C.grid} />
                <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} stroke={C.text} fontSize={12} unit="%"
                       label={{ value: "Risk threshold for action", position: "insideBottom", offset: -10, fill: C.text, fontSize: 12 }} />
                <YAxis domain={[dcaMin, "auto"]} stroke={C.text} fontSize={12} tickFormatter={(v) => v.toFixed(3)} allowDataOverflow />
                <Tooltip {...tip} formatter={(v) => v.toFixed(4)} />
                <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
                <Line dataKey="all" name="Treat everyone" stroke={C.muted} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                <Line dataKey="none" name="Treat no one" stroke={C.text} strokeWidth={1} dot={false} isAnimationActive={false} />
                {calKeys.map((k, i) => <Line key={k} dataKey={k} name={name(k)} stroke={calColors[i]} strokeWidth={2.2} dot={false} isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {ablRows.length > 0 && (
          <Card title="Where the signal comes from"
                sub="Change in cross-validated AUROC (×1000) when a factor is removed (−) or a rejected input is added back (+). Training participants only.">
            <div className="h-80">
              <ResponsiveContainer>
                <BarChart data={ablRows} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 10 }}>
                  <CartesianGrid stroke={C.grid} horizontal={false} />
                  <XAxis type="number" stroke={C.text} fontSize={12} />
                  <YAxis type="category" dataKey="name" width={170} stroke={C.text} fontSize={12} tickLine={false} />
                  <ReferenceLine x={0} stroke={C.text} />
                  <Tooltip {...tip} formatter={(v) => `${v} ×10⁻³`} />
                  <Bar dataKey="d" isAnimationActive={false}>
                    {ablRows.map((r) => <Cell key={r.name} fill={r.d < 0 ? C.accent : C.muted} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
      </div>

      <Card title="Every model on the locked test set"
            sub="Brier: mean squared error of the probabilities (lower is better). E/O: expected ÷ observed events (1 is perfect). Slope: calibration slope (1 is perfect; below 1 means over-confident).">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="text-left text-fg/55">
              <tr>{["Model", "AUROC (95% CI)", "AUPRC", "Brier", "E/O", "Slope"].map((h) => <th key={h} className="pb-2 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-fg/10 tabular-nums">
              {rows.map((r) => (
                <tr key={r.key} className={r.key === served ? "text-accent" : ""}>
                  <td className="py-2.5 pr-3">{name(r.key)}{r.key === served && <span className="ml-2 rounded bg-accent/15 px-1.5 py-0.5 text-[11px]">served</span>}</td>
                  <td className="py-2.5">{f3(r.auroc)} <span className="text-fg/50">({r.auroc_ci[0].toFixed(3)}–{r.auroc_ci[1].toFixed(3)})</span></td>
                  <td className="py-2.5">{f3(r.auprc)}</td>
                  <td className="py-2.5">{r.brier.toFixed(4)}</td>
                  <td className="py-2.5">{r.expected_over_observed.toFixed(2)}</td>
                  <td className="py-2.5">{r.cal_slope.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-sm text-fg/60">
          Versus the recalibrated {name(base)}: ΔAUROC {vsRecal.d_auroc >= 0 ? "+" : ""}{vsRecal.d_auroc.toFixed(3)} (recalibration
          cannot change ranking), ΔBrier {vsRecal.d_brier.toFixed(4)} (95% CI {vsRecal.d_brier_ci[0].toFixed(4)} to {vsRecal.d_brier_ci[1].toFixed(4)}).
        </p>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Overfitting check" sub="AUROC on the data the model was fitted to, in cross-validation, and on the locked test set. Small gaps mean the model generalises.">
          <table className="w-full text-sm tabular-nums">
            <thead className="text-left text-fg/55"><tr>{["Model", "Train", "CV", "Test"].map((h) => <th key={h} className="pb-2 font-medium">{h}</th>)}</tr></thead>
            <tbody className="divide-y divide-fg/10">
              {Object.entries(ev.overfitting_check).map(([m, v]) => (
                <tr key={m}><td className="py-2">{name(m)}</td><td>{f3(v.train_auroc)}</td><td>{f3(v.cv_auroc)}</td><td>{f3(v.test_auroc)}</td></tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Subgroups" sub={`Test-set AUROC and expected ÷ observed events for ${name(served)} and ${name(base)}.`}>
          <div className="max-h-80 overflow-y-auto">
            <table className="w-full text-sm tabular-nums">
              <thead className="sticky top-0 bg-surface text-left text-fg/55">
                <tr>{["Group", "Events", "AUROC model", `AUROC ${base === "pce" ? "PCE" : "FRS"}`, "E/O model", "E/O recal."].map((h) => <th key={h} className="pb-2 font-medium">{h}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-fg/10">
                {ev.subgroups.map((s) => (
                  <tr key={s.by + s.level}>
                    <td className="py-2">{s.by === "exam" ? `Exam ${s.level}` : s.by === "cycle" ? `Cycle ${s.level}` : s.level}</td>
                    <td>{s.events}</td><td>{f3(s[`auroc_${served}`])}</td><td>{f3(s[`auroc_${base}`])}</td>
                    <td>{s[`eo_${served}`].toFixed(2)}</td><td>{s[`eo_${base}_recal`].toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default function Evidence({ initial = "nhanes" }) {
  const [model, setModel] = useState(initial);
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-extrabold tracking-tight sm:text-4xl">The evidence</h1>
          <p className="mt-1 max-w-2xl text-fg/60">Every number here comes from people the models never saw during development.</p>
        </div>
        <div role="tablist" className="inline-flex rounded-lg border border-fg/15 bg-fg/5 p-0.5 text-sm">
          {[["nhanes", "NHANES (external)"], ["framingham", "Framingham"]].map(([k, l]) => (
            <button key={k} role="tab" aria-selected={model === k} onClick={() => setModel(k)}
                    className={`rounded-md px-4 py-2 ${model === k ? "bg-accent font-semibold text-accent-fg" : "text-fg/70 hover:text-fg"}`}>{l}</button>
          ))}
        </div>
      </div>
      <EvidenceView model={model} />
    </div>
  );
}
