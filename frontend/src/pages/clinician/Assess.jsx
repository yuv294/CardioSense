import { useState } from "react";
import { BAND } from "../../lib/ui.js";
import { saveRecord } from "../../lib/storage.js";
import { useAuth } from "../../lib/useAuth.jsx";
import RiskForm from "../../components/RiskForm.jsx";
import RiskMeter from "../../components/RiskMeter.jsx";
import Contributions from "../../components/Contributions.jsx";

function Panel({ title, className = "", children }) {
  return (
    <section className={`rounded-xl border border-fg/10 bg-surface p-6 ${className}`}>
      <h2 className="mb-5 font-display text-lg font-bold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

function agreement(probs) {
  const v = probs.map((p) => p.risk_percent);
  const spread = Math.max(...v) - Math.min(...v);
  return { spread, level: spread < 15 ? "High" : spread < 30 ? "Moderate" : "Low",
           cls: spread < 15 ? "text-low" : spread < 30 ? "text-mid" : "text-high" };
}

export default function ClinicianAssess() {
  const { user } = useAuth();
  const [patient, setPatient] = useState("");
  const [result, setResult] = useState(null);
  const [shownPatient, setShownPatient] = useState("");

  function onResult(res, form) {
    const label = patient.trim() || "Unnamed patient";
    setResult(res); setShownPatient(label);
    saveRecord(user, {
      at: new Date().toISOString(), patient: label, inputs: form,
      result: { risk: res.risk, risk_percent: res.risk_percent, band: res.band,
                flagged_high_risk: res.flagged_high_risk },
    });
  }

  const band = result ? BAND[result.band] : null;
  const cli = result?.clinician;
  const agree = cli ? agreement(cli.model_probabilities) : null;

  return (
    <div className="space-y-8">
      <h1 className="font-display text-3xl font-extrabold tracking-tight">Assess patient</h1>

      <div className="grid items-start gap-8 xl:grid-cols-[1fr_390px]">
        <RiskForm mode="clinician" wide submitLabel="Run assessment" onResult={onResult}
          header={
            <div>
              <label htmlFor="pid" className="mb-1.5 block text-sm font-medium">Patient name or ID</label>
              <input id="pid" value={patient} onChange={(e) => setPatient(e.target.value)}
                     className="w-full max-w-sm rounded-lg border border-fg/15 bg-surface px-3 py-2.5 text-[15px] focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30" />
            </div>
          } />

        <aside className="xl:sticky xl:top-8">
          {!result ? (
            <div className="rounded-xl border border-dashed border-fg/20 p-10 text-center text-fg/50">
              The decision panel appears here after you run an assessment.
            </div>
          ) : (
            <section className="rounded-xl border border-fg/10 bg-surface p-6">
              <div className="text-sm text-fg/60">{shownPatient}</div>
              <div className="mt-1 flex items-end gap-4">
                <span className={`font-display text-7xl font-extrabold leading-none tabular-nums ${band.text}`}>
                  {result.risk_percent}%
                </span>
                <span className={`pb-1 font-display text-xl font-bold ${band.text}`}>{band.label}</span>
              </div>
              <RiskMeter risk={result.risk} threshold={result.threshold} />
              <div className={`rounded-lg px-4 py-3 text-sm font-medium ${
                result.flagged_high_risk ? "bg-high/15 text-high" : "bg-low/15 text-low"}`}>
                {result.flagged_high_risk
                  ? `Above the ${result.threshold_percent}% screening threshold. Flag for follow-up.`
                  : `Below the ${result.threshold_percent}% screening threshold.`}
              </div>
              <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
                <div>
                  <dt className="text-fg/55">Model agreement</dt>
                  <dd className={`font-semibold ${agree.cls}`}>{agree.level} ({agree.spread.toFixed(0)} pt spread)</dd>
                </div>
                <div>
                  <dt className="text-fg/55">Risk percentile</dt>
                  <dd className="font-semibold">{result.risk_percentile}th of cohort</dd>
                </div>
              </dl>
              {result.assumed_fields.length > 0 && (
                <p className="mt-5 rounded-lg bg-mid/10 px-3.5 py-2.5 text-sm text-fg/80">
                  Defaults used for: {result.assumed_fields.join(", ")}.
                </p>
              )}
            </section>
          )}
        </aside>
      </div>

      {result && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Raw model outputs">
            <div className="space-y-3">
              {cli.model_probabilities.map((m) => (
                <div key={m.model} className="flex items-center gap-3 text-sm">
                  <span className="w-40 shrink-0 text-fg/85">{m.label}</span>
                  <div className="relative h-3 flex-1 rounded-full bg-fg/10">
                    <div className="absolute inset-y-0 left-0 rounded-full bg-accent/70" style={{ width: `${Math.min(m.risk_percent, 100)}%` }} />
                  </div>
                  <span className="w-12 text-right tabular-nums">{m.risk_percent}%</span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Similar patients in the cohort">
            {cli.subgroup ? (
              <div>
                <div className="font-display text-5xl font-extrabold tabular-nums">
                  {(cli.subgroup.event_rate * 100).toFixed(0)}%
                </div>
                <p className="mt-2 text-[15px] leading-relaxed text-fg/75">
                  developed heart disease within ten years among the {cli.subgroup.n} people in
                  the {cli.subgroup.key.replace("_", " ")} group. This patient's estimate is {result.risk_percent}%.
                </p>
              </div>
            ) : (
              <p className="text-fg/60">This subgroup is too small for a stable rate.</p>
            )}
          </Panel>

          <Panel title="Every factor's contribution" className="lg:col-span-2">
            <Contributions items={cli.all_contributions} max={23} />
          </Panel>
        </div>
      )}
    </div>
  );
}
