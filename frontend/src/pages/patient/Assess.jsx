import { useState } from "react";
import { BAND } from "../../lib/ui.js";
import { api } from "../../lib/api.js";
import { saveRecord } from "../../lib/storage.js";
import { generateReport } from "../../lib/report.js";
import { useAuth } from "../../lib/useAuth.jsx";
import RiskForm from "../../components/RiskForm.jsx";
import RiskMeter from "../../components/RiskMeter.jsx";
import Contributions from "../../components/Contributions.jsx";
import PopulationCompare from "../../components/PopulationCompare.jsx";

function Panel({ title, children }) {
  return (
    <section className="rounded-2xl bg-surface p-6 shadow-[0_1px_0_rgb(var(--fg)/0.06),0_12px_32px_-16px_rgb(var(--fg)/0.18)]">
      <h2 className="mb-5 font-display text-xl font-bold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

export default function PatientAssess() {
  const { user } = useAuth();
  const [result, setResult] = useState(null);
  const [inputs, setInputs] = useState(null);
  const [schema, setSchema] = useState(null);

  function onResult(res, form) {
    setResult(res); setInputs(form);
    saveRecord(user, { at: new Date().toISOString(), inputs: form, result: res });
    if (!schema) api.schema().then(setSchema).catch(() => {});
  }

  const band = result ? BAND[result.band] : null;

  return (
    <div className="space-y-10">
      <h1 className="font-display text-4xl font-extrabold tracking-tight">Check your heart risk</h1>

      <div className="grid items-start gap-10 lg:grid-cols-[1.15fr_1fr]">
        <RiskForm mode="patient" onResult={onResult} submitLabel="Check my risk" />

        <aside className="lg:sticky lg:top-8">
          {!result ? (
            <div className="rounded-2xl border-2 border-dashed border-fg/15 p-10 text-center text-fg/55">
              Fill in the form to see your result.
            </div>
          ) : (
            <section className="overflow-hidden rounded-2xl bg-surface shadow-[0_1px_0_rgb(var(--fg)/0.06),0_12px_32px_-16px_rgb(var(--fg)/0.18)]">
              <div className="p-7">
                <div className={`font-display text-8xl font-extrabold leading-none tracking-tight ${band.text}`}>
                  {result.risk_percent}<span className="text-5xl">%</span>
                </div>
                <div className={`mt-3 font-display text-2xl font-bold ${band.text}`}>{band.label}</div>
                <p className="mt-2 text-[15px] leading-relaxed text-fg/75">{band.meaning}</p>
                <RiskMeter risk={result.risk} />
                {result.assumed_fields.length > 0 && (
                  <p className="rounded-lg bg-mid/10 px-3.5 py-2.5 text-sm text-fg/80">
                    Typical values were used for: {result.assumed_fields.join(", ")}.
                  </p>
                )}
              </div>
              <button onClick={() => schema && generateReport({ result, inputs, schema, name: user.name })}
                      disabled={!schema}
                      className="w-full border-t border-fg/10 bg-fg/[0.03] py-4 text-sm font-semibold text-accent transition hover:bg-fg/[0.06] disabled:opacity-50">
                Download report (PDF)
              </button>
            </section>
          )}
        </aside>
      </div>

      {result && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="What drove your result">
            <Contributions items={result.top_factors.map((f) => ({
              ...f, contribution: f.direction === "increases" ? f.weight : -f.weight,
            }))} />
          </Panel>
          <Panel title="How you compare">
            <PopulationCompare population={result.population} riskPercentile={result.risk_percentile} />
          </Panel>
        </div>
      )}
    </div>
  );
}
