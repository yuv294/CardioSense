import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api.js";
import { saveRecord } from "../../lib/storage.js";
import { generateReport } from "../../lib/report.js";
import { useAuth } from "../../lib/useAuth.jsx";
import RiskForm from "../../components/RiskForm.jsx";
import ResultCard from "../../components/ResultCard.jsx";
import Contributions from "../../components/Contributions.jsx";
import WhatIf from "../../components/WhatIf.jsx";
import ModelSwitch from "../../components/ModelSwitch.jsx";

export function Panel({ title, sub, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-fg/10 bg-surface p-6 ${className}`}>
      <h2 className="font-display text-xl font-bold tracking-tight">{title}</h2>
      {sub && <p className="mt-1 text-sm text-fg/55">{sub}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

export default function PatientAssess() {
  const { user } = useAuth();
  const [model, setModel] = useState("framingham");
  const [result, setResult] = useState(null);
  const [inputs, setInputs] = useState(null);
  const [schema, setSchema] = useState(null);
  const below = useRef(null);

  useEffect(() => { setResult(null); api.schema(model).then(setSchema).catch(() => {}); }, [model]);

  function onResult(res, form) {
    setResult(res); setInputs(form);
    saveRecord(user, { at: new Date().toISOString(), model, inputs: form,
                       result: { risk_percent: res.risk_percent, band: res.band, band_label: res.band_label, outcome: res.outcome } });
    if (window.innerWidth < 1024) setTimeout(() => below.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }

  return (
    <div className="space-y-10">
      <header className="space-y-5">
        <h1 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl">Check your heart risk</h1>
        <ModelSwitch value={model} onChange={setModel} />
      </header>

      <div className="grid items-start gap-10 lg:grid-cols-[1.1fr_1fr]">
        <RiskForm model={model} mode="patient" onResult={onResult} submitLabel="Show my risk" />
        <aside ref={below} className="lg:sticky lg:top-8">
          {!result ? (
            <div className="rounded-2xl border-2 border-dashed border-fg/15 p-10 text-center text-fg/55">
              Your result appears here. Nothing you enter leaves this device except to be scored.
            </div>
          ) : (
            <ResultCard result={result} footer={
              <button onClick={() => generateReport({ result, inputs, schema, name: user.name })}
                      className="w-full border-t border-fg/10 bg-fg/[0.03] py-4 text-sm font-semibold text-accent transition hover:bg-fg/[0.06]">
                Download report for your doctor (PDF)
              </button>
            } />
          )}
        </aside>
      </div>

      {result && (
        <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
          <Panel title="What's driving it" sub="Compared with a typical person in the study.">
            <Contributions items={result.factors} />
          </Panel>
          <Panel title="What could change it" sub="Your modifiable risk factors.">
            <WhatIf model={model} result={result} inputs={inputs} schema={schema} />
          </Panel>
        </div>
      )}
    </div>
  );
}
