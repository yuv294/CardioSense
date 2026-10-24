import { useEffect, useState } from "react";
import { api } from "../../lib/api.js";
import { saveRecord } from "../../lib/storage.js";
import { useAuth } from "../../lib/useAuth.jsx";
import RiskForm from "../../components/RiskForm.jsx";
import ResultCard from "../../components/ResultCard.jsx";
import Contributions from "../../components/Contributions.jsx";
import WhatIf from "../../components/WhatIf.jsx";
import ModelSwitch from "../../components/ModelSwitch.jsx";
import { Panel } from "../patient/Assess.jsx";

function agreement(probs) {
  const v = probs.filter((p) => p.model !== "refit").map((p) => p.risk_percent);
  const mid = (Math.max(...v) + Math.min(...v)) / 2 || 1;
  const rel = (Math.max(...v) - Math.min(...v)) / mid;
  return rel < 0.35 ? ["High", "text-low"] : rel < 0.7 ? ["Moderate", "text-mid"] : ["Low", "text-high"];
}

export default function ClinicianAssess() {
  const { user } = useAuth();
  const [model, setModel] = useState("nhanes");
  const [patient, setPatient] = useState("");
  const [result, setResult] = useState(null);
  const [inputs, setInputs] = useState(null);
  const [schema, setSchema] = useState(null);
  const [shown, setShown] = useState("");

  useEffect(() => { setResult(null); api.schema(model).then(setSchema).catch(() => {}); }, [model]);

  function onResult(res, form) {
    const label = patient.trim() || "Unnamed patient";
    setResult(res); setInputs(form); setShown(label);
    saveRecord(user, { at: new Date().toISOString(), patient: label, model, inputs: form,
                       result: { risk_percent: res.risk_percent, band: res.band, band_label: res.band_label,
                                 outcome: res.outcome, published: res.published_scores } });
  }

  const [agree, agreeCls] = result ? agreement(result.model_probabilities) : [];
  const peak = result ? Math.max(...result.model_probabilities.map((m) => m.risk_percent), 1) : 1;

  return (
    <div className="space-y-8">
      <header className="space-y-4">
        <h1 className="font-display text-3xl font-extrabold tracking-tight">Assess patient</h1>
        <ModelSwitch value={model} onChange={setModel} />
      </header>

      <div className="grid items-start gap-8 xl:grid-cols-[1fr_420px]">
        <RiskForm model={model} mode="clinician" wide submitLabel="Run assessment" onResult={onResult}
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
            <ResultCard result={result} compact title={shown} footer={
              <div className="grid grid-cols-2 border-t border-fg/10 text-sm">
                <div className="p-4">
                  <div className="text-fg/55">Model agreement</div>
                  <div className={`font-semibold ${agreeCls}`}>{agree}</div>
                </div>
                <div className="border-l border-fg/10 p-4">
                  <div className="text-fg/55">Cohort percentile</div>
                  <div className="font-semibold">{result.risk_percentile}th</div>
                </div>
              </div>
            } />
          )}
        </aside>
      </div>

      {result && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Panel title="Every model's estimate" sub="All calibrated on the training cohort. The served model is highlighted.">
            <div className="space-y-3">
              {result.model_probabilities.map((m) => (
                <div key={m.model} className="flex items-center gap-3 text-sm">
                  <span className={`w-48 shrink-0 truncate ${m.served ? "font-semibold text-accent" : "text-fg/85"}`}>{m.label}</span>
                  <div className="relative h-3 flex-1 rounded-full bg-fg/10">
                    <div className={`absolute inset-y-0 left-0 rounded-full ${m.served ? "bg-accent" : "bg-fg/40"}`}
                         style={{ width: `${(m.risk_percent / peak) * 100}%` }} />
                  </div>
                  <span className="w-12 text-right tabular-nums">{m.risk_percent}%</span>
                </div>
              ))}
            </div>
          </Panel>
          <Panel title="Factor contributions" sub="Exact Shapley values against the cohort's typical patient (odds ratio scale).">
            <Contributions items={result.factors} max={12} />
          </Panel>
          <Panel title="Modifiable risk" className="lg:col-span-2">
            <WhatIf model={model} result={result} inputs={inputs} schema={schema} />
          </Panel>
        </div>
      )}
    </div>
  );
}
