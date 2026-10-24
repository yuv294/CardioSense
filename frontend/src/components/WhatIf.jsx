import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api.js";

// "What could change it": preset targets from the API, plus sliders that
// re-score the person live. Model-based associations, not proof of cause.
const SLIDERS = {
  sysBP: { min: 90, max: 200, step: 1 }, totChol: { min: 120, max: 350, step: 5 }, HDL: { min: 20, max: 100, step: 1 },
  BMI: { min: 17, max: 45, step: 0.5 }, HbA1c: { min: 4.5, max: 11, step: 0.1 }, waist: { min: 60, max: 150, step: 1 },
};

export default function WhatIf({ model, result, inputs, schema }) {
  const [vals, setVals] = useState(inputs);
  const [live, setLive] = useState(null);
  const timer = useRef(null);

  useEffect(() => { setVals(inputs); setLive(null); }, [inputs]);

  function change(name, value) {
    const next = { ...vals, [name]: value };
    if (name === "currentSmoker" && value === 0) next.cigsPerDay = 0;
    if (name === "smoking" && value !== 2) next.cigsPerDay = 0;
    setVals(next);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      api.predict(model, next).then((r) => setLive(r.risk_percent)).catch(() => {});
    }, 250);
  }

  const meta = Object.fromEntries((schema?.features || []).map((f) => [f.name, f]));
  const sliders = Object.keys(SLIDERS).filter((k) => meta[k] && vals[k] !== "" && vals[k] != null);
  const now = result.risk_percent;
  const peak = Math.max(now, ...result.scenarios.map((s) => s.risk_percent), 0.1);
  const smokerKey = "currentSmoker" in meta ? "currentSmoker" : "smoking";
  const isSmoker = smokerKey === "currentSmoker" ? vals.currentSmoker === 1 : vals.smoking === 2;
  const wasSmoker = smokerKey === "currentSmoker" ? inputs.currentSmoker === 1 : inputs.smoking === 2;

  return (
    <div className="grid gap-8 2xl:grid-cols-2">
      <div>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-fg/50">If you reached these targets</h3>
        {result.scenarios.length === 0 ? (
          <p className="text-sm text-fg/65">Your modifiable numbers are already at the usual targets. Keep it up.</p>
        ) : (
          <div className="space-y-2.5">
            {[{ label: "Today", risk_percent: now, reduction_percent: 0 }, ...result.scenarios].map((s) => (
              <div key={s.label} className="grid grid-cols-[minmax(7rem,12rem)_1fr_3.5rem] items-center gap-3 text-sm">
                <span className={`leading-tight ${s.label === "Today" ? "font-semibold" : "text-fg/80"}`}>{s.label}</span>
                <div className="h-3 rounded-full bg-fg/10">
                  <div className={`h-full rounded-full ${s.label === "Today" ? "bg-fg/70" : "bg-accent"}`}
                       style={{ width: `${(s.risk_percent / peak) * 100}%` }} />
                </div>
                <span className="text-right tabular-nums">{s.risk_percent}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        <div className="mb-3 flex items-baseline justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-fg/50">Try it yourself</h3>
          <span className="text-sm">
            {live == null ? <span className="text-fg/50">move a slider</span> : (
              <><span className="text-fg/55">{now}% → </span>
                <b className={`font-display text-xl tabular-nums ${live < now ? "text-low" : live > now ? "text-high" : ""}`}>{live}%</b></>
            )}
          </span>
        </div>
        <div className="space-y-4">
          {wasSmoker && (
            <label className="flex items-center gap-3 text-sm">
              <input type="checkbox" checked={!isSmoker} className="h-4 w-4 accent-[rgb(var(--accent))]"
                     onChange={(e) => change(smokerKey, e.target.checked ? (smokerKey === "smoking" ? 1 : 0) : (smokerKey === "smoking" ? 2 : 1))} />
              Stop smoking
            </label>
          )}
          {sliders.map((k) => (
            <div key={k}>
              <div className="flex justify-between text-sm">
                <label htmlFor={`w-${k}`} className="text-fg/80">{meta[k].label}</label>
                <span className="tabular-nums text-fg/70">{vals[k]} {meta[k].unit}</span>
              </div>
              <input id={`w-${k}`} type="range" {...SLIDERS[k]} value={vals[k]} className="w-full accent-[rgb(var(--accent))]"
                     onChange={(e) => change(k, Number(e.target.value))} />
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs leading-relaxed text-fg/50">
          These are what the model predicts for someone with the changed numbers. They show how risk factors are
          associated with outcomes in the cohort, not a guaranteed effect of treatment.
        </p>
      </div>
    </div>
  );
}
