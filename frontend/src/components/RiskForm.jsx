import { useEffect, useState } from "react";
import { api } from "../lib/api.js";

const SAMPLE = {
  male: 0, age: 45, education: 2, currentSmoker: 0, cigsPerDay: 0, BPMeds: 0,
  prevalentStroke: 0, prevalentHyp: 0, diabetes: 0, totChol: 200, sysBP: 120,
  diaBP: 80, BMI: 24, heartRate: 72, glucose: 85,
};

const GROUPS = [
  { title: "About the person", names: ["male", "age", "education"] },
  { title: "Smoking", names: ["currentSmoker", "cigsPerDay"] },
  { title: "Medical history", names: ["BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"] },
  { title: "Measurements", names: ["sysBP", "diaBP", "totChol", "glucose", "BMI", "heartRate"] },
];

const inputBase =
  "w-full rounded-lg border border-fg/15 bg-surface px-3 py-2.5 text-[15px] text-fg placeholder:text-fg/35 " +
  "focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30";

function Field({ f, value, onChange }) {
  const id = `f-${f.name}`;
  const opts = f.kind === "binary" ? Object.entries(f.options) : null;

  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-sm font-medium text-fg/90">{f.label}</label>
      {opts && opts.length === 2 ? (
        <div role="radiogroup" aria-label={f.label}
             className="grid grid-cols-2 rounded-lg border border-fg/15 bg-fg/5 p-0.5">
          {opts.map(([k, label]) => {
            const on = String(value) === k;
            return (
              <button key={k} type="button" role="radio" aria-checked={on}
                onClick={() => onChange(f.name, Number(k))}
                className={`rounded-md px-3 py-2 text-sm transition-colors ${
                  on ? "bg-accent font-semibold text-accent-fg" : "text-fg/70 hover:text-fg"}`}>
                {label}
              </button>
            );
          })}
        </div>
      ) : opts ? (
        <select id={id} value={value} onChange={(e) => onChange(f.name, Number(e.target.value))} className={inputBase}>
          {opts.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
        </select>
      ) : (
        <div className="relative">
          <input id={id} type="number" inputMode="decimal" value={value} min={f.min} max={f.max}
            placeholder={f.optional ? "Skip if unknown" : ""}
            onChange={(e) => onChange(f.name, e.target.value === "" ? "" : Number(e.target.value))}
            className={`${inputBase} ${f.unit ? "pr-16" : ""} tabular-nums`} />
          {f.unit && (
            <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs text-fg/50">
              {f.unit}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default function RiskForm({ mode, onResult, submitLabel = "Assess risk", wide = false, header = null }) {
  const [schema, setSchema] = useState(null);
  const [form, setForm] = useState(SAMPLE);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { api.schema().then(setSchema).catch((e) => setError(e.message)); }, []);

  const onChange = (name, value) => setForm((f) => ({ ...f, [name]: value }));

  async function submit(e) {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      onResult(await api.predict(form, mode), form);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (error && !schema) {
    return (
      <div className="rounded-xl border border-high/40 bg-high/10 p-5 text-sm">
        The prediction service isn't responding. Start the backend (python app.py) and reload this page.
      </div>
    );
  }
  if (!schema) return <p className="text-fg/50">Loading form…</p>;

  const byName = Object.fromEntries(schema.features.map((f) => [f.name, f]));

  return (
    <form onSubmit={submit} className="space-y-8">
      {header}
      {GROUPS.map((g) => (
        <fieldset key={g.title}>
          <legend className="mb-3 font-display text-lg font-bold tracking-tight">{g.title}</legend>
          <div className={`grid gap-x-4 gap-y-4 sm:grid-cols-2 ${wide ? "xl:grid-cols-3" : ""}`}>
            {g.names.filter((n) => byName[n]).map((n) => (
              <Field key={n} f={byName[n]} value={form[n] ?? ""} onChange={onChange} />
            ))}
          </div>
        </fieldset>
      ))}
      <div className="flex items-center gap-4">
        <button type="submit" disabled={loading}
          className="rounded-lg bg-accent px-6 py-3 font-semibold text-accent-fg transition hover:brightness-110 disabled:opacity-60">
          {loading ? "Working…" : submitLabel}
        </button>
        {error && <p className="text-sm text-high">{error}</p>}
      </div>
    </form>
  );
}
