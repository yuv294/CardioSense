import { useEffect, useState } from "react";
import { api } from "../lib/api.js";

const inputBase =
  "w-full rounded-lg border border-fg/15 bg-surface px-3 py-2.5 text-[15px] text-fg placeholder:text-fg/35 " +
  "focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30";

function Field({ f, value, onChange }) {
  const id = `f-${f.name}`;
  const opts = f.options ? Object.entries(f.options) : null;
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 flex items-baseline justify-between gap-2 text-sm font-medium text-fg/90">
        <span>{f.label}</span>
        {f.optional && <span className="text-xs font-normal text-fg/45">optional</span>}
      </label>
      {f.kind === "binary" ? (
        <div role="radiogroup" aria-label={f.label} className="grid grid-cols-2 rounded-lg border border-fg/15 bg-fg/5 p-0.5">
          {opts.map(([k, label]) => {
            const on = value !== "" && value != null && String(value) === k;
            return (
              <button key={k} type="button" role="radio" aria-checked={on}
                onClick={() => onChange(f.name, on && f.optional ? "" : Number(k))}
                className={`rounded-md px-3 py-2 text-sm transition-colors ${
                  on ? "bg-accent font-semibold text-accent-fg" : "text-fg/70 hover:text-fg"}`}>
                {label}
              </button>
            );
          })}
        </div>
      ) : f.kind === "select" ? (
        <select id={id} value={value ?? ""} className={inputBase}
                onChange={(e) => onChange(f.name, e.target.value === "" ? "" : Number(e.target.value))}>
          {f.optional && <option value="">Prefer not to say / unknown</option>}
          {opts.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
        </select>
      ) : (
        <div className="relative">
          <input id={id} type="number" inputMode="decimal" value={value ?? ""} min={f.min} max={f.max}
            step={f.step || "any"} placeholder={f.optional ? "Leave blank if unknown" : ""}
            onChange={(e) => onChange(f.name, e.target.value === "" ? "" : Number(e.target.value))}
            className={`${inputBase} ${f.unit ? "pr-16" : ""} tabular-nums`} />
          {f.unit && (
            <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs text-fg/50">{f.unit}</span>
          )}
        </div>
      )}
      {f.help && <p className="mt-1 text-xs text-fg/50">{f.help}</p>}
    </div>
  );
}

const EXAMPLES = {
  framingham: { male: 1, age: 55, sysBP: 142, diaBP: 88, BPMeds: 0, prevalentHyp: 1, currentSmoker: 1, cigsPerDay: 15,
                totChol: 245, HDL: 42, BMI: 28.5, diabetes: 0 },
  nhanes: { male: 1, age: 58, race: 3, education: 4, pir: 2.5, smoking: 2, cigsPerDay: 15, BPMeds: 1, prevalentHyp: 1,
            diabetes: 0, famHistory: 0, sysBP: 142, diaBP: 84, totChol: 225, HDL: 42, BMI: 30.1, waist: 104,
            HbA1c: 5.9, eGFR: 78, uacr: 22, crp: 3.1 },
};

export default function RiskForm({ model, mode, onResult, initial, submitLabel = "Calculate risk", wide = false, header = null }) {
  const [schema, setSchema] = useState(null);
  const [form, setForm] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSchema(null); setError(null);
    api.schema(model).then((s) => {
      setSchema(s);
      setForm(initial?.model === model ? initial.inputs : { ...EXAMPLES[model] });
    }).catch((e) => setError(e.message));
  }, [model]); // eslint-disable-line react-hooks/exhaustive-deps

  const onChange = (name, value) => setForm((f) => {
    const next = { ...f, [name]: value };
    // keep smoking answers consistent
    if ((name === "currentSmoker" && value === 0) || (name === "smoking" && value !== 2)) next.cigsPerDay = 0;
    return next;
  });

  async function submit(e) {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      onResult(await api.predict(model, form, mode), form);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (error && !schema) {
    return (
      <div className="rounded-xl border border-high/40 bg-high/10 p-5 text-sm">
        The prediction service isn't responding ({error}). Start the backend (<code>python backend/app.py</code>) and reload.
      </div>
    );
  }
  if (!schema) return <div className="h-96 animate-pulse rounded-2xl bg-fg/5" />;

  const smokerNow = form.currentSmoker === 1 || form.smoking === 2;
  return (
    <form onSubmit={submit} className="space-y-8">
      {header}
      {schema.groups.map((g) => {
        const fields = schema.features.filter((f) => f.group === g.key)
          .filter((f) => f.name !== "cigsPerDay" || smokerNow);
        return (
          <fieldset key={g.key}>
            <legend className="mb-1 font-display text-lg font-bold tracking-tight">{g.title}</legend>
            {g.note ? <p className="mb-3 text-sm text-fg/55">{g.note}</p> : <div className="mb-3" />}
            <div className={`grid gap-x-4 gap-y-4 sm:grid-cols-2 ${wide ? "xl:grid-cols-3" : ""}`}>
              {fields.map((f) => <Field key={f.name} f={f} value={form[f.name]} onChange={onChange} />)}
            </div>
          </fieldset>
        );
      })}
      <div className="flex flex-wrap items-center gap-4">
        <button type="submit" disabled={loading}
          className="rounded-lg bg-accent px-6 py-3 font-semibold text-accent-fg shadow-sm transition hover:brightness-110 disabled:opacity-60">
          {loading ? "Calculating…" : submitLabel}
        </button>
        <button type="button" onClick={() => setForm({ ...EXAMPLES[model] })}
                className="text-sm font-medium text-fg/60 hover:text-fg hover:underline">Load example</button>
        {error && <p role="alert" className="text-sm text-high">{error}</p>}
      </div>
    </form>
  );
}
