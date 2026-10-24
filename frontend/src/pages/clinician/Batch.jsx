import { useEffect, useState } from "react";
import { api } from "../../lib/api.js";
import ModelSwitch from "../../components/ModelSwitch.jsx";

// Minimal CSV parser: comma-separated, optional double quotes, first row = header.
function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') q = false; else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); cell = "";
      if (row.some((c) => c.trim() !== "")) rows.push(row);
      row = [];
    } else cell += ch;
  }
  row.push(cell);
  if (row.some((c) => c.trim() !== "")) rows.push(row);
  const [head, ...body] = rows;
  return { head: head.map((h) => h.trim()), body };
}

const download = (name, text) => {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
};

export default function Batch() {
  const [model, setModel] = useState("framingham");
  const [schema, setSchema] = useState(null);
  const [file, setFile] = useState(null);
  const [out, setOut] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { setOut(null); setErr(null); api.schema(model).then(setSchema).catch(() => {}); }, [model]);

  const cols = schema ? ["id", ...schema.features.map((f) => f.name)] : [];

  function template() {
    const ex = schema.example;
    download(`cardiosense-${model}-template.csv`, [cols.join(","), ["patient-1", ...schema.features.map((f) => ex[f.name] ?? "")].join(",")].join("\n"));
  }

  async function run() {
    setBusy(true); setErr(null); setOut(null);
    try {
      const { head, body } = parseCsv(await file.text());
      const records = body.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] === undefined || r[i].trim() === "" ? null : r[i].trim()])));
      const { results } = await api.batch(model, records);
      setOut(results.map((res, i) => ({ id: records[i].id ?? `row ${i + 1}`, ...res })));
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  const resultCols = out ? [...new Set(out.flatMap((r) => Object.keys(r)))].filter((k) => !["row", "error"].includes(k)) : [];
  const exportResults = () => download(`cardiosense-${model}-results.csv`,
    [[...resultCols, "error"].join(","), ...out.map((r) => [...resultCols, "error"].map((k) => JSON.stringify(r[k] ?? "")).join(","))].join("\n"));

  return (
    <div className="space-y-6">
      <h1 className="font-display text-3xl font-extrabold tracking-tight">Batch scoring</h1>
      <ModelSwitch value={model} onChange={setModel} />
      <section className="rounded-2xl border border-fg/10 bg-surface p-6">
        <p className="text-sm leading-relaxed text-fg/70">
          Upload a CSV with one patient per row and these column names (optional columns may be blank):
        </p>
        <code className="mt-2 block overflow-x-auto rounded-lg bg-fg/5 p-3 text-xs text-fg/80">{cols.join(", ")}</code>
        <div className="mt-5 flex flex-wrap items-center gap-4">
          <button onClick={template} disabled={!schema}
                  className="rounded-lg border border-accent px-4 py-2 text-sm font-semibold text-accent hover:bg-accent/10">Download template</button>
          <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files[0] || null)}
                 className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-fg/10 file:px-4 file:py-2 file:text-fg" />
          <button onClick={run} disabled={!file || busy}
                  className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-accent-fg disabled:opacity-50">
            {busy ? "Scoring…" : "Score file"}
          </button>
        </div>
        {err && <p className="mt-4 text-sm text-high">{err}</p>}
      </section>

      {out && (
        <section className="rounded-2xl border border-fg/10 bg-surface">
          <div className="flex items-center justify-between border-b border-fg/10 px-5 py-3 text-sm">
            <span>{out.filter((r) => !r.error).length} scored · {out.filter((r) => r.error).length} with problems</span>
            <button onClick={exportResults} className="font-semibold text-accent hover:underline">Export results</button>
          </div>
          <div className="max-h-[32rem] overflow-auto">
            <table className="w-full text-left text-sm tabular-nums">
              <thead className="sticky top-0 bg-surface text-fg/55">
                <tr>{[...resultCols, "Problem"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-fg/10">
                {out.map((r, i) => (
                  <tr key={i}>
                    {resultCols.map((k) => <td key={k} className="px-4 py-2">{r[k] ?? ""}</td>)}
                    <td className="px-4 py-2 text-high">{r.error || ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
