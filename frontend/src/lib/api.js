// Tiny API client. In dev, Vite proxies /api to the Flask backend, so the
// base is empty. In production, set VITE_API_BASE to the deployed backend URL
// at build time (or serve the built frontend from Flask itself: same origin).
const BASE = import.meta.env.VITE_API_BASE || "";
const cache = new Map();

async function get(path, { cached = false } = {}) {
  if (cached && cache.has(path)) return cache.get(path);
  const p = fetch(`${BASE}${path}`).then(async (r) => {
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `Request failed (${r.status})`);
    return r.json();
  });
  if (cached) {
    cache.set(path, p);
    p.catch(() => cache.delete(path));
  }
  return p;
}

async function post(path, body) {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.error || `Request failed (${r.status})`);
  }
  return r.json();
}

export const api = {
  models: () => get("/api/models", { cached: true }),
  schema: (model) => get(`/api/schema?model=${model}`, { cached: true }),
  evidence: (model) => get(`/api/evidence?model=${model}`, { cached: true }),
  predict: (model, payload, mode = "patient") => post(`/api/predict?model=${model}&mode=${mode}`, payload),
  batch: (model, records) => post(`/api/batch?model=${model}`, { records }),
  health: () => get("/api/health"),
};
