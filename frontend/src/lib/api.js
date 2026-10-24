// Tiny API client. In dev, Vite proxies /api to the Flask backend, so the
// base is empty. In production, set VITE_API_BASE to your deployed backend URL
// (e.g. https://cardiosense-api.onrender.com) at build time.
const BASE = import.meta.env.VITE_API_BASE || "";

async function get(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`Request failed (${r.status})`);
  return r.json();
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
  schema: () => get("/api/schema"),
  metrics: () => get("/api/metrics"),
  predict: (payload, mode = "patient") => post(`/api/predict?mode=${mode}`, payload),
  health: () => get("/api/health"),
};
