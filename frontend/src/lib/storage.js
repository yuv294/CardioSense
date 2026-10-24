// Saved assessments, kept per signed-in account in this browser only.
const keyFor = (u) => `cs_records_v2:${u.role}:${u.email}`;

export function loadRecords(user) {
  try { return JSON.parse(localStorage.getItem(keyFor(user)) || "[]"); } catch { return []; }
}

function persist(user, list) {
  try { localStorage.setItem(keyFor(user), JSON.stringify(list.slice(0, 200))); } catch { /* ignore */ }
}

export function saveRecord(user, entry) {
  const list = loadRecords(user);
  list.unshift({ id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), ...entry });
  persist(user, list);
}

export function removeRecord(user, id) {
  const list = loadRecords(user).filter((r) => r.id !== id);
  persist(user, list);
  return list;
}

export function clearRecords(user) {
  try { localStorage.removeItem(keyFor(user)); } catch { /* ignore */ }
}
