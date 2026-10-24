// Accounts live in this browser (localStorage). There is no server-side user
// database, so a sign-in only separates the two portals and keeps each
// person's saved assessments apart on this device.
const USERS = "cs_users_v1";
const SESSION = "cs_session_v1";

function read(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function write(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked */ }
}

async function digest(text) {
  if (globalThis.crypto?.subtle) {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  return btoa(unescape(encodeURIComponent(text)));
}

const keyOf = (role, email) => `${role}:${email.trim().toLowerCase()}`;

export async function register({ name, email, password, role }) {
  const users = read(USERS, {});
  const k = keyOf(role, email);
  if (users[k]) throw new Error("An account with this email already exists. Sign in instead.");
  const salt = Math.random().toString(36).slice(2);
  users[k] = { name: name.trim(), email: email.trim().toLowerCase(), role, salt, hash: await digest(salt + password) };
  write(USERS, users);
  return startSession(users[k]);
}

export async function login({ email, password, role }) {
  const u = read(USERS, {})[keyOf(role, email)];
  if (!u || u.hash !== (await digest(u.salt + password))) {
    throw new Error("Email or password is incorrect.");
  }
  return startSession(u);
}

function startSession(u) {
  const session = { name: u.name, email: u.email, role: u.role };
  write(SESSION, session);
  return session;
}

export function currentSession() { return read(SESSION, null); }
export function endSession() { try { localStorage.removeItem(SESSION); } catch { /* ignore */ } }

// One-click demo accounts so the app can be shown without typing credentials.
export const DEMO = {
  patient:   { name: "Demo Patient",   email: "patient@cardiosense.demo",   password: "demo1234", role: "patient" },
  clinician: { name: "Dr. Demo",       email: "clinician@cardiosense.demo", password: "demo1234", role: "clinician" },
};
export async function demoLogin(role) {
  const d = DEMO[role];
  try { return await login(d); } catch { return register(d); }
}
