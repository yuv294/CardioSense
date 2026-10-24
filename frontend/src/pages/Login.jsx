import { useState } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { login, register, demoLogin } from "../lib/auth.js";
import { useAuth } from "../lib/useAuth.jsx";
import Ecg, { Wordmark } from "../components/Ecg.jsx";

const COPY = {
  patient:   { title: "Patient sign in",   alt: "Create a patient account" },
  clinician: { title: "Clinician sign in", alt: "Create a clinician account" },
};

const input =
  "w-full rounded-lg border border-fg/15 bg-surface px-3.5 py-3 text-[15px] placeholder:text-fg/35 " +
  "focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30";

export default function Login() {
  const { role } = useParams();
  const { user, setUser } = useAuth();
  const nav = useNavigate();
  const [creating, setCreating] = useState(false);
  const [f, setF] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!COPY[role]) return <Navigate to="/" replace />;
  if (user && user.role === role) return <Navigate to={`/${role}`} replace />;

  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  async function enter(action) {
    setBusy(true); setError(null);
    try {
      const s = await action();
      setUser(s);
      nav(`/${role}`, { replace: true });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const submit = (e) => {
    e.preventDefault();
    enter(() => creating
      ? register({ ...f, role })
      : login({ email: f.email, password: f.password, role }));
  };

  return (
    <div data-role={role} className="grid min-h-screen lg:grid-cols-2">
      <div className="relative hidden overflow-hidden lg:block">
        <div className="absolute inset-0 bg-fg/[0.03]" />
        <Ecg className="absolute inset-x-0 top-1/2 h-56 w-full -translate-y-1/2" />
        <Link to="/" className="absolute left-10 top-8"><Wordmark className="text-xl" /></Link>
      </div>

      <div className="flex items-center px-6 py-10 sm:px-14">
        <div className="mx-auto w-full max-w-sm">
          <Link to="/" className="mb-10 block lg:hidden"><Wordmark className="text-xl" /></Link>
          <h1 className="font-display text-4xl font-extrabold tracking-tight">
            {creating ? COPY[role].alt : COPY[role].title}
          </h1>

          <form onSubmit={submit} className="mt-8 space-y-4">
            {creating && (
              <div>
                <label htmlFor="name" className="mb-1.5 block text-sm font-medium">Full name</label>
                <input id="name" required value={f.name} onChange={set("name")} autoComplete="name" className={input} />
              </div>
            )}
            <div>
              <label htmlFor="email" className="mb-1.5 block text-sm font-medium">Email</label>
              <input id="email" type="email" required value={f.email} onChange={set("email")} autoComplete="email" className={input} />
            </div>
            <div>
              <label htmlFor="pw" className="mb-1.5 block text-sm font-medium">Password</label>
              <input id="pw" type="password" required minLength={6} value={f.password} onChange={set("password")}
                     autoComplete={creating ? "new-password" : "current-password"} className={input} />
            </div>
            {error && <p role="alert" className="text-sm text-high">{error}</p>}
            <button disabled={busy}
              className="w-full rounded-lg bg-accent py-3 font-semibold text-accent-fg transition hover:brightness-110 disabled:opacity-60">
              {creating ? "Create account" : "Sign in"}
            </button>
          </form>

          <div className="mt-6 flex flex-col gap-2 text-sm">
            <button type="button" onClick={() => { setCreating(!creating); setError(null); }}
                    className="text-left text-accent hover:underline">
              {creating ? "I already have an account" : COPY[role].alt}
            </button>
            <button type="button" disabled={busy} onClick={() => enter(() => demoLogin(role))}
                    className="text-left text-fg/60 hover:text-fg hover:underline">
              Continue with a demo account
            </button>
            <Link to="/" className="text-fg/60 hover:text-fg hover:underline">Choose a different role</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
