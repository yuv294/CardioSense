import { Link } from "react-router-dom";
import Ecg, { Wordmark } from "../components/Ecg.jsx";

const ROLES = [
  { to: "/login/patient", title: "I'm a patient", line: "Check your own ten-year heart risk and see what is driving it." },
  { to: "/login/clinician", title: "I'm a clinician", line: "Assess patients with every model's estimate and the full breakdown." },
];

export default function Gate() {
  return (
    <div data-role="clinician" className="relative min-h-screen overflow-hidden">
      <Ecg className="absolute inset-x-0 top-1/2 h-72 w-full -translate-y-1/2 opacity-90" />
      <div className="absolute inset-0 bg-gradient-to-r from-canvas via-canvas/70 to-transparent" />

      <div className="relative mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-8">
        <Wordmark className="text-xl" />
        <div className="grid flex-1 items-center gap-12 py-12 lg:grid-cols-[1.2fr_1fr]">
          <h1 className="font-display text-5xl font-extrabold leading-[1.02] tracking-tight sm:text-7xl">
            Ten-year heart risk, explained.
          </h1>
          <div className="space-y-3">
            {ROLES.map((r) => (
              <Link key={r.to} to={r.to}
                className="group block rounded-2xl border border-fg/15 bg-surface/80 p-6 backdrop-blur transition hover:border-accent hover:bg-surface">
                <div className="flex items-center justify-between">
                  <span className="font-display text-2xl font-bold tracking-tight">{r.title}</span>
                  <span className="text-accent transition group-hover:translate-x-1">›</span>
                </div>
                <p className="mt-2 max-w-sm text-[15px] text-fg/70">{r.line}</p>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
