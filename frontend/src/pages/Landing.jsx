import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api.js";
import { MODEL_NAMES } from "../lib/ui.js";
import Ecg, { Wordmark } from "../components/Ecg.jsx";

const ROLES = [
  { to: "/login/patient", title: "I'm a patient", line: "Check your ten-year risk, see what drives it and what would change it." },
  { to: "/login/clinician", title: "I'm a clinician", line: "Every model side by side, published scores, batch scoring and patient records." },
];

function Result({ m }) {
  const n = m.nested;
  const sig = (n || m).d_auroc_ci[0] > 0;
  return (
    <div className="rounded-2xl border border-fg/15 bg-surface/80 p-6 backdrop-blur">
      <div className="text-xs font-semibold uppercase tracking-wide text-accent">{m.outcome_short}</div>
      <div className="mt-1 text-sm text-fg/60">{m.title}</div>
      <div className="mt-4 flex items-end gap-6">
        <div>
          <div className="font-display text-4xl font-extrabold tabular-nums">{m.auroc.toFixed(3)}</div>
          <div className="text-xs text-fg/55">our model, AUROC</div>
        </div>
        <div>
          <div className="font-display text-4xl font-extrabold tabular-nums text-fg/45">{m.baseline_auroc.toFixed(3)}</div>
          <div className="text-xs text-fg/55">{MODEL_NAMES[m.baseline]}</div>
        </div>
      </div>
      <p className="mt-4 text-sm leading-relaxed text-fg/70">
        <b className={sig ? "text-low" : "text-fg"}>{sig ? "Better ranking of risk." : "Ranking: a tie."}</b>{" "}
        Locked test set ({m.events_test} events): {m.d_auroc >= 0 ? "+" : ""}{m.d_auroc.toFixed(3)} AUROC
        (95% CI {m.d_auroc_ci[0].toFixed(3)} to {m.d_auroc_ci[1].toFixed(3)}).
        {n && <> Nested cross-validation ({n.events.toLocaleString()} events): {n.d_auroc >= 0 ? "+" : ""}{n.d_auroc.toFixed(3)}
          (CI {n.d_auroc_ci[0].toFixed(3)} to {n.d_auroc_ci[1].toFixed(3)}).</>}
        {n && n.d_brier_ci[1] < 0 && " Probabilities are better calibrated than the published score."}
      </p>
    </div>
  );
}

export default function Landing() {
  const [models, setModels] = useState(null);
  useEffect(() => { api.models().then(setModels).catch(() => setModels([])); }, []);

  return (
    <div data-role="clinician" className="relative min-h-screen overflow-hidden">
      <Ecg className="pointer-events-none absolute inset-x-0 top-[30rem] h-48 w-full opacity-25" />
      <div className="relative mx-auto max-w-6xl px-6 py-8">
        <header className="flex items-center justify-between">
          <Wordmark className="text-xl" />
          <nav className="flex gap-5 text-sm text-fg/70">
            <Link to="/evidence" className="hover:text-fg">Evidence</Link>
            <Link to="/about" className="hover:text-fg">Method</Link>
          </nav>
        </header>

        <section className="grid items-center gap-12 py-16 lg:grid-cols-[1.25fr_1fr]">
          <div>
            <h1 className="font-display text-5xl font-extrabold leading-[1.02] tracking-tight sm:text-7xl">
              Ten-year heart risk, <span className="text-accent">tested against the clinical scores.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-fg/70">
              Four machine-learning models trained on two cohorts and judged on people they never saw, against the
              Framingham Risk Score and the ACC/AHA Pooled Cohort Equations.
            </p>
          </div>
          <div className="space-y-3">
            {ROLES.map((r) => (
              <Link key={r.to} to={r.to}
                className="group block rounded-2xl border border-fg/15 bg-surface/80 p-6 backdrop-blur transition hover:border-accent hover:bg-surface">
                <div className="flex items-center justify-between">
                  <span className="font-display text-2xl font-bold tracking-tight">{r.title}</span>
                  <span className="text-xl text-accent transition group-hover:translate-x-1">→</span>
                </div>
                <p className="mt-2 max-w-sm text-[15px] text-fg/70">{r.line}</p>
              </Link>
            ))}
          </div>
        </section>

        <section className="pb-16">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-fg/50">Results on held-out people</h2>
          {!models ? <div className="h-48 animate-pulse rounded-2xl bg-fg/5" /> : models.length === 0 ? (
            <p className="text-fg/60">Start the backend to see the results.</p>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {[...models].sort((a) => (a.name === "nhanes" ? -1 : 1)).map((m) => <Result key={m.name} m={m} />)}
            </div>
          )}
          <p className="mt-6 max-w-3xl text-sm leading-relaxed text-fg/55">
            Educational project. The models are trained on research cohorts, have not been clinically validated and must not be
            used to diagnose or treat anyone. <Link to="/evidence" className="text-accent hover:underline">See the full evidence →</Link>
          </p>
        </section>
      </div>
    </div>
  );
}
