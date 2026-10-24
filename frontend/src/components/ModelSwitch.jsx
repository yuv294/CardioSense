import { CALCULATORS } from "../lib/ui.js";

// Segmented control between the two calculators.
export default function ModelSwitch({ value, onChange }) {
  return (
    <div role="tablist" aria-label="Calculator" className="grid gap-2 sm:grid-cols-2">
      {Object.entries(CALCULATORS).map(([key, c]) => {
        const on = key === value;
        return (
          <button key={key} role="tab" aria-selected={on} onClick={() => onChange(key)}
            className={`rounded-xl border p-4 text-left transition ${
              on ? "border-accent bg-accent/10 ring-1 ring-accent" : "border-fg/15 bg-surface hover:border-fg/30"}`}>
            <div className="flex items-center justify-between">
              <span className="font-display text-lg font-bold tracking-tight">{c.short}</span>
              <span className="text-xs text-fg/50">{c.cohort}</span>
            </div>
            <p className="mt-1 text-sm leading-snug text-fg/65">{c.blurb}</p>
          </button>
        );
      })}
    </div>
  );
}
