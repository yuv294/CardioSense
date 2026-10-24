import { GLOSSARY } from "../../lib/ui.js";

export default function Learn() {
  return (
    <div className="max-w-3xl">
      <h1 className="font-display text-4xl font-extrabold tracking-tight">What the numbers mean</h1>
      <dl className="mt-8 divide-y divide-fg/10 border-y border-fg/10">
        {GLOSSARY.map((g) => (
          <div key={g.term} className="grid gap-1 py-6 sm:grid-cols-[14rem_1fr] sm:gap-8">
            <dt>
              <div className="font-display text-lg font-bold leading-tight tracking-tight">{g.term}</div>
              <div className="mt-0.5 text-sm text-accent">{g.short}</div>
            </dt>
            <dd className="text-[15px] leading-relaxed text-fg/80">{g.body}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
