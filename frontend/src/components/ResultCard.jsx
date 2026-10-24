import { BAND } from "../lib/ui.js";
import RiskMeter from "./RiskMeter.jsx";

// The headline: the number, its band, the scale, and the honest context
// (typical person, people like you, and the published clinical score).
export default function ResultCard({ result, compact = false, footer = null, title = null }) {
  const b = BAND[result.band];
  const peers = result.peers;
  return (
    <section className="overflow-hidden rounded-2xl border border-fg/10 bg-surface shadow-[0_12px_32px_-18px_rgb(var(--fg)/0.25)]">
      <div className="p-6 sm:p-7">
        {title && <div className="mb-1 text-sm text-fg/60">{title}</div>}
        <div className="text-xs font-semibold uppercase tracking-wide text-fg/50">{result.outcome}</div>
        <div className="mt-1 flex items-end gap-4">
          <div className={`font-display font-extrabold leading-none tracking-tight tabular-nums ${b.text} ${compact ? "text-6xl" : "text-7xl sm:text-8xl"}`}>
            {result.risk_percent}<span className="text-[0.5em]">%</span>
          </div>
          <div className={`pb-2 font-display text-2xl font-bold ${b.text}`}>{result.band_label}</div>
        </div>
        <p className="mt-3 text-[15px] leading-relaxed text-fg/75">{result.band_text}</p>
        <RiskMeter risk={result.risk} bands={result.bands} typical={result.typical_risk_percent / 100} />

        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 border-t border-fg/10 pt-4 text-sm">
          <div>
            <dt className="text-fg/55">Typical person in the cohort</dt>
            <dd className="font-semibold tabular-nums">{result.typical_risk_percent}%</dd>
          </div>
          {peers && (
            <div>
              <dt className="text-fg/55">Your age and sex, on average</dt>
              <dd className="font-semibold tabular-nums">{(peers.mean_risk * 100).toFixed(1)}%</dd>
            </div>
          )}
          {result.published_scores.map((s) => (
            <div key={s.name}>
              <dt className="text-fg/55">{s.label}</dt>
              <dd className="font-semibold tabular-nums">
                {s.risk_percent}%
                <span className="ml-1.5 text-xs font-normal text-fg/50" title="After recalibration to this outcome and cohort">
                  ({s.recalibrated_percent}% recalibrated)
                </span>
              </dd>
            </div>
          ))}
        </dl>
        {result.published_scores.length > 0 && (
          <p className="mt-3 text-xs leading-relaxed text-fg/50">
            Published scores predict a broader outcome (fatal and non-fatal heart attacks and strokes) in other cohorts, so
            their raw figure can be much higher. The recalibrated figure puts them on this calculator's scale.
          </p>
        )}

        {result.unknown_fields.length > 0 && (
          <p className="mt-4 rounded-lg bg-mid/10 px-3.5 py-2.5 text-sm text-fg/80">
            Not provided: {result.unknown_fields.join(", ")}. The model handles these as unknown; adding them can sharpen the estimate.
          </p>
        )}
      </div>
      {footer}
    </section>
  );
}
