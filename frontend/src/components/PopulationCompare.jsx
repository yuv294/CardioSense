// Where this person sits against the 4,240-person cohort, overall and per measurement.
export default function PopulationCompare({ population, riskPercentile }) {
  return (
    <div className="space-y-5">
      <p className="text-[15px] leading-relaxed text-fg/85">
        Your estimated risk is higher than <b className="text-fg">{riskPercentile}%</b> of
        the people in the study. Across the whole cohort,{" "}
        {(population.cohort_event_rate * 100).toFixed(0)}% developed heart disease within ten years.
      </p>
      <div className="space-y-3">
        {population.your_features.map((f) => (
          <div key={f.feature}>
            <div className="flex justify-between text-sm mb-1.5">
              <span className="text-fg/70">{f.label} <b className="ml-1 font-semibold text-fg">{f.value}</b></span>
              <span className="tabular-nums text-fg/60">{f.percentile}th</span>
            </div>
            <div className="relative h-1.5 rounded-full bg-fg/10">
              <div className="absolute inset-y-0 left-0 rounded-full bg-accent/50" style={{ width: `${f.percentile}%` }} />
              <div className="absolute top-1/2 h-3.5 w-3.5 -translate-y-1/2 rounded-full bg-accent ring-2 ring-surface"
                   style={{ left: `calc(${f.percentile}% - 7px)` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
