// Signed bars: how far each factor moves this person's risk away from the
// typical person's (exact Shapley values on the log-odds scale).
export default function Contributions({ items, max = 8 }) {
  const shown = items.filter((d) => Math.abs(d.contribution) >= 0.005).slice(0, max);
  const peak = Math.max(...shown.map((d) => Math.abs(d.contribution)), 0.001);
  if (!shown.length) return <p className="text-sm text-fg/60">Every factor is close to the typical person's.</p>;
  return (
    <div className="space-y-2.5">
      {shown.map((d) => {
        const up = d.contribution > 0;
        const w = (Math.abs(d.contribution) / peak) * 50;
        // log-odds -> "times the odds", easier to read than raw log-odds
        const times = Math.exp(Math.abs(d.contribution));
        return (
          <div key={d.factor} className="flex items-center gap-3 text-sm">
            <span className="w-32 shrink-0 truncate text-fg/85 sm:w-40">{d.factor}</span>
            <div className="relative h-4 flex-1">
              <div className="absolute inset-y-0 left-1/2 w-px bg-fg/20" />
              <div className={`absolute inset-y-0.5 rounded-sm ${up ? "bg-high" : "bg-low"}`}
                   style={up ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
            </div>
            <span className={`w-14 text-right text-xs tabular-nums ${up ? "text-high" : "text-low"}`}>
              {up ? "×" : "÷"}{times.toFixed(times < 1.95 ? 2 : 1)}
            </span>
          </div>
        );
      })}
      <div className="flex justify-between pt-1 text-xs text-fg/50">
        <span>Lowers your risk</span>
        <span>Raises your risk</span>
      </div>
    </div>
  );
}
