// Signed bars: how far each factor pushed this person's risk up or down.
export default function Contributions({ items, max = 8 }) {
  const shown = items.slice(0, max);
  const peak = Math.max(...shown.map((d) => Math.abs(d.contribution ?? d.weight)), 0.001);

  return (
    <div className="space-y-2">
      {shown.map((d) => {
        const val = d.contribution ?? (d.direction === "increases" ? d.weight : -d.weight);
        const up = val > 0;
        const w = (Math.abs(val) / peak) * 50;
        return (
          <div key={d.feature} className="flex items-center gap-3 text-sm">
            <span className="w-40 shrink-0 truncate text-fg/85">{d.label}</span>
            <div className="relative flex-1 h-4">
              <div className="absolute left-1/2 inset-y-0 w-px bg-fg/20" />
              <div className={`absolute inset-y-0.5 rounded-sm ${up ? "bg-high" : "bg-low"}`}
                   style={up ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
            </div>
            <span className={`w-12 text-right text-xs tabular-nums ${up ? "text-high" : "text-low"}`}>
              {up ? "+" : "−"}{Math.abs(val).toFixed(2)}
            </span>
          </div>
        );
      })}
      <div className="flex justify-between pt-1 text-xs text-fg/50">
        <span>Lowers risk</span>
        <span>Raises risk</span>
      </div>
    </div>
  );
}
