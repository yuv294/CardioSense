// A horizontal scale with the calculator's band edges and a marker where this
// person sits. The scale runs to 2x the "high" edge so both edges are visible.
export default function RiskMeter({ risk, bands, typical }) {
  const edges = bands.slice(0, -1).map((b) => b.edge);
  const MAX = edges[edges.length - 1] * 2;
  const at = (p) => `${(Math.min(p, MAX) / MAX) * 100}%`;
  const colors = ["bg-low/70", "bg-mid/70", "bg-high/70"];
  const widths = [edges[0], edges[1] - edges[0], MAX - edges[1]].map((w) => `${(w / MAX) * 100}%`);
  return (
    <div className="pb-5 pt-7">
      <div className="relative flex h-2.5">
        {widths.map((w, i) => (
          <div key={i} className={`h-full ${colors[i]} ${i === 0 ? "rounded-l-full" : ""} ${i === 2 ? "rounded-r-full" : ""}`}
               style={{ width: w }} />
        ))}
        {typical != null && (
          <div className="absolute -bottom-1.5 -top-1.5 w-0.5 -translate-x-1/2 bg-fg/40" style={{ left: at(typical) }}
               title={`Typical person: ${(typical * 100).toFixed(1)}%`} />
        )}
        <div className="absolute -top-6 -translate-x-1/2 text-[11px] font-semibold tabular-nums transition-[left] duration-700"
             style={{ left: at(risk) }}>you</div>
        <div className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-surface bg-fg shadow transition-[left] duration-700 ease-out"
             style={{ left: at(risk) }} />
      </div>
      <div className="relative mt-2 h-4 text-xs tabular-nums text-fg/55">
        <span className="absolute left-0">0%</span>
        {edges.map((e) => (
          <span key={e} className="absolute -translate-x-1/2" style={{ left: at(e) }}>{+(e * 100).toFixed(1)}%</span>
        ))}
        <span className="absolute right-0">{+(MAX * 100).toFixed(0)}%+</span>
      </div>
    </div>
  );
}
