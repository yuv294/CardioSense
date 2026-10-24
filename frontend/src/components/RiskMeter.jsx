// A horizontal 0-40% scale with the 10% and 20% band edges, and a marker
// where this person sits. Shows the answer in context instead of as a bare number.
const MAX = 0.4;

export default function RiskMeter({ risk, threshold }) {
  const at = (p) => `${Math.min(p, MAX) / MAX * 100}%`;
  return (
    <div className="pt-6 pb-5">
      <div className="relative h-2.5 rounded-full overflow-visible flex">
        <div className="h-full bg-low/70 rounded-l-full" style={{ width: "25%" }} />
        <div className="h-full bg-mid/70" style={{ width: "25%" }} />
        <div className="h-full bg-high/70 rounded-r-full" style={{ width: "50%" }} />
        {threshold != null && (
          <div className="absolute -top-1 -bottom-1 w-px bg-fg/50" style={{ left: at(threshold) }} />
        )}
        <div className="absolute -top-5 -translate-x-1/2 flex flex-col items-center transition-[left] duration-700 ease-out"
             style={{ left: at(risk) }}>
          <span className="h-2 w-2 rotate-45 bg-fg" />
        </div>
        <div className="absolute top-0 h-full w-1 -translate-x-1/2 rounded bg-fg ring-2 ring-surface transition-[left] duration-700 ease-out"
             style={{ left: at(risk) }} />
      </div>
      <div className="relative mt-2 h-4 text-xs text-fg/55 tabular-nums">
        <span className="absolute left-0">0%</span>
        <span className="absolute -translate-x-1/2" style={{ left: "25%" }}>10%</span>
        <span className="absolute -translate-x-1/2" style={{ left: "50%" }}>20%</span>
        <span className="absolute right-0">40%+</span>
      </div>
    </div>
  );
}
