// An ECG trace drawn as one SVG path. It draws itself once on load.
function beat(x) {
  return (
    `L${x + 30},100 Q${x + 50},78 ${x + 70},100 L${x + 90},100 L${x + 98},110 ` +
    `L${x + 112},16 L${x + 128},176 L${x + 138},100 L${x + 160},100 ` +
    `Q${x + 190},58 ${x + 220},100 L${x + 240},100`
  );
}

export default function Ecg({ beats = 5, className = "", draw = true }) {
  let d = "M0,100";
  for (let i = 0; i < beats; i++) d += " " + beat(i * 240);
  return (
    <svg viewBox={`0 0 ${beats * 240} 200`} preserveAspectRatio="none"
         className={className} aria-hidden>
      <path d={d} pathLength="1" fill="none" stroke="rgb(var(--accent))"
            strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
            className={draw ? "ecg-line ecg-glow" : ""} />
    </svg>
  );
}

export function Wordmark({ className = "" }) {
  return (
    <span className={`inline-flex items-center gap-2 font-display font-extrabold tracking-tight ${className}`}>
      <svg viewBox="0 0 28 20" className="h-4 w-6" aria-hidden>
        <path d="M0,10 L7,10 L10,3 L14,18 L17,10 L28,10" fill="none"
              stroke="rgb(var(--accent))" strokeWidth="2.6"
              strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      CardioSense
    </span>
  );
}
