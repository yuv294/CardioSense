// Generate a one-page PDF health-assessment report with jsPDF.
// Pure client-side: no data leaves the browser. The layout is deliberately
// plain and document-like — this is something a person might hand to a doctor.
import { jsPDF } from "jspdf";

const INK = [22, 38, 58];
const TEAL = [14, 124, 134];
const MUTE = [110, 120, 130];
const BANDHEX = { low: [59, 140, 110], intermediate: [201, 138, 43], high: [181, 80, 60] };

export function generateReport({ result, inputs, schema, name }) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const M = 48;
  let y = 56;

  // header
  doc.setFont("helvetica", "bold"); doc.setFontSize(20); doc.setTextColor(...INK);
  doc.text("CardioSense", M, y);
  doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(...MUTE);
  doc.text(result.outcome_long, M, y + 16, { maxWidth: W - 2 * M - 150 });
  const dateStr = new Date().toLocaleDateString(undefined,
    { year: "numeric", month: "long", day: "numeric" });
  doc.text(name ? `${name}  ·  ${dateStr}` : dateStr, W - M, y, { align: "right" });
  y += 34;
  doc.setDrawColor(220); doc.line(M, y, W - M, y); y += 30;

  // the headline number
  const band = result.band;
  doc.setFont("helvetica", "bold"); doc.setFontSize(46);
  doc.setTextColor(...(BANDHEX[band] || INK));
  doc.text(`${result.risk_percent}%`, M, y + 10);
  doc.setFontSize(13); doc.setTextColor(...INK);
  doc.text(result.band_label, M + 150, y - 6);
  doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(...MUTE);
  const blurb = doc.splitTextToSize(result.band_text, W - M - (M + 150));
  doc.text(blurb, M + 150, y + 12);
  y += 52;

  doc.setFontSize(9); doc.setTextColor(...MUTE);
  const ctx = [`Typical person in the cohort: ${result.typical_risk_percent}%`,
    ...result.published_scores.map((p) => `${p.label}: ${p.risk_percent}% (recalibrated ${p.recalibrated_percent}%)`)];
  doc.text(ctx.join("   ·   "), M, y, { maxWidth: W - 2 * M }); y += 22;
  doc.setDrawColor(235); doc.line(M, y, W - M, y); y += 26;

  // top factors
  doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.setTextColor(...INK);
  doc.text("What drove this estimate", M, y); y += 18;
  doc.setFont("helvetica", "normal"); doc.setFontSize(10);
  result.factors.filter((f) => Math.abs(f.contribution) >= 0.02).slice(0, 6).forEach((f) => {
    const up = f.direction === "increases";
    doc.setTextColor(...(up ? BANDHEX.high : BANDHEX.low)); doc.text(up ? "+" : "-", M, y);
    doc.setTextColor(...INK);
    doc.text(`${f.factor}: ${up ? "raises" : "lowers"} risk (odds x${Math.exp(f.contribution).toFixed(2)} vs a typical person)`, M + 16, y);
    y += 16;
  });
  y += 8;
  if (result.scenarios.length) {
    doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.setTextColor(...INK);
    doc.text("If modifiable factors reached their targets", M, y); y += 18;
    doc.setFont("helvetica", "normal"); doc.setFontSize(10);
    result.scenarios.forEach((sc) => { doc.text(`${sc.label}: ${sc.risk_percent}%`, M + 16, y); y += 15; });
    y += 8;
  }

  // input summary
  doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.setTextColor(...INK);
  doc.text("Your inputs", M, y); y += 18;
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(...INK);
  const meta = Object.fromEntries((schema?.features || []).map((f) => [f.name, f]));
  const entries = Object.entries(inputs).filter(([k]) => meta[k]);
  const colW = (W - 2 * M) / 2;
  entries.forEach(([k, v], i) => {
    const m = meta[k] || {};
    let shown = v;
    if (m.options) shown = m.options[v] ?? v;
    if (v === "" || v == null) shown = "not provided";
    const assumed = m.unit && v !== "" && v != null ? ` ${m.unit}` : "";
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = M + col * colW;
    const yy = y + row * 15;
    doc.setTextColor(...MUTE); doc.text(`${m.label || k}:`, x, yy);
    doc.setTextColor(...INK); doc.text(`${shown}${assumed}`, x + 130, yy);
  });
  y += Math.ceil(entries.length / 2) * 15 + 20;

  // questions for a doctor
  doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.setTextColor(...TEAL);
  doc.text("Questions to discuss with a doctor", M, y); y += 18;
  doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(...INK);
  const qs = [
    "Given these numbers, what lifestyle changes would help most?",
    "Should any of my measurements be re-checked or monitored?",
    "Do my family history or symptoms change this picture?",
    "Would a full clinical risk assessment be worthwhile?",
  ];
  qs.forEach((q) => {
    const lines = doc.splitTextToSize("•  " + q, W - 2 * M);
    doc.text(lines, M, y); y += lines.length * 14 + 2;
  });
  y += 10;

  // footer disclaimer
  doc.setDrawColor(235); doc.line(M, y, W - M, y); y += 16;
  doc.setFontSize(8); doc.setTextColor(...MUTE);
  const disc = doc.splitTextToSize(
    result.disclaimer + " It is a statistical estimate for a group of similar people, not a diagnosis.", W - 2 * M);
  doc.text(disc, M, y);

  doc.save(`cardiosense-report-${new Date().toISOString().slice(0, 10)}.pdf`);
}
