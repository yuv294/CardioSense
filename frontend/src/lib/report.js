// Generate a one-page PDF health-assessment report with jsPDF.
// Pure client-side: no data leaves the browser. The layout is deliberately
// plain and document-like — this is something a person might hand to a doctor.
import { jsPDF } from "jspdf";
import { BAND } from "./ui.js";

const INK = [22, 38, 58];
const TEAL = [14, 124, 134];
const MUTE = [110, 120, 130];
const BANDHEX = { low: [59, 140, 110], borderline: [201, 138, 43], high: [181, 80, 60] };

export function generateReport({ result, inputs, schema, name }) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const W = doc.internal.pageSize.getWidth();
  const M = 48;
  let y = 56;

  // header
  doc.setFont("helvetica", "bold"); doc.setFontSize(20); doc.setTextColor(...INK);
  doc.text("CardioSense", M, y);
  doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(...MUTE);
  doc.text("10-year coronary heart disease risk estimate", M, y + 16);
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
  doc.text(BAND[band].label, M + 150, y - 6);
  doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(...MUTE);
  const blurb = doc.splitTextToSize(BAND[band].meaning, W - M - (M + 150));
  doc.text(blurb, M + 150, y + 12);
  y += 52;

  if (result.flagged_high_risk) {
    doc.setTextColor(...BANDHEX.high); doc.setFontSize(9);
    doc.text("Flagged for follow-up at the model's screening threshold.", M, y);
    y += 18;
  }
  doc.setDrawColor(235); doc.line(M, y, W - M, y); y += 26;

  // top factors
  doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.setTextColor(...INK);
  doc.text("What drove this estimate", M, y); y += 18;
  doc.setFont("helvetica", "normal"); doc.setFontSize(10);
  result.top_factors.forEach((f) => {
    const arrow = f.direction === "increases" ? "↑" : "↓";
    const col = f.direction === "increases" ? BANDHEX.high : BANDHEX.low;
    doc.setTextColor(...col); doc.text(arrow, M, y);
    doc.setTextColor(...INK);
    doc.text(`${f.label} — ${f.direction} risk`, M + 16, y);
    y += 16;
  });
  y += 12;

  // input summary
  doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.setTextColor(...INK);
  doc.text("Your inputs", M, y); y += 18;
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(...INK);
  const meta = Object.fromEntries((schema?.features || []).map((f) => [f.name, f]));
  const entries = Object.entries(inputs);
  const colW = (W - 2 * M) / 2;
  entries.forEach(([k, v], i) => {
    const m = meta[k] || {};
    let shown = v;
    if (m.kind === "binary" && m.options) shown = m.options[v] ?? v;
    const assumed = (result.assumed_fields || []).includes(k) ? "  (assumed)" : "";
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
    "Statistical estimate from the Framingham Heart Study cohort. It is not a diagnosis.", W - 2 * M);
  doc.text(disc, M, y);

  doc.save(`cardiosense-report-${new Date().toISOString().slice(0, 10)}.pdf`);
}
