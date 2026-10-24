// Band styling. The band edges themselves come from the API (they differ by
// calculator: 10/20% for CHD events, 1/5% for CVD death).
export const BAND = {
  low: { text: "text-low", bg: "bg-low", soft: "bg-low/10", hex: [24, 128, 86] },
  intermediate: { text: "text-mid", bg: "bg-mid", soft: "bg-mid/10", hex: [176, 112, 8] },
  high: { text: "text-high", bg: "bg-high", soft: "bg-high/10", hex: [208, 44, 64] },
};
// records saved by v1 used "borderline"
BAND.borderline = BAND.intermediate;

export const CALCULATORS = {
  framingham: {
    short: "Quick check",
    blurb: "12 everyday questions. Estimates your chance of a first heart attack, angina or other coronary event.",
    cohort: "Framingham Heart Study",
  },
  nhanes: {
    short: "Full check",
    blurb: "Adds kidney, sugar and inflammation lab results. Estimates your chance of dying from heart disease or stroke.",
    cohort: "NHANES, United States",
  },
};

export const MODEL_NAMES = {
  logreg: "Logistic regression", rf: "Random forest", gbm: "Gradient boosting", mlp: "Neural network",
  ensemble: "Ensemble of the four", refit: "Published formula, refitted", frs_best: "Framingham score",
  frs_office: "Framingham office score", frs_lab: "Framingham lab score", pce: "Pooled Cohort Equations",
  frs_best_recal: "Framingham, recalibrated", pce_recal: "PCE, recalibrated", served: "Served model",
  frs_office_recal: "Framingham office, recalibrated",
};

export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
export const f3 = (v) => (v == null ? "–" : v.toFixed(3));

export const GLOSSARY = [
  { term: "Ten-year risk", short: "What the percentage means",
    body: "Out of 100 people with a profile like yours, how many have the event within ten years. It describes a group, not a certainty about you." },
  { term: "Systolic blood pressure", short: "Top number",
    body: "The pressure in your arteries each time the heart beats, the first number in a reading like 120/80. Pressure that stays high strains the heart and arteries." },
  { term: "Diastolic blood pressure", short: "Bottom number",
    body: "The pressure between beats while the heart refills, the second number in 120/80." },
  { term: "Total and HDL cholesterol", short: "Blood fats",
    body: "Total cholesterol is everything carried in your blood. HDL is the part that carries cholesterol away from artery walls, so higher HDL goes with lower risk." },
  { term: "Body mass index and waist", short: "Weight for height",
    body: "BMI is weight in kilograms divided by height in metres squared. Waist size adds where the weight sits; fat around the middle is the more harmful kind." },
  { term: "HbA1c", short: "3-month blood sugar",
    body: "The share of red blood cells carrying sugar. 6.5% or more is in the diabetes range; 5.7 to 6.4% is prediabetes." },
  { term: "eGFR", short: "Kidney filtering",
    body: "How well the kidneys filter blood, worked out from a creatinine blood test. Below 60 means reduced kidney function, which goes with higher heart risk." },
  { term: "Urine albumin / creatinine", short: "Protein leak",
    body: "A small urine test. Albumin leaking into urine (30 mg/g or more) is an early sign of blood-vessel damage, in the kidneys and elsewhere." },
  { term: "hs-CRP", short: "Inflammation",
    body: "A blood marker of inflammation. Values above 3 mg/L are linked with higher heart risk." },
  { term: "Calibration", short: "Do the percentages hold up?",
    body: "If a model says 20% for a thousand people, about two hundred of them should go on to have the event. That is what calibration checks." },
  { term: "AUROC", short: "Ranking accuracy",
    body: "Pick one person who had the event and one who did not. AUROC is how often the model gives the first person the higher risk. 0.5 is a coin flip, 1.0 is perfect." },
];
