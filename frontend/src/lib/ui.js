export const BAND = {
  low: {
    label: "Lower risk",
    text: "text-low", bg: "bg-low", hex: [24, 128, 86],
    meaning: "Fewer than 1 in 10 people with this profile develop heart disease within ten years.",
  },
  borderline: {
    label: "Borderline",
    text: "text-mid", bg: "bg-mid", hex: [176, 112, 8],
    meaning: "Between 1 in 10 and 1 in 5. Worth reviewing habits and numbers with a doctor.",
  },
  high: {
    label: "Higher risk",
    text: "text-high", bg: "bg-high", hex: [208, 44, 64],
    meaning: "1 in 5 or more. Worth a conversation with a clinician soon.",
  },
};

export function bandOf(p) {
  if (p < 0.1) return "low";
  if (p < 0.2) return "borderline";
  return "high";
}

export const GLOSSARY = [
  { term: "Systolic blood pressure", short: "Top number",
    body: "The pressure in your arteries each time the heart beats, the first number in a reading like 120/80. Pressure that stays high strains the heart and arteries." },
  { term: "Diastolic blood pressure", short: "Bottom number",
    body: "The pressure between beats while the heart refills, the second number in 120/80." },
  { term: "Total cholesterol", short: "mg/dL, blood test",
    body: "All the cholesterol carried in your blood. Very high levels can build up in artery walls and narrow them." },
  { term: "Body mass index", short: "Weight for height",
    body: "Weight in kilograms divided by height in metres squared. It is a rough screen and cannot tell muscle from fat, so read it with your other numbers." },
  { term: "Fasting glucose", short: "Blood sugar",
    body: "Blood sugar after several hours without food. Values that stay high point towards diabetes, which raises heart risk." },
  { term: "Hypertension", short: "Diagnosed high blood pressure",
    body: "A standing diagnosis, not a single high reading. It is one of the biggest risk factors you can change." },
  { term: "Ten-year risk", short: "What the percentage means",
    body: "The share of people with a profile like yours who develop coronary heart disease within ten years. It describes a group, not a certainty for you." },
  { term: "Calibration", short: "Do the percentages hold up?",
    body: "If the model says 20% for a thousand people, about two hundred of them should go on to have an event. That is what calibration checks." },
];
