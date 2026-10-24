// The method in plain language, with the literature it rests on.
const STEPS = [
  ["Two cohorts, two questions",
   "Framingham (4,434 people, 1956–68, 24-year follow-up) is the home ground of the Framingham score: it was derived there. NHANES 1999–2010 (13,702 US adults aged 40–79, deaths tracked to 2019) is an external, multi-ethnic, modern cohort with lab results the published scores do not use."],
  ["Clean outcomes",
   "Framingham: first coronary event within 10 years of a visit, rebuilt from adjudicated event times; people with prior heart disease or stroke removed. NHANES: death from heart disease or stroke within 10 years, from the CDC's linked death records. Deaths from other causes count as no event."],
  ["Locked test set",
   "One in five people is set aside before any modelling and scored once at the end. Each Framingham participant contributes up to three visits (landmarking), so every split keeps a person's visits together."],
  ["Four model families, tuned for small clinical data",
   "Elastic-net logistic regression with splines; random forest with large leaves; LightGBM with monotone constraints (risk can only rise with age, blood pressure, smoking…); a bagged neural network with weight decay. Settings are chosen by grouped cross-validation on log-loss, then the most regularised setting within one standard error of the best is taken. Each model is recalibrated on its own out-of-fold predictions, and the four are averaged into an ensemble."],
  ["Fair baselines",
   "The published Framingham score and Pooled Cohort Equations exactly as published, the same scores recalibrated to the cohort, and the published formula refitted on our data. The last one separates the value of machine learning and new inputs from the value of simply refitting."],
  ["Honest statistics",
   "Participant-level bootstrap intervals; paired differences; calibration intercept and slope; decision-curve net benefit; subgroup results; and nested cross-validation over the whole cohort so that every event counts."],
];

const LIT = [
  ["D'Agostino RB et al. General cardiovascular risk profile for use in primary care. Circulation 2008.", "The Framingham score used as the baseline."],
  ["Goff DC et al. 2013 ACC/AHA guideline on the assessment of cardiovascular risk. Circulation 2014.", "The Pooled Cohort Equations, the US guideline score."],
  ["Christodoulou E et al. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol 2019.", "Why the Framingham-only result is a tie: the model family matters less than the information."],
  ["Weng SF et al. Can machine-learning improve cardiovascular risk prediction using routine clinical data? PLoS One 2017.", "ML gains appeared with 378,000 patients and many more variables."],
  ["Alaa AM et al. Cardiovascular disease risk prediction using automated machine learning: a prospective study of 423,604 UK Biobank participants. PLoS One 2019.", "Gains came from hundreds of extra variables, not from the algorithm alone."],
  ["Khan SS et al. Development and validation of the AHA PREVENT equations. Circulation 2024.", "Kidney function (eGFR, urine albumin) and HbA1c add information beyond the classic factors: the inputs that move NHANES."],
  ["Grinsztajn L et al. Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022.", "Why gradient boosting is the strongest single learner here."],
  ["van Houwelingen HC. Dynamic prediction by landmarking in event history analysis. Scand J Stat 2007.", "Using every Framingham visit as a starting point."],
  ["Van Calster B et al. Calibration: the Achilles heel of predictive analytics. BMC Med 2019.", "Why calibration is reported alongside AUROC."],
  ["Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making 2006.", "Net benefit: is the model useful for decisions?"],
  ["Riley RD et al. Calculating the sample size required for developing a clinical prediction model. BMJ 2020.", "Events per parameter, and why the models are heavily regularised."],
  ["Collins GS et al. TRIPOD+AI statement. BMJ 2024.", "The reporting checklist this project follows."],
];

export default function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-12">
      <header>
        <h1 className="font-display text-4xl font-extrabold tracking-tight">How CardioSense was built</h1>
        <p className="mt-4 text-lg leading-relaxed text-fg/70">
          Version 1 compared four models with the Framingham score on a one-visit Kaggle extract of the Framingham cohort,
          and none of them beat it. That result was correct, and the literature explains why: on the classic risk factors a
          well-built clinical equation is close to the ceiling, and Framingham data is that equation's home ground. Version 2
          kept the four model families but changed the question: does machine learning help when it is given information
          the published scores leave out, in a population they were not built on?
        </p>
      </header>
      <ol className="space-y-6">
        {STEPS.map(([t, b], i) => (
          <li key={t} className="grid grid-cols-[2.5rem_1fr] gap-3">
            <span className="font-display text-2xl font-extrabold text-accent">{i + 1}</span>
            <div>
              <h2 className="font-display text-lg font-bold tracking-tight">{t}</h2>
              <p className="mt-1 leading-relaxed text-fg/75">{b}</p>
            </div>
          </li>
        ))}
      </ol>
      <section>
        <h2 className="font-display text-2xl font-bold tracking-tight">Literature</h2>
        <ul className="mt-4 divide-y divide-fg/10 border-y border-fg/10">
          {LIT.map(([ref, why]) => (
            <li key={ref} className="py-4">
              <div className="text-[15px] leading-snug">{ref}</div>
              <div className="mt-1 text-sm text-fg/55">{why}</div>
            </li>
          ))}
        </ul>
      </section>
      <section className="rounded-2xl border border-mid/30 bg-mid/10 p-6 text-[15px] leading-relaxed text-fg/80">
        <b>Limitations.</b> Both cohorts are American; Framingham is mostly white and historical. NHANES deaths come from death
        certificates, so non-fatal events are not counted. Family history was asked from 2005 only. Survey weights are not used,
        so results describe the sample rather than the US population. The what-if figures are associations, not trial results.
        Nothing here has been clinically validated.
      </section>
    </div>
  );
}
