"""Static report figures (PNG) from the JSON reports, for the README and slides.

    python figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import COHORTS, REPORTS  # noqa: E402
from models import LABELS  # noqa: E402


def main():
    for name in COHORTS:
        p = REPORTS / name / "evaluation.json"
        if not p.exists():
            continue
        ev = json.loads(p.read_text())
        fig_dir = REPORTS / name / "figures"
        served, base = ev["served_model"], ev["primary_baseline"]

        rows = sorted(((m, r) for m, r in ev["metrics"].items() if not m.endswith("_recal")), key=lambda kv: kv[1]["auroc"])
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1))
        for i, (m, r) in enumerate(rows):
            col = "#0e7c86" if m == served else "#b07008" if m in ev["metrics"] and m.startswith(("frs", "pce")) else "#8a96a3"
            ax.errorbar(r["auroc"], i, xerr=[[r["auroc"] - r["auroc_ci"][0]], [r["auroc_ci"][1] - r["auroc"]]],
                        fmt="o", color=col, capsize=3)
        ax.set_yticks(range(len(rows)), [LABELS.get(m, m) for m, _ in rows], fontsize=8)
        ax.set_xlabel("AUROC on the locked test set (95% CI)")
        ax.set_title(ev["title"], fontsize=10)
        fig.tight_layout(); fig.savefig(fig_dir / "auroc.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        for m, col in ((served, "#0e7c86"), (base, "#b07008"), (f"{base}_recal", "#d02c40")):
            pts = ev["calibration_curves"][m]
            ax.plot([q["predicted"] for q in pts], [q["observed"] for q in pts], "o-", color=col, label=LABELS.get(m.replace("_recal", ""), m) + (" (recal.)" if m.endswith("_recal") else ""))
        mx = max(max(q["predicted"], q["observed"]) for k in ev["calibration_curves"].values() for q in k)
        ax.plot([0, mx], [0, mx], "k--", lw=1)
        ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed risk"); ax.legend(fontsize=8); ax.set_title("Calibration", fontsize=10)
        fig.tight_layout(); fig.savefig(fig_dir / "calibration.png", dpi=150); plt.close(fig)

        d = ev["decision_curve"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(d["thresholds"], d["treat_all"], "--", color="#8a96a3", label="Treat all")
        ax.axhline(0, color="k", lw=1)
        for m, col in ((served, "#0e7c86"), (base, "#b07008"), (f"{base}_recal", "#d02c40")):
            ax.plot(d["thresholds"], d[m], color=col, label=LABELS.get(m.replace("_recal", ""), m))
        top = max(d[served])
        ax.set_ylim(-0.2 * top, top * 1.15); ax.set_xlabel("Risk threshold"); ax.set_ylabel("Net benefit")
        ax.legend(fontsize=8); ax.set_title("Decision curve", fontsize=10)
        fig.tight_layout(); fig.savefig(fig_dir / "decision_curve.png", dpi=150); plt.close(fig)
        print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
