from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"


def save_figure(name):
    plt.tight_layout()
    plt.savefig(OUTPUT / name, dpi=180, bbox_inches="tight")
    plt.close()


train = pd.read_csv(OUTPUT / "train_scores.csv")
test = pd.read_csv(OUTPUT / "test_scores.csv")
bands = pd.read_csv(OUTPUT / "score_bands.csv")

# ROC curves
plt.figure(figsize=(7, 5))
for name, data in (("Train", train), ("Test", test)):
    fpr, tpr, _ = roc_curve(data["label"], data["pred"])
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} AUC = {auc(fpr, tpr):.4f}")
plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend()
plt.grid(alpha=0.2)
save_figure("roc_curve.png")

# KS curves: cumulative bad rate minus cumulative good rate.
plt.figure(figsize=(7, 5))
for name, data in (("Train", train), ("Test", test)):
    ordered = data.groupby("pred")["label"].agg(["count", "sum"]).sort_index(ascending=False)
    bad = ordered["sum"]
    good = ordered["count"] - bad
    cum_bad = bad.cumsum() / bad.sum()
    cum_good = good.cumsum() / good.sum()
    population = ordered["count"].cumsum() / ordered["count"].sum()
    ks_curve = cum_bad - cum_good
    plt.plot(population, ks_curve, linewidth=2, label=f"{name} KS = {ks_curve.max():.4f}")
plt.xlabel("Cumulative population share")
plt.ylabel("Cumulative bad rate - cumulative good rate")
plt.title("KS curves")
plt.legend()
plt.grid(alpha=0.2)
save_figure("ks_curve.png")

# Score distribution by observed outcome.
plt.figure(figsize=(7, 5))
plt.hist(test.loc[test["label"] == 0, "score"], bins=40, alpha=0.65, density=True, label="Good")
plt.hist(test.loc[test["label"] == 1, "score"], bins=40, alpha=0.65, density=True, label="Bad")
plt.xlabel("Score")
plt.ylabel("Density")
plt.title("Test score distribution")
plt.legend()
plt.grid(axis="y", alpha=0.2)
save_figure("score_distribution.png")

# Bad rate and sample volume by score band.
labels = [f"{int(row.score_min)}-{int(row.score_max)}" for row in bands.itertuples()]
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(labels, bands["sample_count"], color="#9ecae1", label="Sample count")
ax1.set_xlabel("Score band")
ax1.set_ylabel("Sample count")
ax1.tick_params(axis="x", rotation=25)
ax2 = ax1.twinx()
ax2.plot(labels, bands["bad_rate"] * 100, color="#d62728", marker="o", linewidth=2, label="Bad rate")
ax2.set_ylabel("Bad rate (%)")
ax2.set_ylim(0, 100)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.title("Risk by score band")
save_figure("score_band_bad_rate.png")

print(f"Evaluation charts saved to: {OUTPUT}")
