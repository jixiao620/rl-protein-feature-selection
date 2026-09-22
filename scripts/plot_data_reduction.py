"""Plot AUROC vs classifier train_size for each held-out disease.

Reads analysis_new/results.json and creates figures/data_reduction.png.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("analysis_new/results.json") as f:
    R = json.load(f)

DISEASES = ["I11", "I119", "I129", "I80"]
TITLES = {"I11": "I11 · Hypertensive heart disease",
          "I119": "I11.9 · HTN heart disease w/o HF",
          "I129": "I12.9 · Hypertensive CKD",
          "I80":  "I80 · Phlebitis / thrombophlebitis"}

colors = {"I11": "#c0392b", "I119": "#e67e22", "I129": "#2c7fb8", "I80": "#27ae60"}

fig, ax = plt.subplots(figsize=(8.5, 5))

for code in DISEASES:
    dr = R[code]["data_reduction"]
    tsizes = sorted(int(k) for k in dr.keys())
    means = [dr[str(t)]["mean"] for t in tsizes]
    stds  = [dr[str(t)]["std"]  for t in tsizes]
    ax.errorbar(tsizes, means, yerr=stds, marker="o", capsize=3, lw=1.8,
                color=colors[code], label=TITLES[code])

ax.set_xscale("log")
ax.set_xticks([500, 1500, 5000, 15000])
ax.set_xticklabels(["500", "1,500", "5,000", "15,000"])
ax.set_xlabel("Classifier training set size (patients)", fontsize=11)
ax.set_ylabel("AUROC on held-out test set (mean ± SD, 3 panels)", fontsize=11)
ax.set_title("Robustness of RL-selected 50-protein panels to reduced training data",
             fontsize=12)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9.5, frameon=False)

plt.tight_layout()
out = "figures/data_reduction.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")

# Also print a compact table
print("\nAUROC vs train size (LR classifier, mean±SD across seeds 42,1,2):")
print(f"{'disease':<8} " + "  ".join(f"n={t:<5}" for t in [500,1500,5000,15000]))
for code in DISEASES:
    dr = R[code]["data_reduction"]
    row = f"{code:<8} " + "  ".join(f"{dr[str(t)]['mean']:.3f}±{dr[str(t)]['std']:.3f}" for t in [500,1500,5000,15000])
    print(row)
