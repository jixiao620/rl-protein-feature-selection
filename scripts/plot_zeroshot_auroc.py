"""Regenerate the zero-shot AUROC figure using README-reported numbers.

Grouped bar chart by disease, three bars per group (Random / RL no-emb / RL+emb)
with SD-across-seeds error bars. Numbers reproduced from README Zero-Shot table.
"""
import matplotlib.pyplot as plt
import numpy as np

DISEASES = ["I11", "I119", "I129", "I80"]
LABELS = [
    "I11\nHypertensive\nheart disease",
    "I11.9\nHTN heart dis.\nw/o HF",
    "I12.9\nHypertensive\nCKD",
    "I80\nPhlebitis /\nthrombophlebitis",
]

# From README (mean, sd)
random_m  = [0.716, 0.747, 0.878, 0.644]
random_s  = [0.047, 0.060, 0.033, 0.016]
noemb_m   = [0.737, 0.736, 0.875, 0.638]
noemb_s   = [0.083, 0.106, 0.029, 0.003]
withemb_m = [0.782, 0.784, 0.880, 0.654]
withemb_s = [0.028, 0.018, 0.024, 0.010]

x = np.arange(len(DISEASES))
w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 5.2))

c_rand = "#7f8c8d"
c_noemb = "#f39c12"
c_full = "#2c7fb8"

b1 = ax.bar(x - w, random_m,  w, yerr=random_s,  color=c_rand,
            edgecolor="black", lw=0.5, capsize=3, label="Random 50 proteins")
b2 = ax.bar(x,     noemb_m,   w, yerr=noemb_s,   color=c_noemb,
            edgecolor="black", lw=0.5, capsize=3, label="RL, no disease embedding")
b3 = ax.bar(x + w, withemb_m, w, yerr=withemb_s, color=c_full,
            edgecolor="black", lw=0.5, capsize=3, label="RL + hyperbolic disease embedding (ours)")

# annotate values on top of each bar
for bars, means in ((b1, random_m), (b2, noemb_m), (b3, withemb_m)):
    for r, m in zip(bars, means):
        ax.text(r.get_x() + r.get_width()/2, m + 0.008, f"{m:.3f}",
                ha="center", va="bottom", fontsize=8.5)

# Reference lines: mean across diseases
ax.axhline(np.mean(withemb_m), ls="--", color=c_full, alpha=0.5, lw=1)
ax.text(len(DISEASES)-0.3, np.mean(withemb_m)+0.003,
        f"Ours mean = {np.mean(withemb_m):.3f}",
        fontsize=9, color=c_full, ha="right")

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=9.5)
ax.set_ylabel("AUROC on held-out disease (mean ± SD, 3 seeds)", fontsize=11)
ax.set_title("Zero-shot generalization to unseen cardiovascular diseases",
             fontsize=12.5)
ax.set_ylim(0.55, 1.0)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left", frameon=False, fontsize=10)

# annotate "trained on 10 held-in I-block diseases" below the plot
fig.text(0.5, -0.02,
         "Trained on 10 in-block diseases (I10 I25 I48 I519 I20 I21 I84 I83 I95 I50)  ·  "
         "evaluated zero-shot on 4 held-out",
         ha="center", va="top", fontsize=9, style="italic", color="#555")

plt.tight_layout()
out = "figures/zeroshot_auroc.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")
print(f"means: random={np.mean(random_m):.3f} no-emb={np.mean(noemb_m):.3f} w-emb={np.mean(withemb_m):.3f}")
