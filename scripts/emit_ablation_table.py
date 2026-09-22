"""Print the ablation table (Markdown) using new LR-based numbers.

Compares:
  Random 50-protein panel (LR)         — lower bound
  RL + hyperbolic embedding (LR)       — our method (n=15000)
  Supervised ceiling on all 2941 (LR)  — upper bound

All numbers use L2 LR + StandardScaler + class_weight balanced,
identical fixed split (first 15000 train / rest test), sample_n=20000.
"""
import json
import numpy as np

with open("analysis_new/results.json") as f:
    R = json.load(f)

DISEASES = ["I11", "I119", "I129", "I80"]
DESC = {"I11": "Hypertensive heart disease",
        "I119": "HTN heart dis. w/o HF",
        "I129": "Hypertensive CKD",
        "I80":  "Phlebitis / thrombophlebitis"}

rand_means, rand_stds = [], []
panel_means, panel_stds = [], []
ceils = []

print("| Disease | Description | Random 50-panel | **RL + emb (50-panel, ours)** | Supervised ceiling (all 2941) |")
print("|---------|-------------|-----------------|-------------------------------|-------------------------------|")
for code in DISEASES:
    c = R[code]
    rand_m, rand_s = c["random_mean"], c["random_std"]
    p_m, p_s = c["panel_mean"], c["panel_std"]
    ceil = c["ceiling_all2941"]
    rand_means.append(rand_m); rand_stds.append(rand_s)
    panel_means.append(p_m); panel_stds.append(p_s)
    ceils.append(ceil)
    print(f"| {code} | {DESC[code]} | {rand_m:.3f} ± {rand_s:.3f} | **{p_m:.3f} ± {p_s:.3f}** | {ceil:.3f} |")

# mean row
r_m = np.mean(rand_means); p_m = np.mean(panel_means); c_m = np.mean(ceils)
print(f"| **Mean** |   | **{r_m:.3f}** | **{p_m:.3f}** | **{c_m:.3f}** |")
print()
print(f"Ceiling gap (mean): {c_m - p_m:+.3f} AUROC "
      f"({100*(c_m - p_m)/c_m:.1f}% relative)")
print(f"Random-to-ours gap (mean): {p_m - r_m:+.3f} AUROC "
      f"({100*(p_m - r_m)/r_m:.1f}% relative)")
print(f"Fraction of ceiling gap closed (vs random): "
      f"{100*(p_m - r_m)/max(c_m - r_m, 1e-6):.1f}%")
