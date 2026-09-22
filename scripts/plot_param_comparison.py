"""Compare parameter counts: standard DQN (4-hidden-layer MLP) vs Factorized DQN.

Standard MLP-Q (rlfs/models/q_network.py):
    obs_dim = N + 1 (mask + timestep, no disease emb)
    hidden -> hidden -> hidden -> hidden -> N
Factorized (rlfs/models/factorized_q_network.py):
    state_dim = N + 1 + 10 (mask + timestep + disease emb)
    hidden -> hidden -> emb_dim; plus N x emb_dim embedding table

We report:
  - total params
  - output-head params only (the piece that grows most aggressively with N):
      standard: hidden*N + N     (final Linear -> N actions)
      factorized: hidden*emb_dim + emb_dim + N*emb_dim
"""
import matplotlib.pyplot as plt
import numpy as np


def standard_dqn_params(n, hidden=256, n_hidden_layers=4):
    obs = n + 1
    p = 0
    p += obs * hidden + hidden
    for _ in range(n_hidden_layers - 1):
        p += hidden * hidden + hidden
    p += hidden * n + n
    return p


def standard_dqn_head_params(n, hidden=256, emb_dim=64):
    return hidden * n + n


def factorized_params(n, hidden=256, emb_dim=64, disease_dim=10):
    state = n + 1 + disease_dim
    p = 0
    p += state * hidden + hidden
    p += hidden * hidden + hidden
    p += hidden * emb_dim + emb_dim
    p += n * emb_dim
    return p


def factorized_head_params(n, hidden=256, emb_dim=64):
    return hidden * emb_dim + emb_dim + n * emb_dim


Ns = np.array([100, 250, 500, 1000, 2000, 2941, 5000, 10000])
std_total = np.array([standard_dqn_params(n) for n in Ns])
fac_total = np.array([factorized_params(n) for n in Ns])
std_head = np.array([standard_dqn_head_params(n) for n in Ns])
fac_head = np.array([factorized_head_params(n) for n in Ns])

print(f"{'N':>6}  {'std total':>12}  {'fac total':>12}  {'std head':>12}  {'fac head':>12}")
for n, s, f, sh, fh in zip(Ns, std_total, fac_total, std_head, fac_head):
    print(f"{n:>6}  {s:>12,}  {f:>12,}  {sh:>12,}  {fh:>12,}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
colour_std = "#c0392b"
colour_fac = "#2c7fb8"

ax1.plot(Ns, std_total / 1e6, marker="o", lw=2, color=colour_std, label="Standard DQN (4-layer MLP)")
ax1.plot(Ns, fac_total / 1e6, marker="s", lw=2, color=colour_fac, label="Factorized DQN (ours)")
ax1.axvline(2941, ls="--", color="gray", alpha=0.5)
ax1.text(2941 * 1.03, ax1.get_ylim()[1] * 0.05, "UKB Olink\nN = 2941",
         fontsize=9, color="gray", va="bottom")
ax1.set_xlabel("Number of proteins (action space size) N", fontsize=11)
ax1.set_ylabel("Total parameters (M)", fontsize=11)
ax1.set_title("(a) Total model parameters", fontsize=12)
ax1.legend(loc="upper left", frameon=False, fontsize=10)
ax1.grid(alpha=0.3)

ax2.plot(Ns, std_head / 1e3, marker="o", lw=2, color=colour_std, label="Standard: H·N")
ax2.plot(Ns, fac_head / 1e3, marker="s", lw=2, color=colour_fac, label="Factorized: (H+N)·d")
ax2.axvline(2941, ls="--", color="gray", alpha=0.5)
i_target = np.where(Ns == 2941)[0][0]
saving = std_head[i_target] / fac_head[i_target]
ax2.annotate(
    f"{saving:.1f}× smaller\nat N = 2941",
    xy=(2941, fac_head[i_target] / 1e3),
    xytext=(2941 * 1.15, std_head[i_target] / 1e3 * 0.55),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
)
ax2.set_xlabel("Number of proteins (action space size) N", fontsize=11)
ax2.set_ylabel("Action-head parameters (K)", fontsize=11)
ax2.set_title("(b) Action-space-related parameters", fontsize=12)
ax2.legend(loc="upper left", frameon=False, fontsize=10)
ax2.grid(alpha=0.3)

plt.suptitle(
    "Factorized Q-network reduces action-head parameters by ~$4\\times$ (H=256, d=64)",
    fontsize=12,
    y=1.02,
)
plt.tight_layout()
out = "figures/param_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"\nSaved: {out}")
print(f"At N=2941: std_total={std_total[i_target]:,}, fac_total={fac_total[i_target]:,}, "
      f"reduction={std_total[i_target]/fac_total[i_target]:.2f}x")
print(f"At N=2941: std_head={std_head[i_target]:,}, fac_head={fac_head[i_target]:,}, "
      f"reduction={std_head[i_target]/fac_head[i_target]:.2f}x")
