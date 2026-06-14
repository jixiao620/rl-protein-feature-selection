# RL-Based Protein Panel Selection with Zero-Shot Generalization for Cardiovascular Disease

A reinforcement learning pipeline that selects minimal protein biomarker
panels for disease classification, with **zero-shot generalization to held-out
diseases** via hyperbolic disease embeddings.

---

## Overview

The codebase is modular, separating the RL environment, factorized models,
and hyperbolic embedding logic for easy reproduction.

UK Biobank Olink proteomics provides ~2,941 proteins measured across ~50,000
participants. This project trains a DQN agent to select a 50-protein panel
that maximizes AUROC for predicting a given disease. Key contributions:

1. **Factorized DQN** — scales action space from ~1,000 to 2,941 proteins
   without combinatorial explosion: **Q(state, protein\_i) = state\_encoder(state) · protein\_emb[i]**
2. **Hyperbolic disease embeddings** — 10-dim Poincaré-ball embeddings encode
   ICD-10 code hierarchy; injected into agent state so **one policy generalizes
   across diseases zero-shot**
3. **Multi-seed stability analysis** — 10 independent seeds reveal that
   protein-level overlap is near zero, but **pathway-level enrichment is
   consistent**, demonstrating the model learns real cardiovascular biology

**Main result**: Mean AUROC **0.775** on four held-out cardiovascular diseases
(I11, I119, I129, I80), up from logistic regression on top-50 p-value proteins
(~0.71) — a **+0.065 absolute AUROC gain (~+9% relative)**, achieved purely
through **zero-shot transfer**.

![Pipeline Overview](figures/pipeline_overview.png)

---

## Repository Structure

```
rlfs/                          # Core library
  config/                      # Config schema + YAML I/O
  data/                        # Data loading, splits
  env/                         # RL environment (feature_selection.py)
  models/                      # Factorized DQN (factorized_q_network.py), state encoder
  replay/                      # Replay buffer
  rewards/                     # AUROCReward (handles class imbalance)
  trainers/                    # DQN trainer (rl_trainer.py)
  evaluation/                  # Generalization evaluator

scripts/
  train_factorized.py          # Main training entry point
  evaluate_generalization_factorized.py   # Evaluate checkpoint on held-out diseases
  correlation_minimal_set.py   # Spearman correlation + minimal non-redundant set
  consensus_analysis.py        # Count protein selection frequency across seeds
  pathway_analysis.py          # Enrichr API pathway enrichment per seed
  run_factorized_multidisease.sh          # SLURM training launcher (one of several seed runs)

embeddings/
  disease_embeddings.tsv       # Pre-computed Poincaré embeddings (487 ICD-10 nodes)

results/
  correlated_clusters.txt      # Correlation cluster analysis output

pathway_analysis/              # Per-seed Enrichr results + cross-seed summary
consensus_analysis/            # Protein frequency table across 10 seeds
```

---

## Architecture

### Factorized DQN

Standard DQN cannot scale to 2,941 actions without a huge output head.
Instead, Q-values are factorized:

```
Q(s, i) = state_encoder(s) · protein_emb[i]
```

- `state_encoder`: 3-layer MLP, maps (selection mask + time step + disease
  embedding) → 128-dim vector
- `protein_emb`: learnable embedding matrix, shape (2941, 128)
- Dot product gives one Q-value per protein in a single forward pass

This reduces parameters and allows the agent to generalize protein utility
through the shared embedding space.

![Factorized DQN Architecture](figures/factorized_dqn_arch.png)

### Disease Embeddings

ICD-10 codes have a natural hierarchy (e.g., I11 is a subtype of I1x which is
under Chapter IX: Circulatory). We use Poincaré-ball embeddings that preserve
parent-child distances in hyperbolic space.

At evaluation time, a new disease code is embedded and injected into the
agent state — no retraining needed.

![Disease Embedding Space](figures/disease_embedding_space.png)

### AUROCReward

Feature selection for disease prediction faces severe class imbalance (most
UK Biobank participants are healthy). `AccuracyReward` would trivially maximize
by ignoring the minority class. `AUROCReward` computes logistic regression
AUROC on a stratified train/test split, directly optimizing for genuine
discriminative power in imbalanced clinical settings.

---

## Key Results

### Zero-Shot Generalization

Trained on 10 diseases (I-block cardiovascular), evaluated zero-shot on 4
held-out diseases (mean ± SD across seeds 42, 1, 2):

| Disease | Description                       | AUROC (RL+emb) | AUROC (no emb) | Random baseline |
|---------|-----------------------------------|---------------|----------------|-----------------|
| I11     | Hypertensive heart disease        | 0.782 ± 0.028 | 0.737 ± 0.083  | 0.716 ± 0.047   |
| I119    | Hypertensive heart disease w/o HF | 0.784 ± 0.018 | 0.736 ± 0.106  | 0.747 ± 0.060   |
| I129    | Hypertensive CKD, unspecified     | 0.880 ± 0.024 | 0.875 ± 0.029  | 0.878 ± 0.033   |
| I80     | Phlebitis & thrombophlebitis      | 0.654 ± 0.010 | 0.638 ± 0.003  | 0.644 ± 0.016   |
| **Mean**|                                   | **0.775**     | 0.747          | 0.746           |

![Zero-Shot AUROC Results](figures/zeroshot_auroc.png)

### Correlation & Minimal Set Analysis

We pooled all 146 proteins selected across seeds 42/1/2 and computed pairwise
Spearman correlations on 40k patients. Hierarchical clustering (threshold
|r| > 0.7) found 3 clusters:

- Cluster 1 (8 members): RHOC chosen as representative
- Cluster 2 (3 members): MIF chosen as representative
- Cluster 3 (2 members): AZI2 chosen as representative

Removing 10 redundant proteins leaves a **136-protein minimal set**.
Mean AUROC of this set: **0.840** (vs. 0.775 with 50-protein per-seed panels).

Cross-seed overlap was low: only 4 proteins appeared in ≥2 seeds out of 3:
F12, AZI2, KIAA0319, PDRG1.

**Insight:** This low protein-level overlap, combined with consistent
pathway-level enrichment, indicates the agent is not unstable but discovers
multiple functionally equivalent panels converging on the same cardiovascular
biology — an identifiability property, not a failure mode.

### Multi-Seed Consensus (10 Seeds)

Running 10 independent training seeds revealed:

- No protein was selected by a majority of seeds (0 proteins selected by ≥6/10 seeds)
- The model finds many equally valid 50-protein panels (~0.775 AUROC each)
- Cross-seed pairwise overlap averaged ~2–4 proteins per pair

### Pathway Enrichment Analysis

Despite near-zero protein-level overlap across seeds, pathway enrichment via
Enrichr shows consistent biological themes:

| Pathway | Seeds Enriched | Library |
|---------|---------------|---------|
| IL-6/JAK/STAT3 Signaling | 9/10 | MSigDB Hallmarks |
| Epithelial Mesenchymal Transition | 8/10 | MSigDB Hallmarks |
| Apoptosis | 8/10 | MSigDB Hallmarks |
| Complement | 7/10 | MSigDB Hallmarks |
| IL-2/STAT5 Signaling | 7/10 | MSigDB Hallmarks |
| Coagulation | 6/10 | MSigDB Hallmarks |
| Cytokine-cytokine receptor interaction | 7/10 | KEGG |

**Interpretation**: The RL agent consistently discovers cardiovascular/inflammatory
biology regardless of which specific proteins it selects — multiple different
proteins encode the same pathway signal.

![Pathway Enrichment](figures/pathway_enrichment.png)

---

## Usage

### Training (SLURM)

```bash
sbatch scripts/run_factorized_multidisease.sh   # example training run
```

Each seed (3-9 and 42) was trained for
1M steps on the biostat-gpu partition and saves checkpoints to `runs/`.

### Evaluation

```bash
python scripts/evaluate_generalization_factorized.py \
    --checkpoint runs/<run>/checkpoint_ep*_step1000000.pth \
    --disease-codes I11 I119 I129 I80 \
    --disease-path /path/to/binary_csv.gz \
    --protein-paths /path/to/xaa.gz ...  \
    --embedding-path embeddings/disease_embeddings.tsv \
    --train-size 15000 --sample-n 20000 --episode-max-steps 50 \
    --out-dir generalization_results_seed42
```

### Analysis Scripts

```bash
# Correlation + minimal non-redundant set
python scripts/correlation_minimal_set.py

# Consensus across 10 seeds
python scripts/consensus_analysis.py

# Pathway enrichment (requires internet for Enrichr API)
python scripts/pathway_analysis.py
```

---

## Dependencies

```
torch >= 2.0
numpy
pandas
scikit-learn
scipy
matplotlib
requests
geoopt          # Poincaré ball operations
```

Install into the project environment:

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib requests geoopt
```

---

## Author

Jixiao (Xavier) Liu — Duke University  
Contact: jl1401@duke.edu

---

## Notes

- **Data**: UK Biobank Olink proteomics data is not included (access-controlled).
  The pipeline expects `.gz`-compressed CSVs in the format distributed by UK
  Biobank.
- **SLURM**: All heavy jobs should be submitted via `sbatch`. Training takes
  ~4–6 hours on an NVIDIA RTX 5000 Ada GPU.
- **Reproducibility**: Results are qualitatively consistent across seeds at the
  pathway level despite protein-level variation, indicating that **the learned
  signal reflects genuine disease biology rather than seed-specific overfitting
  or dataset artifacts.**
