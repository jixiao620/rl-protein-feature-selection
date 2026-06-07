#!/usr/bin/env python3
"""
Pathway enrichment analysis for proteins selected by each seed.
Uses Enrichr REST API (no extra dependencies beyond requests).
Compares enriched pathways across seeds to find common biological themes.
"""

import sys, os, json, time
sys.path.insert(0, "/work/jl1401/rl_feature_selection")

import numpy as np
import pandas as pd
import requests

R = "/work/jl1401/rl_feature_selection"
OUT = os.path.join(R, "pathway_analysis")
os.makedirs(OUT, exist_ok=True)

feat_cols = np.load(f"{R}/feat_cols.npy")

# Load selected proteins for each seed
seed_dirs = {
    42: f"{R}/generalization_results_factorized_final",
    1:  f"{R}/generalization_results_seed1",
    2:  f"{R}/generalization_results_seed2",
}
for s in range(3, 10):
    d = f"{R}/generalization_results_seed{s}"
    if os.path.isdir(d) and os.path.exists(f"{d}/selected_I11.npy"):
        seed_dirs[s] = d

print(f"Seeds: {sorted(seed_dirs.keys())}")

def clean_genes(raw_names):
    """Extract clean HGNC gene symbols; expand combined names like CGB3_CGB5_CGB8."""
    genes = []
    for name in raw_names:
        # Split on underscore for combined entries, filter numeric-only parts
        parts = name.split("_")
        for p in parts:
            p = p.strip()
            if p and not p.isdigit() and len(p) >= 2:
                genes.append(p)
    return list(dict.fromkeys(genes))  # deduplicate, preserve order

seed_genes = {}
for seed, d in seed_dirs.items():
    idxs = np.load(f"{d}/selected_I11.npy").tolist()
    raw = [feat_cols[i].split(":")[0] for i in idxs]
    genes = clean_genes(raw)
    seed_genes[seed] = genes
    print(f"  seed{seed}: {genes[:5]}... ({len(genes)} genes after cleaning)")

# Enrichr API functions
ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

def enrichr_query(gene_list, description="query"):
    """Submit gene list to Enrichr and return list ID."""
    genes_str = "\n".join(gene_list)
    resp = requests.post(
        f"{ENRICHR_URL}/addList",
        files={"list": (None, genes_str), "description": (None, description)},
        timeout=30,
    )
    if not resp.ok:
        print(f"    Enrichr addList failed ({resp.status_code}): {resp.text[:200]}")
        resp.raise_for_status()
    return resp.json()["userListId"]

def enrichr_results(list_id, library):
    """Get enrichment results for a given library."""
    resp = requests.get(
        f"{ENRICHR_URL}/enrich",
        params={"userListId": list_id, "backgroundType": library},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get(library, [])
    # Each entry: [rank, term, pval, zscore, combined_score, genes, adj_pval, ...]
    rows = []
    for entry in data:
        rows.append({
            "rank": entry[0],
            "term": entry[1],
            "pval": entry[2],
            "adj_pval": entry[6],
            "combined_score": entry[4],
            "genes": ";".join(entry[5]),
            "n_genes": len(entry[5]),
        })
    return pd.DataFrame(rows)

LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]

# Run enrichment for each seed
print("\nRunning Enrichr queries...")
all_results = {}  # {seed: {library: df}}

for seed in sorted(seed_dirs.keys()):
    genes = seed_genes[seed]
    print(f"\n  seed{seed} ({len(genes)} genes)...")
    all_results[seed] = {}
    list_id = enrichr_query(genes, description=f"RLFS_seed{seed}")
    time.sleep(0.5)

    for lib in LIBRARIES:
        try:
            df = enrichr_results(list_id, lib)
            all_results[seed][lib] = df
            sig = df[df["adj_pval"] < 0.05] if len(df) > 0 else df
            print(f"    {lib}: {len(df)} terms, {len(sig)} significant (adj_p<0.05)")
        except Exception as e:
            print(f"    {lib}: ERROR {e}")
            all_results[seed][lib] = pd.DataFrame()
        time.sleep(0.3)

# Find common enriched terms across seeds
print("\n" + "="*60)
print("CROSS-SEED PATHWAY OVERLAP ANALYSIS")
print("="*60)

seeds = sorted(seed_dirs.keys())
n_seeds = len(seeds)
summary_rows = []

for lib in LIBRARIES:
    print(f"\n--- {lib} ---")

    # Collect top terms per seed (adj_p < 0.05, or top 20 if few significant)
    seed_terms = {}
    for seed in seeds:
        df = all_results[seed].get(lib, pd.DataFrame())
        if len(df) == 0:
            seed_terms[seed] = set()
            continue
        sig = df[df["adj_pval"] < 0.05].copy()
        if len(sig) < 5:
            sig = df.head(20)
        seed_terms[seed] = set(sig["term"].tolist())

    # Count how many seeds each term appears in
    from collections import Counter
    term_counts = Counter()
    for terms in seed_terms.values():
        for t in terms:
            term_counts[t] += 1

    # Show terms in ≥ half the seeds
    threshold = max(2, n_seeds // 3)
    common = [(t, c) for t, c in term_counts.items() if c >= threshold]
    common.sort(key=lambda x: -x[1])

    print(f"  Terms in ≥{threshold}/{n_seeds} seeds: {len(common)}")
    for term, count in common[:20]:
        # Get combined scores for this term across seeds
        scores = []
        for seed in seeds:
            df = all_results[seed].get(lib, pd.DataFrame())
            if len(df) > 0:
                row = df[df["term"] == term]
                if len(row) > 0:
                    scores.append(row.iloc[0]["combined_score"])
        mean_score = np.mean(scores) if scores else 0
        print(f"    [{count}/{n_seeds} seeds, score={mean_score:.1f}] {term}")
        summary_rows.append({
            "library": lib,
            "term": term,
            "n_seeds": count,
            "mean_combined_score": round(mean_score, 2),
        })

# Save all per-seed results
for seed in seeds:
    seed_out = os.path.join(OUT, f"seed{seed}")
    os.makedirs(seed_out, exist_ok=True)
    for lib in LIBRARIES:
        df = all_results[seed].get(lib, pd.DataFrame())
        if len(df) > 0:
            df.to_csv(os.path.join(seed_out, f"{lib}.csv"), index=False)

# Save cross-seed summary
df_summary = pd.DataFrame(summary_rows)
if len(df_summary) > 0:
    df_summary = df_summary.sort_values(["library", "n_seeds", "mean_combined_score"],
                                         ascending=[True, False, False])
    df_summary.to_csv(f"{OUT}/cross_seed_common_pathways.csv", index=False)

    print(f"\n{'='*60}")
    print(f"TOP SHARED PATHWAYS (across all libraries)")
    print(f"{'='*60}")
    top = df_summary[df_summary["n_seeds"] >= n_seeds // 2].sort_values(
        ["n_seeds", "mean_combined_score"], ascending=[False, False])
    for _, row in top.head(30).iterrows():
        print(f"  [{row['n_seeds']}/{n_seeds}] {row['term'][:70]}  ({row['library'].split('_')[0]})")

# Also: gene-level overlap for interest
print(f"\n{'='*60}")
print(f"MOST FREQUENTLY SELECTED GENES (≥2 seeds)")
print(f"{'='*60}")
from collections import Counter
gene_counter = Counter()
for genes in seed_genes.values():
    for g in genes:
        gene_counter[g] += 1
for gene, cnt in sorted(gene_counter.items(), key=lambda x: -x[1]):
    if cnt >= 2:
        print(f"  {gene}: {cnt}/{n_seeds} seeds")

with open(f"{OUT}/summary.json", "w") as f:
    json.dump({
        "n_seeds": n_seeds,
        "seeds": seeds,
        "libraries": LIBRARIES,
        "n_common_terms_per_library": {
            lib: int(len(df_summary[df_summary["library"] == lib]))
            for lib in LIBRARIES
        },
    }, f, indent=2)

print(f"\nResults saved to {OUT}/")
