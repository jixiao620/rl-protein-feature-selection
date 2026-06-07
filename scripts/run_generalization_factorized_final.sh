#!/bin/bash
#SBATCH --job-name=eval_fact_f
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/jl1401/rl_feature_selection/logs/eval_factorized_final_%j.out
#SBATCH --error=/work/jl1401/rl_feature_selection/logs/eval_factorized_final_%j.err

set -e
REPO_DIR="/work/jl1401/rl_feature_selection"
ORIG="/work/jl1401/icl_protein_disease/data/data/Original_extracted/Original"

source /work/jl1401/miniconda3/etc/profile.d/conda.sh
conda activate rlfs
cd "${REPO_DIR}"

CHECKPOINT="/work/jl1401/rl_feature_selection/runs/20260502-185658_factorized_10disease/checkpoint_ep20000_step1000000.pth"

python scripts/evaluate_generalization_factorized.py \
    --checkpoint "${CHECKPOINT}" \
    --disease-codes I11 I119 I129 I80 \
    --disease-path "${ORIG}/data/binary_csv.gz" \
    --protein-paths \
        "${ORIG}/data/xaa.gz" \
        "${ORIG}/data/xaa_2.gz" \
        "${ORIG}/data/xab.gz" \
        "${ORIG}/data/xac.gz" \
        "${ORIG}/data/xad.gz" \
        "${ORIG}/data/xae.gz" \
        "${ORIG}/data/xaf.gz" \
        "${ORIG}/data/xag.gz" \
        "${ORIG}/data/xah.gz" \
        "${ORIG}/data/xai.gz" \
        "${ORIG}/data/xaj.gz" \
    --embedding-path "${REPO_DIR}/embeddings/disease_embeddings.tsv" \
    --train-size 15000 \
    --sample-n 20000 \
    --episode-max-steps 50 \
    --out-dir generalization_results_factorized_final

echo "=== Done ==="
