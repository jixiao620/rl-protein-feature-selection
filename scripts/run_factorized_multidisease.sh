#!/bin/bash
#SBATCH --job-name=fact_dqn
#SBATCH --partition=biostat-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/jl1401/rl_feature_selection/logs/factorized_%j.out
#SBATCH --error=/work/jl1401/rl_feature_selection/logs/factorized_%j.err

set -e
REPO_DIR="/work/jl1401/rl_feature_selection"
ORIG="/work/jl1401/icl_protein_disease/data/data/Original_extracted/Original"

source /work/jl1401/miniconda3/etc/profile.d/conda.sh
conda activate rlfs

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

# Training diseases: 10 highest-prevalence I-block diseases
# (excluding test diseases I11, I119, I129, I80)
# Prevalence: I10(51.7%) I25(23.3%) I48(19.5%) I519(17.9%) I20(17.5%)
#             I21(14.2%) I84(13.9%) I83(12.1%) I95(11.8%) I50(11.7%)
python scripts/train_factorized.py \
    --train-disease-codes I10 I25 I48 I519 I20 I21 I84 I83 I95 I50 \
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
    --max-total-steps 1000000 \
    --episode-max-steps 50 \
    --seed 42 \
    --lr 1e-4 \
    --batch-size 2048 \
    --buffer-size 200000 \
    --target-tau 0.005 \
    --protein-emb-dim 64 \
    --hidden 256 \
    --eps-end 0.05 \
    --eps-decay 50000 \
    --run-name factorized_10disease

echo "=== Done ==="
