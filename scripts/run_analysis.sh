#!/bin/bash
#SBATCH --job-name=rlfs_analysis
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

set -e
cd /work/jl1401/rl_feature_selection
mkdir -p logs analysis_new

source /work/jl1401/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn_v2

python scripts/ablation_and_data_reduction.py --out-dir analysis_new
