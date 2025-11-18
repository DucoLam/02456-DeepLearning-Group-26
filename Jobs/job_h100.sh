#!/bin/bash
#BSUB -J dino_optuna_h100_100trials     # Job name
#BSUB -q gpuh100                        # H100 GPU queue (change if name differs)
#BSUB -n 4                              # CPU cores
#BSUB -R "span[hosts=1]"                # All cores on same host

# Request 1 H100 GPU in exclusive mode
#BSUB -gpu "num=1:mode=exclusive_process"

# Memory per core (8GB → total 32GB)
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

# Walltime limit
#BSUB -W 12:00

# Output and error logs
#BSUB -o dino_h100_%J.out
#BSUB -e dino_h100_%J.err

# -------------------------------------
# Load required modules
# -------------------------------------
module load python3/3.10.11
# If needed on this queue, also:
# module load cuda/12.1

# -------------------------------------
# Go to project directory
# -------------------------------------
cd /dtu/blackhole/0f/222031/02456-DeepLearning-Group-26

# Activate virtual environment
source .venv/bin/activate

# -------------------------------------
# Run your experiment — ALL categories, 100 trials
# -------------------------------------
python run_mvtec_optuna_study.py \
    --mvtec-root ./Dataset \
    --categories bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper \
    --batch-size 500 \
    --n-trials 100 \
    --percentile-min 0.0 \
    --percentile-max 1.0 \
    --threshold-min 0.4 \
    --threshold-max 1.0 \
    --device cuda
