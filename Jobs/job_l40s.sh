#!/bin/bash
#BSUB -J dino_optuna_l40s_100trials     # Job name
#BSUB -q gpul40s                        # L40S GPU queue
#BSUB -n 4                              # CPU cores
#BSUB -R "span[hosts=1]"                # Same host for all cores

# Request 1 GPU (exclusive use)
#BSUB -gpu "num=1:mode=exclusive_process"

# Memory per core (8GB → total 32GB)
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

# Walltime limit (up to 24 hours allowed on GPU queues)
#BSUB -W 12:00

# Output and error logs
#BSUB -o dino_l40s_%J.out
#BSUB -e dino_l40s_%J.err

# -------------------------------------
# Load required modules
# -------------------------------------
module load python3/3.10.11
# module load cuda/12.1   # Uncomment if needed

# -------------------------------------
# Go to project directory  (EDIT THIS)
# -------------------------------------
cd /dtu/blackhole/0f/222031/02456-DeepLearning-Group-26

# Activate virtual environment
source .venv/bin/activate

# -------------------------------------
# Run experiment — ALL categories, 100 trials
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
