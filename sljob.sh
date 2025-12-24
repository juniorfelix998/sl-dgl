#!/bin/bash
#SBATCH --job-name=SPLITFED-UNLEARNING
#SBATCH --partition=gpu-week-long
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err
# Activate conda
source /usr/local/anaconda3/etc/profile.d/conda.sh
# Activate environment
conda activate venv

echo "Starting SL LEARNING..."
# Run command
srun --exclusive -n1 \
python -u main.py \
echo "SL LEARNING COMPLETE"