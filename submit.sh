#!/bin/bash
#SBATCH --job-name=grokking
#SBATCH --output=grokking_%j.out
#SBATCH --error=grokking_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

module unload cuda 2>/dev/null
module load anaconda3/2024.06
module load cuda/12.8.0

source activate llms

python run.py
