#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=2GB
#SBATCH --job-name=inference
#SBATCH --partition=gpu

module --force purge
source $HOME/venvs/thesis/bin/activate

python inference.py
