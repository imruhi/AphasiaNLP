#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --gpus-per-node=2
#SBATCH --mem=5GB
#SBATCH --job-name=phi2
#SBATCH --partition=gpu

source $HOME/venvs/first_env/bin/activate

python phi.py