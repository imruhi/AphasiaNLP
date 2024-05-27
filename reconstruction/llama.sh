#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=5GB
#SBATCH --job-name=llama_finetune
#SBATCH --partition=gpu

source $HOME/venvs/first_env/bin/activate

python llama.py