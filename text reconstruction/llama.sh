#!/bin/bash
#SBATCH --time=09:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=6GB
#SBATCH --job-name=llama_finetune
#SBATCH --partition=gpu

source $HOME/venvs/thesis/bin/activate

python llama.py
python inference.py