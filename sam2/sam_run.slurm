#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a40|a100_40gb"
#SBATCH --job-name=sam_run
#SBATCH --mem=30000
#SBATCH --time=20:20:00
#SBATCH --output=sam_run.txt
#SBATCH --error=sam_run.txt

module load gcccore/12.2.0
module load python/3.10.8
#module load cuda

export PYTHONPATH=/path/to/custom_directory:$PYTHONPATH

python sam_run.py