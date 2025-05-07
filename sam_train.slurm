#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a100_40gb|a40"
#SBATCH --job-name=sam_train
#SBATCH --mem=30000
#SBATCH --time=10:20:00
#SBATCH --output=sam_train.txt
#SBATCH --error=sam_train.txt

# module load gcccore/12.2.0
# module load python/3.10.8
#module load cuda


python3 training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 1