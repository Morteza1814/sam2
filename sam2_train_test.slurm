#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=sam2_train_sav5k_16nf_15pf_frzall_frz7frm_loadchckpntcp
#SBATCH --constraint=h100_94gb|a100_80gb
#SBATCH --mem=100000
#SBATCH --time=4-00:00:00
#SBATCH --output=sam2_train_sav5k_16nf_15pf_frzall_frz7frm_loadchckpntcp.txt
#SBATCH --error=sam2_train_sav5k_16nf_15pf_frzall_frz7frm_loadchckpntcp.txt

# module load gcccore/12.2.0
# module load python/3.10.8
# module load cuda

# source ~/envs/sam2env/bin/activate
#export PYTHONPATH=/u/rgq5aw/GIT/sam2:$PYTHONPATH

echo "GPU info for this job:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv


python3 /u/rgq5aw/GIT/sam2/training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_SAV_finetune_sam2_16nf_15pf_singleMem.yaml \
    --use-cluster 0 \
    --num-gpus 1