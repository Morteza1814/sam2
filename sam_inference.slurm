#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a40"
#SBATCH --job-name=sam_inference
#SBATCH --mem=300000
#SBATCH --time=10:20:00
#SBATCH --output=sam_inference.txt
#SBATCH --error=sam_inference.txt

# module load gcccore/12.2.0
# module load python/3.10.8
#module load cuda


python3 ./tools/vos_inference.py   --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml   --sam2_checkpoint ./sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml/checkpoints/checkpoint.pt   --base_video_dir /bigtemp/rgq5aw/samData/sav/sav_val/JPEGImages_24fps   --input_mask_dir /bigtemp/rgq5aw/samData/sav/sav_val/Annotations_6fps   --per_obj_png_file  --video_list_file /bigtemp/rgq5aw/samData/sav/sav_val/sav_val.txt    --output_mask_dir ./outputs/sav_val_pred_pngs