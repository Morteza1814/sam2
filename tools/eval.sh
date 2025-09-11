#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="h100_94gb|a100_80gb|a100_40gb|a40"
#SBATCH --mem=100000
#SBATCH --time=20:00:00
#SBATCH --output=%x.txt        # expands to job name
#SBATCH --error=%x.txt         # expands to job name

echo "GPU info for this job:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

source ~/envs/sam2origenv/bin/activate


if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch --job-name=<jobname> $0 <config_file_path> <checkpoint_path>"
    exit 1
fi

CONFIG_FILE="$1"
CHECKPOINT_PATH="$2"

TMP_DIR="/bigtemp2/rgq5aw/tmp/$SLURM_JOB_NAME"

mkdir -p "$TMP_DIR"
find "$TMP_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

echo "config_file: /u/rgq5aw/GIT/sam2_orig/sam2/$CONFIG_FILE"
echo "checkpoint_path: $CHECKPOINT_PATH"
echo "====================================config file===================================="
cat "/u/rgq5aw/GIT/sam2_orig/sam2/$CONFIG_FILE"
echo "==================================================================================="
# Run inference
python3 vos_inference.py \
    --sam2_cfg "$CONFIG_FILE" \
    --sam2_checkpoint "$CHECKPOINT_PATH" \
    --base_video_dir /p/lava/morteza/sam2/datasets/lvos/val/JPEGImages \
    --input_mask_dir /p/lava/morteza/sam2/datasets/lvos/val/Annotations \
    --output_mask_dir "$TMP_DIR" \
    --track_object_appearing_later_in_video

echo "Inference completed. Now running evaluation..."

cd /u/rgq5aw/GIT/lvos-evaluation/

python3 evaluation_method.py \
    --task semi-supervised \
    --results_path "$TMP_DIR" \
    --mp_nums 32 \
    --m_class mp \
    --use_cache \
    --lvos_path /p/lava/morteza/sam2/datasets/lvos/val/

rm -rf "$TMP_DIR"
