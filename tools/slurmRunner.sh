#!/bin/sh
set -eu

if [ $# -ne 1 ]; then
  echo "Usage: $0 <id>" >&2
  exit 1
fi

ID="$1"
TRAIN_DATASET="sav5k"
ALPHA="02"

# # run temp save checkpoint before training at the beginning of run_train
# NF=16
# PF=15
# MEMTYPE="singleMem"
# EXP_STR="sam2_train_${NF}nf_${PF}pf_${TRAIN_DATASET}"

# RUN_TAG="${EXP_STR}_${MEMTYPE}_train_frzall_eval_full"
# CKPT_PATH="/p/lava/morteza/sam2/log/sam2_train_${NF}nf_${PF}pf_frzall/checkpoints/checkpoint_${ID}.pt"
# CONFIG="configs/sam2.1/sam2.1_hiera_b+_${PF}pf.yaml"

# if [ ! -f "$CKPT_PATH" ]; then
#   echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
#   exit 2
# fi

# if [ ! -f "/u/rgq5aw/GIT/sam2_orig/sam2/$CONFIG" ]; then
#   echo "ERROR: config not found: $CONFIG" >&2
#   exit 2
# fi

# sbatch --job-name="${RUN_TAG}_e${ID}" eval.sh "$CONFIG" "$CKPT_PATH"


# NF=28
# PF=27
# MEMTYPE="singleMem"
# EXP_STR="sam2_train_${NF}nf_${PF}pf_${TRAIN_DATASET}"

# RUN_TAG="${EXP_STR}_${MEMTYPE}_train_frzall_eval_full"
# CKPT_PATH="/p/lava/morteza/sam2/log/sam2_train_${NF}nf_${PF}pf_frzall/checkpoints/checkpoint_${ID}.pt"
# CONFIG="configs/sam2.1/sam2.1_hiera_b+_${PF}pf.yaml"

# if [ ! -f "$CKPT_PATH" ]; then
#   echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
#   exit 2
# fi

# if [ ! -f "/u/rgq5aw/GIT/sam2_orig/sam2/$CONFIG" ]; then
#   echo "ERROR: config not found: $CONFIG" >&2
#   exit 2
# fi

# sbatch --job-name="${RUN_TAG}_e${ID}" eval.sh "$CONFIG" "$CKPT_PATH"

NF=28
PF=27
MEMTYPE="singleMem"
EXP_STR="sam2_train_${TRAIN_DATASET}_${NF}nf_${PF}pf"

RUN_TAG="${EXP_STR}_${MEMTYPE}_train_frzall_frz7frm_eval_full_new_1"
CKPT_PATH="/p/lava/morteza/sam2/log/sam2_train_${TRAIN_DATASET}_${NF}nf_${PF}pf_frzall_frz7frm_new/checkpoints/checkpoint_${ID}.pt"
CONFIG="configs/sam2.1/sam2.1_hiera_b+_${PF}pf.yaml"

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
  exit 2
fi

if [ ! -f "/u/rgq5aw/GIT/sam2_orig/sam2/$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

sbatch --job-name="${RUN_TAG}_e${ID}" eval.sh "$CONFIG" "$CKPT_PATH"