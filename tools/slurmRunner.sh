#!/bin/sh
set -eu

if [ $# -ne 1 ]; then
  echo "Usage: $0 <id>" >&2
  exit 1
fi

ID="$1"
TRAIN_DATASET="1ksav"
ALPHA="1"

# run nf=8 pf=7 dual = False check logdir
NF=8
PF=7
MEMTYPE="singleMem"
EXP_STR="sam2_train_${NF}nf_${PF}pf_${TRAIN_DATASET}"

RUN_TAG="${EXP_STR}_${MEMTYPE}_fullEval"
CKPT_DIR="/p/lava/morteza/sam2/log/${EXP_STR}_${MEMTYPE}/checkpoints"
CKPT_PATH="${CKPT_DIR}/checkpoint_${ID}.pt"
CONFIG="configs/sam2.1/sam2.1_hiera_b+_${PF}pf_${MEMTYPE}.yaml"

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
  exit 2
fi

if [ ! -f "/u/rgq5aw/GIT/sam2/sam2/$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

sbatch --job-name="${RUN_TAG}_e${ID}" eval.sh "$CONFIG" "$CKPT_PATH"

# run nf=16 pf=15 dual = False check logdir
NF=16
PF=15
MEMTYPE="singleMem"
EXP_STR="sam2_train_${NF}nf_${PF}pf_${TRAIN_DATASET}"

RUN_TAG="${EXP_STR}_${MEMTYPE}_fullEval"
CKPT_DIR="/p/lava/morteza/sam2/log/${EXP_STR}_${MEMTYPE}/checkpoints"
CKPT_PATH="${CKPT_DIR}/checkpoint_${ID}.pt"
CONFIG="configs/sam2.1/sam2.1_hiera_b+_${PF}pf_${MEMTYPE}.yaml"

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
  exit 2
fi

if [ ! -f "/u/rgq5aw/GIT/sam2/sam2/$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

sbatch --job-name="${RUN_TAG}_e${ID}" eval.sh "$CONFIG" "$CKPT_PATH"

# run nf=16 pf=15 dual = True Freeze Alpha check logdir
NF=16
PF=15
MEMTYPE="dualMem"
EXP_STR="sam2_train_${NF}nf_${PF}pf_${TRAIN_DATASET}"

RUN_TAG="${EXP_STR}_${MEMTYPE}_alpha${ALPHA}frozen_fullEval"
CKPT_DIR="/p/lava/morteza/sam2/log/${EXP_STR}_${MEMTYPE}_alpha${ALPHA}frozen/checkpoints"
CKPT_PATH="${CKPT_DIR}/checkpoint_${ID}.pt"
CONFIG="configs/sam2.1/sam2.1_hiera_b+_${PF}pf_${MEMTYPE}.yaml"

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
  exit 2
fi

if [ ! -f "/u/rgq5aw/GIT/sam2/sam2/$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

sbatch --job-name="${RUN_TAG}_e${ID}" eval.sh "$CONFIG" "$CKPT_PATH"

# run nf=16 pf=15 dual = True Trainable Alpha check logdir
NF=16
PF=15
MEMTYPE="dualMem"
EXP_STR="sam2_train_${NF}nf_${PF}pf_${TRAIN_DATASET}"

RUN_TAG="${EXP_STR}_${MEMTYPE}_alpha${ALPHA}_fullEval"
CKPT_DIR="/p/lava/morteza/sam2/log/${EXP_STR}_${MEMTYPE}_alpha${ALPHA}/checkpoints"
CKPT_PATH="${CKPT_DIR}/checkpoint_${ID}.pt"
CONFIG="configs/sam2.1/sam2.1_hiera_b+_${PF}pf_${MEMTYPE}.yaml"

if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CKPT_PATH" >&2
  exit 2
fi

if [ ! -f "/u/rgq5aw/GIT/sam2/sam2/$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

sbatch --job-name="${RUN_TAG}_e${ID}" eval.sh "$CONFIG" "$CKPT_PATH"