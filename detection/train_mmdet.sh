#!/usr/bin/env bash

if [ "$#" -lt 7 ]; then
    echo "Usage: me.sh Relative_config_filename Remove_old_if_exist_0_or_1 Exp_name Tag gpus nb_gpus port [others]"
    exit
fi

PYTHON=${PYTHON:-"python"}

CONFIG_FILE=$1
RM_OLD=$2
EXP_NAME=$3
TAG=$4
GPUS=$5
NUM_GPUS=$6
PORT=${PORT:-$7}


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CONFIG_FILENAME=${CONFIG_FILE##*/}
CONFIG_BASE="${CONFIG_FILENAME%.*}"

WORK_DIR=${DIR}/../work_dirs/detection/${EXP_NAME}/${CONFIG_BASE}_$TAG

if [ -d $WORK_DIR ]; then
  echo "$WORK_DIR --- Already exists"
  if [ $2 -gt 0 ]; then
    while true; do
        read -p "Are you sure to delete this result directory? " yn
        case $yn in
            [Yy]* ) rm -r $WORK_DIR; mkdir -p $WORK_DIR; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
  else
    echo "Resume"
  fi
else
    mkdir -p $WORK_DIR
fi

# export NCCL_DEBUG=INFO

TORCH_DISTRIBUTED_DEBUG=INFO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS \
  torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:$PORT \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    $DIR/train_mmdet.py $CONFIG_FILE \
    --amp \
    --resume "auto" \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    --auto-scale-lr \
    ${@:8}
