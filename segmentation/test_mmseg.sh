#!/usr/bin/env bash

if [ "$#" -lt 5 ]; then
    echo "Usage: me.sh Relative_config_filename Checkpoint_filename gpus nb_gpus port [others]"
    exit
fi

PYTHON=${PYTHON:-"python"}

CONFIG_FILE=$1
CHK_FILE=$2
GPUS=$3
NUM_GPUS=$4
PORT=${PORT:-$5}


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CONFIG_FILENAME=${CONFIG_FILE##*/}
CONFIG_BASE="${CONFIG_FILENAME%.*}"

WORK_DIR="$( cd "$( dirname "${CHK_FILE}" )" >/dev/null 2>&1 && pwd )"/$CONFIG_BASE

if [ -d $WORK_DIR ]; then 
  echo "... Done already!"
  exit 
fi 

# export NCCL_DEBUG=INFO

TORCH_DISTRIBUTED_DEBUG=INFO OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS \
  torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:$PORT \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    $DIR/test_mmseg.py \
    $CONFIG_FILE \
    $CHK_FILE \
    --launcher pytorch \
    --work-dir $WORK_DIR \
    ${@:6}
