#!/usr/bin/env bash

if [ "$#" -lt 12 ]; then
    echo "Usage: me Config_file Model_name Dataset_name Img_size Remove_old_if_exist_0_or_1 Resume_or_not_if_exist Exp_name Tag Gpus Nb_gpus Workers Port [others]"
    exit
fi

CONFIG_FILE=$1
MODEL=$2
DATASET=$3
DATA_SIZE=$4
RM_OLD_IF_EXIST=$5
RESUM_OLD_IF_EXIST=$6
EXP_NAME=$7
TAG=$8
GPUS=$9
NUM_GPUS=${10}
WORKERS=${11}
PORT=${12}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# datasets 
NUM_CLASSES=0
if [ "$DATASET" = "IMNET" ]; then
  DATA_DIR=$DIR/../datasets/IMNET/
  if [ ! -d $DATA_DIR ]; then
    echo "not found $DATA_DIR"
    exit
  fi
  NUM_CLASSES=1000
else
  echo "Unknown $DATASET"
  exit 
fi

# dirs 
WORK_DIR=$DIR/../work_dirs/classification/$EXP_NAME

EXPERIMET="$DATASET"_"$DATA_SIZE"_"$MODEL"_"$TAG"

# training has completed?
EXPERIMENT_DIR=$WORK_DIR/TrainingFinished/$EXPERIMET
if [ -d $EXPERIMENT_DIR ]; then
  echo "$EXPERIMENT_DIR --- Training Finished!!!!"
  exit 
fi

EXPERIMENT_DIR=$WORK_DIR/$EXPERIMET
if [ -d $EXPERIMENT_DIR ]; then
  echo "$EXPERIMENT_DIR --- Already exists"
  if [ $RM_OLD_IF_EXIST -gt 0 ]; then
    while true; do
        read -p "Are you sure to delete this result directory? " yn
        case $yn in
            [Yy]* ) rm -r $EXPERIMENT_DIR; mkdir -p $EXPERIMENT_DIR; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
  else
    if [ $RESUM_OLD_IF_EXIST -gt 0 ]; then
      echo "Auto-resume"
    else
      echo "Skip"
      exit
    fi
  fi
fi

# TORCH_DISTRIBUTED_DEBUG=INFO \
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS \
  torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:$PORT \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    $DIR/train_timm.py  \
    --data-dir $DATA_DIR \
    --img-size $DATA_SIZE \
    --num-classes $NUM_CLASSES \
    --config $CONFIG_FILE \
    --model $MODEL \
    --workers $WORKERS \
    --channels-last \
    --pin-mem \
    --use-multi-epochs-loader \
    --output $WORK_DIR \
    --experiment $EXPERIMET \
    ${@:13}

    



