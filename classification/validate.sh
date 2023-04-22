#!/usr/bin/env bash

if [ "$#" -lt 6 ]; then
    echo "Usage: me model_name checkpoint_file dataset_name img_size gpus num_gpus [others]"
    exit
fi

MODEL=$1
CHECKPOINT_FILE=$2
DATASET=$3
IMAGE_SIZE=$4
GPUS=$5
NUM_GPUS=$6

PYTHON=${PYTHON:-"python"}

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


CUDA_VISIBLE_DEVICES=$GPUS $PYTHON \
  $DIR/validate_timm.py  $DATA_DIR --dataset $DATASET \
  --img-size $IMAGE_SIZE --workers 8 --num-gpu $NUM_GPUS \
  --model $MODEL --checkpoint $CHECKPOINT_FILE --pin-mem --channels-last --amp \
  ${@:7}





