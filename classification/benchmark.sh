#!/usr/bin/env bash

if [ "$#" -lt 4 ]; then
    echo "Usage: me model_name img_size num_classes gpu [others]"
    exit
fi

MODEL_NAME=$1
IMAGE_SIZE=$2
NUM_CLASSES=$3
GPU=$4


PYTHON=${PYTHON:-"python"}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

RESULT_FILE=$DIR/../work_dirs/classification/all_benchmark_results.csv

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
  $DIR/benchmark_timm.py  --results-file $RESULT_FILE \
  --model $MODEL_NAME --bench inference \
  --num-bench-iter 100 \
  --batch-size 128 --img-size $IMAGE_SIZE --num-classes $NUM_CLASSES \
  --opt adamw --opt-eps 1e-8 --momentum 0.9 --weight-decay 0.05 \
  --smoothing 0.1 --drop-path 0.1 \
  --amp --channels-last \
  ${@:5}
#   --clip-grad 1.0 --clip-mode norm
