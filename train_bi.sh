#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -o experiments/bi_dir_interpolation/slurm.out
#SBATCH -e experiments/bi_dir_interpolation/slurm.err

# Set up directories
EXPERIMENT_DIR="experiments/bi_dir_interpolation"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"

mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="configs/bi_interpolation/local_v15.yaml"
INIT_CKPT="ckpt/bi_dir_interpolation/init_local.ckpt"
NUM_GPUS=8
BATCH_SIZE=3
NUM_WORKERS=32
MAX_STEPS=100000


python src/train/train.py \
    --config-path ${CONFIG_PATH} \
    ---resume-path ${INIT_CKPT} \
    ---gpus ${NUM_GPUS} \
    ---batch-size ${BATCH_SIZE} \
    ---logdir ${LOGS_DIR} \
    ---checkpoint-dirpath ${LOCAL_CKPT_DIR} \
    ---training-steps ${MAX_STEPS} \
    ---num-workers ${NUM_WORKERS}
