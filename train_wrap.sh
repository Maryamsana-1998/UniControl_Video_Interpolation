#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -o experiments/uni_dir_wrap/slurm.out
#SBATCH -e experiments/uni_dir_wrap/slurm.err

# Set up directories
EXPERIMENT_DIR="experiments/uni_dir_wrap"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"

mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters

# training with data aug + lpips + uni pre trained ckpt + wraping + uni directional decoding 
CONFIG_PATH="configs/uni_wrap/local_v15.yaml"
INIT_CKPT="ckpt/uni_wrap/init_local_uni.ckpt"
NUM_GPUS=6
BATCH_SIZE=1
NUM_WORKERS=16
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
