#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v8
#SBATCH -o experiments/uni_interpolation/slurm.out
#SBATCH -e experiments/uni_interpolation/slurm.err

# Set up directories
EXPERIMENT_DIR="experiments/uni_interpolation"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"

mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="configs/uni_interpolation/local_v15.yaml"
INIT_CKPT="ckpt/uni_dir_interpolation/init_local_uni.ckpt"
NUM_GPUS=8
BATCH_SIZE=1
NUM_WORKERS=32
MAX_STEPS=100000

# lpips + data aug + uni pretrained weights + uni directional decoding 

python src/train/train.py \
    --config-path ${CONFIG_PATH} \
    ---resume-path ${INIT_CKPT} \
    ---gpus ${NUM_GPUS} \
    ---batch-size ${BATCH_SIZE} \
    ---logdir ${LOGS_DIR} \
    ---checkpoint-dirpath ${LOCAL_CKPT_DIR} \
    ---training-steps ${MAX_STEPS} \
    ---num-workers ${NUM_WORKERS}
