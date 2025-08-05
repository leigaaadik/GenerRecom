#!/bin/bash

# 设置环境变量
export RUNTIME_SCRIPT_DIR="."

export TRAIN_DATA_PATH="${RUNTIME_SCRIPT_DIR}/TencentGR_1k"
export TRAIN_LOG_PATH="${RUNTIME_SCRIPT_DIR}/logs"
export TRAIN_TF_EVENTS_PATH="${RUNTIME_SCRIPT_DIR}/tensorboard"
export TRAIN_CKPT_PATH="${RUNTIME_SCRIPT_DIR}/checkpoints"

# # show ${RUNTIME_SCRIPT_DIR}
# echo "当前脚本目录: ${RUNTIME_SCRIPT_DIR}"
# # enter train workspace
# cd "${RUNTIME_SCRIPT_DIR}"

# write your code below
python main.py