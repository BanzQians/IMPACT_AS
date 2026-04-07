#!/usr/bin/env bash
set -euo pipefail

# ====== 基本配置（可按需修改）======
ROOT="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "${ROOT}/.." && pwd)"        # 数据集根目录，默认指向 external/
DATASET="learningcell_front"            # 数据集文件夹名（位于 ${BASE}/${DATASET}）
GPU=0
SEED=0

# 训练超参
EPOCHS=30
LR=5e-4
WD=5e-4
BATCH=2
LAYER_SIZES="2048 128 40"   # 显式指定输入维度，避免默认 64
N_CLUSTERS=62      # 聚类/动作数，按 mapping 行数设置，酌情调整
CKPT_EVERY=1       # 每多少个 epoch 存一次 ckpt（1=每个 epoch）

# 日志
RUN_TAG="learningcell_front_asot"
LOG_DIR="logs_learningcell_$(date +%F_%H%M)"
mkdir -p "${LOG_DIR}"

echo "[RUN] base_path=${BASE} dataset=${DATASET} gpu=${GPU} seed=${SEED}"
# 运行前检查必须文件/目录
test -d "${BASE}/${DATASET}/groundTruth" || { echo "ERR: groundTruth missing"; exit 2; }
test -d "${BASE}/${DATASET}/features"    || { echo "ERR: features missing";    exit 2; }
test -f "${BASE}/${DATASET}/mapping/mapping.txt" || { echo "ERR: mapping/mapping.txt missing"; exit 2; }
test -f "${BASE}/${DATASET}/splits/train.split1.train" || { echo "ERR: splits train missing"; exit 2; }

act="all"   # 使用默认 all，不对 activity 做筛选
clu="${N_CLUSTERS}"
ts="$(date +%F_%H%M%S)"
log="${LOG_DIR}/train_${act}_${ts}.log"

echo "==> Training activity: ${act} (clusters=${clu})"
echo "    log: ${log}"

CUDA_VISIBLE_DEVICES=${GPU} WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
python3 src/train.py \
  -p "${BASE}" \
  -d "${DATASET}" \
  -ac "${act}" \
  -c "${clu}" \
  -ne "${EPOCHS}" -bs "${BATCH}" -g "${GPU}" \
  -s --rho 0.2 -lat 0.1 -r 0.04 -ae 0.7 -at 0.4 \
  -ls ${LAYER_SIZES} \
  --ckpt-every "${CKPT_EVERY}" \
  -lr "${LR}" -wd "${WD}" -vf 1 \
  --group "${RUN_TAG}" --wandb -v -ua \
  --seed "${SEED}" \
  2>&1 | tee -a "${log}"

echo "[DONE] log saved: ${log}"
