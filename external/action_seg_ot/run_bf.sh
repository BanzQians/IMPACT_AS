#!/usr/bin/env bash
set -euo pipefail

# ====== 基本配置 ======
DATA_ROOT="/cvhci/data/activity/Action_Segmentation_Datasets"
DATASET="Breakfast"
GPU=0
SEED=0

# 训练轮数 & 优化器
EPOCHS=80           # ← 从 15 提到 80（先跑满一个较长的收敛周期）
LR=5e-4             # ← 稍保守一点，有利于后期稳定
WD=5e-4             # ← 稍强一点的正则，抑制过拟合/抖动
BATCH=2

# 分组 & 日志
RUN_TAG="main_results_longrun"
LOG_DIR="logs_${DATASET}_$(date +%F_%H%M)"
mkdir -p "${LOG_DIR}"

# Breakfast 的 10 个活动及对应的聚类数
actions=(pancake salat friedegg scrambledegg sandwich juice milk tea cereals coffee)
clusters=(14      8     9        12           9         8     5    7   5       7)

echo "[RUN] base_path=${DATA_ROOT} dataset=${DATASET} gpu=${GPU} seed=${SEED}"
# 运行前硬检查
test -d "${DATA_ROOT}/${DATASET}/groundTruth" || { echo "ERR: groundTruth missing"; exit 2; }
test -d "${DATA_ROOT}/${DATASET}/features"    || { echo "ERR: features missing";    exit 2; }

for i in "${!actions[@]}"; do
  act="${actions[$i]}"
  clu="${clusters[$i]}"
  ts="$(date +%F_%H%M%S)"
  log="${LOG_DIR}/train_${act}_${ts}.log"

  echo "==> Training activity: ${act} (clusters=${clu})"
  echo "    log: ${log}"

  CUDA_VISIBLE_DEVICES=${GPU} WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
  python3 src/train.py \
    -p "${DATA_ROOT}" \
    -d "${DATASET}" \
    -ac "${act}" \
    -c "${clu}" \
    -ne "${EPOCHS}" -bs "${BATCH}" -g "${GPU}" \
    -s --rho 0.2 -lat 0.1 -r 0.04 -ae 0.7 -at 0.4 \
    -lr "${LR}" -wd "${WD}" -vf 5 \
    --group "${RUN_TAG}" --wandb -v -ua \
    --seed "${SEED}" \
    2>&1 | tee -a "${log}"

  # ====== （可选）如果 src/train.py 支持这些参数，就去掉下面的注释启用 ======
  # --num_workers 16 --pin_memory --persistent_workers \
  # --scheduler cosine \
  # --early_stop --patience 12 --monitor val_mof_full \
  # --class_balance log \
  # --smooth_coef 0.2 --consistency_coef 0.1 \
done
