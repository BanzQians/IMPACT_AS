#!/usr/bin/env bash
set -euo pipefail

# ====== 基本配置（改这里）======
BASE="/cvhci/temp/qiany"       # 我们把数据放在 $BASE/Breakfast/...
DATASET="Breakfast"            # 保持 Breakfast，让解析走 Breakfast 分支
GPU=0
SEED=0

# 训练轮数 & 优化器
EPOCHS=80        # 先冒烟，确认能跑；稳定后再增大
LR=5e-4
WD=5e-4
BATCH=2

# 分组 & 日志
RUN_TAG="custom_quicktest_resnet"
LOG_DIR="logs_custom_$(date +%F_%H%M)"
mkdir -p "${LOG_DIR}"

# 只训练我们伪装出来的 activity = custom
actions=(custom)
clusters=(7)    # 聚类数随便给一个小值（如 5~12）；后面可以再调

echo "[RUN] base_path=${BASE} dataset=${DATASET} gpu=${GPU} seed=${SEED}"
# 运行前硬检查
test -d "${BASE}/${DATASET}/groundTruth" || { echo "ERR: groundTruth missing"; exit 2; }
test -d "${BASE}/${DATASET}/features"    || { echo "ERR: features missing";    exit 2; }
test -f "${BASE}/${DATASET}/mapping/mapping.txt" || { echo "ERR: mapping/mapping.txt missing"; exit 2; }
test -f "${BASE}/${DATASET}/splits/train.split1.train" || { echo "ERR: splits train missing"; exit 2; }

for i in "${!actions[@]}"; do
  act="${actions[$i]}"
  clu="${clusters[$i]}"
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
    -lr "${LR}" -wd "${WD}" -vf 1 \
    --group "${RUN_TAG}" --wandb -v -ua \
    --seed "${SEED}" \
    2>&1 | tee -a "${log}"
done
