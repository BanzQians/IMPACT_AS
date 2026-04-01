#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.siglip2.yml"
ENV_ROOT="${CONDA_ENV_ROOT:-${HOME}/IsaacDrive/conda_envs}"
ENV_PREFIX="${SIGLIP2_CONDA_PREFIX:-${ENV_ROOT}/siglip2}"
MODEL_ID="${SIGLIP2_MODEL_ID:-google/siglip2-base-patch16-224}"
MODEL_DIR="${SIGLIP2_MODEL_DIR:-${ROOT_DIR}/external/huggingface/google--siglip2-base-patch16-224}"
CONDA_EXE="${CONDA_EXE:-conda}"

echo "[SIGLIP2] repo root: ${ROOT_DIR}"
echo "[SIGLIP2] env prefix: ${ENV_PREFIX}"
echo "[SIGLIP2] model id: ${MODEL_ID}"
echo "[SIGLIP2] model dir: ${MODEL_DIR}"

mkdir -p "${ENV_ROOT}"

if [[ -d "${ENV_PREFIX}" ]]; then
  "${CONDA_EXE}" env update -p "${ENV_PREFIX}" -f "${ENV_FILE}" --prune
else
  "${CONDA_EXE}" env create -p "${ENV_PREFIX}" -f "${ENV_FILE}"
fi

"${ENV_PREFIX}/bin/python" "${ROOT_DIR}/tools/download_siglip2_model.py" \
  --model-id "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}"

echo "[SIGLIP2] setup complete"
