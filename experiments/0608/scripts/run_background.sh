#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE="${1:-core}"
shift || true
EXTRA_ARGS=("$@")
OUTPUT_DIR="${OUTPUT_DIR:-.}"
mkdir -p "${OUTPUT_DIR}/logs"

PID_FILE="${OUTPUT_DIR}/background_${PHASE}.pid"
LOG_FILE="${OUTPUT_DIR}/logs/background_${PHASE}.log"

if [ -f "${PID_FILE}" ]; then
  OLD_PID="$(cat "${PID_FILE}")"
  OLD_STATE="$(ps -o stat= -p "${OLD_PID}" 2>/dev/null || true)"
  if [ -n "${OLD_STATE}" ] && [[ "${OLD_STATE}" != *Z* ]]; then
    echo "[BACKGROUND_ALREADY_RUNNING] phase=${PHASE} pid=${OLD_PID} log=${LOG_FILE}"
    exit 0
  fi
fi

setsid python -u scripts/run_0608_experiments.py \
  --phase "${PHASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "[BACKGROUND_STARTED] phase=${PHASE} pid=${PID} log=${LOG_FILE} args=${EXTRA_ARGS[*]}"
