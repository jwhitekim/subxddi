#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE="${1:-core}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
PID_FILE="${OUTPUT_DIR}/background_${PHASE}.pid"
LOG_FILE="${OUTPUT_DIR}/logs/background_${PHASE}.log"

if [ -f "${PID_FILE}" ]; then
  PID="$(cat "${PID_FILE}")"
  STATE="$(ps -o stat= -p "${PID}" 2>/dev/null || true)"
  if [ -n "${STATE}" ] && [[ "${STATE}" != *Z* ]]; then
    echo "[BACKGROUND_STATUS] phase=${PHASE} status=running pid=${PID}"
  elif [ -n "${STATE}" ] && [[ "${STATE}" == *Z* ]]; then
    echo "[BACKGROUND_STATUS] phase=${PHASE} status=zombie pid=${PID}"
  else
    echo "[BACKGROUND_STATUS] phase=${PHASE} status=stopped pid=${PID}"
  fi
else
  echo "[BACKGROUND_STATUS] phase=${PHASE} status=no_pid"
fi

RESULTS_CSV="${OUTPUT_DIR}/all_results.csv"
FAILED_CSV="${OUTPUT_DIR}/failed_runs.csv"
if [ -f "${RESULTS_CSV}" ]; then
  COMPLETED="$(python - "${RESULTS_CSV}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as f:
    print(sum(1 for row in csv.DictReader(f) if row.get("status") == "completed"))
PY
)"
  echo "[RESULT_COUNTS] completed=${COMPLETED} results=${RESULTS_CSV}"
fi
if [ -f "${FAILED_CSV}" ]; then
  FAILED="$(awk 'NR>1{c++} END{print c+0}' "${FAILED_CSV}")"
  echo "[FAILED_COUNTS] failed=${FAILED} failed_csv=${FAILED_CSV}"
fi

if [ -f "${LOG_FILE}" ]; then
  echo "[LOG_TAIL] file=${LOG_FILE}"
  tail -n 50 "${LOG_FILE}"
else
  echo "[LOG_TAIL] file=${LOG_FILE} missing"
fi
