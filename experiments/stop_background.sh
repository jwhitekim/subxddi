#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE="${1:-core}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
PID_FILE="${OUTPUT_DIR}/background_${PHASE}.pid"

if [ ! -f "${PID_FILE}" ]; then
  echo "[BACKGROUND_STOP] phase=${PHASE} status=no_pid"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
STATE="$(ps -o stat= -p "${PID}" 2>/dev/null || true)"
if [ -n "${STATE}" ] && [[ "${STATE}" != *Z* ]]; then
  kill "${PID}"
  sleep 2
  if ps -p "${PID}" > /dev/null 2>&1; then
    kill -TERM "${PID}" || true
  fi
  echo "[BACKGROUND_STOPPED] phase=${PHASE} pid=${PID}"
elif [ -n "${STATE}" ] && [[ "${STATE}" == *Z* ]]; then
  echo "[BACKGROUND_STOP] phase=${PHASE} status=zombie pid=${PID}"
else
  echo "[BACKGROUND_STOP] phase=${PHASE} status=already_stopped pid=${PID}"
fi
