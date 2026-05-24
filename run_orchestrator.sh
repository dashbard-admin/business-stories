#!/usr/bin/env bash
# Cron-friendly wrapper. Each invocation runs one stage of one episode
# and exits. Stale-lock reclamation lives in pipeline/state.py so a
# crashed prior run won't wedge the queue.

set -euo pipefail

ROOT="/Users/cantemir/Projects/business_success_stories"
cd "${ROOT}"

# Activate venv if present. Fall back to system python3.
if [[ -d "${ROOT}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

export PIPELINE_CONFIG="${ROOT}/config.yaml"

LOGDIR="${ROOT}/logs"
mkdir -p "${LOGDIR}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOGFILE="${LOGDIR}/run.${TS}.log"

python3 -m pipeline.hermes_orchestrator "$@" >>"${LOGFILE}" 2>&1
