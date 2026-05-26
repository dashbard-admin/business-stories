#!/usr/bin/env bash
# Cron-friendly wrapper. Each invocation runs one stage of one episode
# and exits. Stale-lock reclamation lives in pipeline/state.py so a
# crashed prior run won't wedge the queue.

set -euo pipefail

# Derive the project root from the location of THIS script — portable
# across machines and checkout locations. Resolve symlinks so `cron`
# can invoke this via any path.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
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
