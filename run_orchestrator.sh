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

# Single daily rolling log. Cleaner than one file per invocation
# (which produces hundreds of tiny files under a 5-minute launchd /
# cron schedule). pipeline/hermes_orchestrator.py also writes its
# own structured log at logs/orch.YYYY-MM-DD.log via _setup_logging,
# so this file captures only the wrapper-level glue + any stray
# subprocess stderr.
LOGFILE="${LOGDIR}/run.$(date -u +%Y-%m-%d).log"

python3 -m pipeline.hermes_orchestrator "$@" >>"${LOGFILE}" 2>&1
