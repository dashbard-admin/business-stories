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

# Detach python with nohup + & so the orchestrator survives the
# calling shell exiting. This matches the user's manual invocation
#     nohup python3 -m pipeline.hermes_orchestrator 2>&1 &
# which is known to work: the file lock at state/locks/orchestrator.lock
# correctly serialises concurrent invocations, so when an external
# scheduler (cron, launchd, an AI agent's shell-execution tool) fires
# this script every N seconds, the first call holds the lock and runs
# a full stage uninterrupted; subsequent calls detect the lock, log
# "lock contention", and exit immediately without touching the
# running orchestrator.
#
# Without the & and nohup the python process would be a synchronous
# child of the calling shell. When the shell is killed (e.g. an AI
# agent's per-invocation timeout fires after 2-10 minutes), the
# SIGTERM propagates to the python child and kills it mid-stage,
# losing all work for that stage.
#
# stdin redirected from /dev/null so an interactive launcher doesn't
# accidentally hand the orchestrator a tty.
nohup python3 -m pipeline.hermes_orchestrator "$@" \
    </dev/null >>"${LOGFILE}" 2>&1 &
echo "orchestrator pid=$! → ${LOGFILE}"
