#!/usr/bin/env bash
# Run the current/next episode continuously until S12 produces final.mp4.
# Any `needs_human` gate on the selected episode is approved automatically.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT}"

if [[ -d "${ROOT}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

export PIPELINE_CONFIG="${ROOT}/config.yaml"

LOGDIR="${ROOT}/logs"
mkdir -p "${LOGDIR}"
TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOGFILE="${LOGDIR}/full_auto.${TS}.log"
exec > >(tee -a "${LOGFILE}") 2>&1

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-80}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
EP_ID="${1:-}"

echo "full-auto runner log: ${LOGFILE}"

queue_state() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from pipeline.state import find_episode_workspace, load_queue

want = sys.argv[1].strip() or None
queue = load_queue()
episodes = queue.get("episodes") or []

if want:
    ep = next((item for item in episodes if item.get("id") == want), None)
else:
    ep = next((item for item in episodes if item.get("current_stage") != "DONE"), None)

if ep is None:
    print("NONE\t\tfalse\tfalse\t\t0")
    raise SystemExit(0)

workspace = find_episode_workspace(ep["id"])
final_path = workspace / "05_video" / "final.mp4" if workspace else None
final_exists = bool(final_path and final_path.exists() and final_path.stat().st_size > 0)
blocker_count = len(ep.get("blockers") or [])
has_needs_human = any(
    (stage or {}).get("status") == "needs_human"
    for stage in (ep.get("stages") or {}).values()
)

print("\t".join([
    ep["id"],
    str(ep.get("current_stage") or ""),
    "true" if blocker_count or has_needs_human else "false",
    "true" if final_exists else "false",
    str(final_path or ""),
    str(blocker_count),
]))
PY
}

next_runnable_id() {
  "${PYTHON_BIN}" - <<'PY'
from pipeline.state import load_queue, next_pending_episode
queue = load_queue()
pick = next_pending_episode(queue)
print(pick[0]["id"] if pick else "")
PY
}

if [[ -z "${EP_ID}" ]]; then
  read -r EP_ID _stage _blocked _final _path _count < <(queue_state "")
  if [[ "${EP_ID}" == "NONE" ]]; then
    echo "No queued episode found; enqueueing one episode."
    "${PYTHON_BIN}" -m pipeline.hermes_orchestrator --enqueue 1
    read -r EP_ID _stage _blocked _final _path _count < <(queue_state "")
  fi
fi

if [[ -z "${EP_ID}" || "${EP_ID}" == "NONE" ]]; then
  echo "Could not select an episode to run."
  exit 2
fi

echo "selected episode: ${EP_ID}"

for ((iteration = 1; iteration <= MAX_ITERATIONS; iteration++)); do
  read -r current_ep current_stage blocked final_exists final_path blocker_count < <(queue_state "${EP_ID}")

  if [[ "${current_ep}" == "NONE" ]]; then
    echo "Episode ${EP_ID} is not in the queue."
    exit 2
  fi

  echo "iteration ${iteration}/${MAX_ITERATIONS}: ${EP_ID} stage=${current_stage} blocked=${blocked} final=${final_exists}"

  if [[ "${final_exists}" == "true" ]]; then
    echo "final video ready: ${final_path}"
    exit 0
  fi

  if [[ "${blocked}" == "true" ]]; then
    echo "auto-approving ${blocker_count} blocker(s) for ${EP_ID}"
    "${PYTHON_BIN}" -m pipeline.hermes_orchestrator --approve "${EP_ID}" || true
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  if [[ "${current_stage}" == "DONE" ]]; then
    echo "Episode ${EP_ID} is DONE but final.mp4 was not found."
    exit 2
  fi

  runnable_id="$(next_runnable_id)"
  if [[ -n "${runnable_id}" && "${runnable_id}" != "${EP_ID}" ]]; then
    echo "Next runnable episode is ${runnable_id}, not ${EP_ID}; refusing to advance the wrong episode."
    echo "Run without an episode id to process the queue head, or clear earlier queued work first."
    exit 2
  fi

  if ! "${PYTHON_BIN}" -m pipeline.hermes_orchestrator; then
    echo "orchestrator returned non-zero; checking queue state on next loop"
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "Stopped after MAX_ITERATIONS=${MAX_ITERATIONS} without producing final.mp4 for ${EP_ID}."
exit 2
