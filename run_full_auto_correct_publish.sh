#!/usr/bin/env bash
# Run one episode continuously until final.mp4 exists, auto-handle
# review gates with the strongest available stage-specific action,
# then build and upload the YouTube package.
#
# Gate policy:
#   S07 -> --auto-correct-s07, fallback to --approve if no rewrite applies
#   S08 -> --auto-correct-s08
#   S09 -> --auto-approve-s09
#   S10+ -> stop; these are build failures, not review gates

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
LOGFILE="${LOGDIR}/full_auto_correct_publish.${TS}.log"
exec > >(tee -a "${LOGFILE}") 2>&1
# Long foreground runs may be launched from shells/agents whose stdin
# disappears later. Python can fail during startup if it inherits that
# bad descriptor, so default every subprocess to /dev/null.
exec </dev/null

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
EP_ID="${1:-}"

echo "full-auto correct+publish runner log: ${LOGFILE}"

orchestrator() {
  "${PYTHON_BIN}" -m pipeline.hermes_orchestrator "$@" </dev/null
}

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

handle_review_gate() {
  local stage="$1"
  local episode_id="$2"
  local blocker_count="$3"

  case "${stage}" in
    S7)
      echo "auto-correcting S07 blocker(s) for ${episode_id}"
      if ! orchestrator --auto-correct-s07 "${episode_id}"; then
        echo "S07 auto-correct did not apply cleanly; approving gate to keep review-stage automation moving"
        orchestrator --approve "${episode_id}"
      fi
      ;;
    S8)
      echo "auto-correcting S08 blocker(s) for ${episode_id}"
      orchestrator --auto-correct-s08 "${episode_id}"
      ;;
    S9)
      echo "auto-approving S09 visual review blocker(s) for ${episode_id}"
      orchestrator --auto-approve-s09 "${episode_id}"
      ;;
    *)
      echo "Episode ${episode_id} is blocked at ${stage}; not auto-clearing build-stage failures."
      echo "Blocked stages from S10 onward require operator review."
      echo "Review state/episode_queue.json and the latest log before rerunning."
      return 2
      ;;
  esac

  echo "handled ${blocker_count} blocker(s) at ${stage}"
}

if [[ -z "${EP_ID}" ]]; then
  read -r EP_ID _stage _blocked _final _path _count < <(queue_state "")
  if [[ "${EP_ID}" == "NONE" ]]; then
    echo "No queued episode found; enqueueing one episode."
    orchestrator --enqueue 1
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
    break
  fi

  if [[ "${blocked}" == "true" ]]; then
    handle_review_gate "${current_stage}" "${EP_ID}" "${blocker_count}"
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  if [[ "${current_stage}" == "DONE" ]]; then
    echo "Episode ${EP_ID} is DONE but final.mp4 was not found."
    exit 2
  fi

  if ! orchestrator --run-episode "${EP_ID}"; then
    echo "orchestrator returned non-zero; checking queue state on next loop"
  fi
  sleep "${SLEEP_SECONDS}"
done

read -r _ep _stage _blocked final_exists final_path _count < <(queue_state "${EP_ID}")
if [[ "${final_exists}" != "true" ]]; then
  echo "Stopped after MAX_ITERATIONS=${MAX_ITERATIONS} without producing final.mp4 for ${EP_ID}."
  exit 2
fi

echo "building YouTube package for ${EP_ID}"
orchestrator --build-youtube-package "${EP_ID}"

echo "uploading YouTube package for ${EP_ID}"
UPLOAD_ARGS=(--upload-youtube-package "${EP_ID}" --approve-youtube-upload)
if [[ -n "${YOUTUBE_PRIVACY:-}" ]]; then
  UPLOAD_ARGS+=(--youtube-privacy "${YOUTUBE_PRIVACY}")
fi
if [[ -n "${YOUTUBE_PUBLISH_AT:-}" ]]; then
  UPLOAD_ARGS+=(--youtube-publish-at "${YOUTUBE_PUBLISH_AT}")
fi
orchestrator "${UPLOAD_ARGS[@]}"

echo "completed build/package/upload for ${EP_ID}"
