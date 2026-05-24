"""Hermes Orchestrator — cron entry point.

Every invocation:
  1. Acquires the global orchestrator lock.
  2. Loads the queue.
  3. Finds the next pending stage of the next non-blocked episode.
  4. Executes that one stage.
  5. Persists state, releases lock, exits.

Serial-by-stage model makes the pipeline trivial to recover from:
each cron run is idempotent and reentrant.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
import traceback

from .config import load_config
from .state import (
    enqueue_episodes,
    file_lock,
    load_queue,
    mark_stage_done,
    mark_stage_failed,
    next_pending_episode,
    save_queue,
)

logger = logging.getLogger("hermes.orchestrator")


# Map stage IDs to (module_path, function_name, display_name).
# Stage functions take (episode_dict, queue_dict) and return:
#   - None         : success
#   - str (reason) : needs_human
STAGE_DISPATCH = {
    "S1":  ("pipeline.stages.s01_topic_discovery",   "run", "Topic Discovery"),
    "S2":  ("pipeline.stages.s02_source_gathering",  "run", "Source Gathering"),
    "S3":  ("pipeline.stages.s03_fact_extraction",   "run", "Fact Extraction"),
    "S4":  ("pipeline.stages.s04_fact_verification", "run", "Fact Verification"),
    "S5":  ("pipeline.stages.s05_asset_hunt",        "run", "PD Asset Hunt"),
    "S6":  ("pipeline.stages.s06_script_generation", "run", "Script Generation"),
    "S7":  ("pipeline.stages.s07_script_critique",   "run", "Script Critique"),
    "S8":  ("pipeline.stages.s08_beat_sheet",        "run", "Beat Sheet"),
    "S9":  ("pipeline.stages.s09_flux_render",       "run", "FLUX Render"),
    "S10": ("pipeline.stages.s10_kokoro_render",     "run", "Kokoro TTS"),
    "S11": ("pipeline.stages.s11_audio_post",        "run", "Audio Post"),
    "S12": ("pipeline.stages.s12_video_assembly",    "run", "Video Assembly"),
}


def _stage_label(stage_id: str) -> str:
    entry = STAGE_DISPATCH.get(stage_id)
    if not entry or len(entry) < 3:
        return stage_id
    return f"{stage_id} ({entry[2]})"


def dispatch_stage(stage_id: str, episode: dict, queue: dict) -> str | None:
    module_path, func_name = STAGE_DISPATCH[stage_id][:2]
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func(episode, queue)


def run_one_invocation() -> int:
    cfg = load_config()
    lock_path = cfg.state_dir / "locks" / "orchestrator.lock"
    stale = cfg.orchestrator["stale_lock_hours"] * 3600
    max_runtime = cfg.orchestrator["per_invocation_max_runtime_seconds"]

    started = time.time()

    try:
        with file_lock(lock_path, stale_seconds=stale):
            queue = load_queue()
            pick = next_pending_episode(queue)
            if pick is None:
                logger.info("queue empty or fully blocked; nothing to do")
                return 0

            episode, stage_id = pick
            incident_name = (episode.get("incident") or {}).get(
                "company_name", "<unset>"
            )
            logger.info(
                "running %s for episode %s (topic=%s)",
                _stage_label(stage_id), episode["id"], incident_name,
            )

            if (time.time() - started) > max_runtime:
                logger.warning("budget exhausted before stage; deferring")
                return 0

            try:
                reason = dispatch_stage(stage_id, episode, queue)
            except KeyboardInterrupt:
                raise
            except Exception:
                tb = traceback.format_exc()
                logger.error("stage %s raised:\n%s", _stage_label(stage_id), tb)
                mark_stage_failed(queue, episode["id"], stage_id,
                                  f"exception: {tb.splitlines()[-1][:200]}")
                save_queue(queue)
                return 2

            if reason is None:
                mark_stage_done(queue, episode["id"], stage_id)
                logger.info("stage %s done for %s",
                            _stage_label(stage_id), episode["id"])
            else:
                mark_stage_failed(queue, episode["id"], stage_id, reason)
                logger.warning("stage %s needs_human for %s: %s",
                               _stage_label(stage_id), episode["id"], reason)

            save_queue(queue)
            return 0
    except RuntimeError as e:
        logger.warning("lock contention: %s", e)
        return 0


def _setup_logging(verbose: int) -> None:
    from datetime import datetime, timezone

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(level)

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    try:
        logs_dir = load_config().logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"orch.{datetime.now(timezone.utc):%Y-%m-%d}.log"
        already = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == str(log_path)
            for h in root.handlers
        )
        if not already:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(fmt)
            root.addHandler(fh)
    except Exception as e:
        logger.warning("file logging setup skipped: %s", e)


def cli() -> int:
    parser = argparse.ArgumentParser(prog="hermes-orchestrator")
    parser.add_argument("--enqueue", type=int, metavar="N",
                        help="add N empty episode records to the queue and exit")
    parser.add_argument("--status", action="store_true",
                        help="print queue status and exit")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.enqueue:
        ids = enqueue_episodes(args.enqueue)
        print(f"enqueued {len(ids)}: {', '.join(ids)}")
        return 0

    if args.status:
        q = load_queue()
        for ep in q["episodes"]:
            inc = (ep.get("incident") or {}).get("company_name", "<no topic>")
            blocked = "BLOCKED" if ep.get("blockers") else ""
            print(f"{ep['id']:8s}  stage={ep['current_stage']:4s}  "
                  f"{blocked:7s}  {inc}")
        return 0

    return run_one_invocation()


if __name__ == "__main__":
    sys.exit(cli())
