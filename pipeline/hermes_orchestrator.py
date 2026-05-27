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
    clear_blockers,
    enqueue_episodes,
    enqueue_manual_episode,
    file_lock,
    find_episode,
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
    # Added Batch D 2026-05-27.
    "S13": ("pipeline.stages.s13_packaging",         "run", "Packaging"),
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
    parser.add_argument(
        "--inject-topic", metavar="FILE",
        help=(
            "queue ONE episode with a manually-authored incident JSON. "
            "Bypasses the writer LLM and the rolling-window rotation "
            "hints. Optional pins for archetype/narrator/visual_style "
            "can live in the JSON; otherwise S01 picks them via the "
            "cooldown engine. The demand-validation gate still runs "
            "unless --no-validate is also given."
        ),
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help=(
            "with --inject-topic: skip the SearXNG demand-validation "
            "gate. Use when you genuinely want to cover a niche topic "
            "below the min_youtube_results floor (a personal-interest "
            "story the channel cares about regardless of search demand)."
        ),
    )
    parser.add_argument(
        "--preview", action="store_true",
        help=(
            "modifier flag (use with --enqueue or --inject-topic). "
            "Tags the new episode as preview_mode — S06 generates "
            "only Act 0 + Act 5 (~360 words, ~8 beats), and the rest "
            "of the pipeline renders only those beats. Tone-check the "
            "voice, visual style, hook, and closing without committing "
            "the full 3-4hr compute. Output: 05_video/final_preview.mp4."
        ),
    )
    parser.add_argument(
        "--approve", metavar="EP_ID",
        help=(
            "clear any S07 brand-safety gate or S08 in-flight gate on "
            "the named episode so it can advance to the next stage. "
            "Use after reviewing the flag file or beat sheet."
        ),
    )
    parser.add_argument(
        "--rerender", nargs=2, metavar=("EP_ID", "BEAT_ID"),
        help=(
            "re-run S09 FLUX render for a single beat. Use after "
            "editing beat_sheet.json's prompt for that beat, or after "
            "a FLUX render came out subjectively bad. Beat ID is the "
            "beat's `id` field (e.g. BEAT_023). The existing render "
            "and any Grok-corrected version are archived to "
            "03_assets/quarantine/ before the re-render."
        ),
    )
    parser.add_argument(
        "--from-edited-prompt", action="store_true",
        help=(
            "with --rerender: re-read the beat's FLUX prompt fresh "
            "from beat_sheet.json (operator edited it) instead of "
            "re-using the prompt the original S09 invocation built."
        ),
    )
    parser.add_argument(
        "--authorize-youtube", action="store_true",
        help=(
            "one-time OAuth dance for the YouTube Analytics API. "
            "Requires YOUTUBE_OAUTH_CLIENT_ID + "
            "YOUTUBE_OAUTH_CLIENT_SECRET in .env. Opens a browser; "
            "paste the auth code back. Token cached to "
            "state/youtube_oauth_token.json."
        ),
    )
    parser.add_argument(
        "--set-video-id", nargs=2, metavar=("EP_ID", "YT_VIDEO_ID"),
        help=(
            "after uploading episode EP_ID to YouTube, bind it to its "
            "video_id so S14 can fetch performance metrics. Stored on "
            "the episode record."
        ),
    )
    parser.add_argument(
        "--analyse-performance", action="store_true",
        help=(
            "out-of-band: run S14 over every episode with a "
            "youtube_video_id, pulling latest performance metrics "
            "into 06_metadata/youtube_performance.json and updating "
            "state/performance_history.json. Manual run only — not "
            "in the per-cron stage flow."
        ),
    )
    parser.add_argument("--status", action="store_true",
                        help="print queue status and exit")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.enqueue:
        ids = enqueue_episodes(args.enqueue, preview_mode=args.preview)
        tag = " (preview_mode)" if args.preview else ""
        print(f"enqueued {len(ids)}{tag}: {', '.join(ids)}")
        return 0

    if args.inject_topic:
        return _inject_topic_cmd(
            args.inject_topic,
            skip_validation=args.no_validate,
            preview_mode=args.preview,
        )

    if args.approve:
        return _approve_cmd(args.approve)

    if args.rerender:
        ep_id, beat_id = args.rerender
        return _rerender_cmd(
            ep_id, beat_id, from_edited_prompt=args.from_edited_prompt,
        )

    if args.authorize_youtube:
        from .youtube_analytics import authorize_oauth
        return authorize_oauth()

    if args.set_video_id:
        ep_id, vid = args.set_video_id
        return _set_video_id_cmd(ep_id, vid)

    if args.analyse_performance:
        from .stages.s14_performance_writeback import run as s14_run
        return s14_run()

    if args.status:
        q = load_queue()
        for ep in q["episodes"]:
            inc = (ep.get("incident") or {}).get("company_name", "<no topic>")
            tags: list[str] = []
            if ep.get("incident_origin") == "manual":
                tags.append("manual")
            if ep.get("preview_mode"):
                tags.append("preview")
            sf = ep.get("safety_flags_count") or {}
            if sf.get("high"):
                tags.append(f"safety_flags={sf['high']}H/{sf.get('low', 0)}L")
            elif sf.get("low"):
                tags.append(f"safety_flags=0H/{sf['low']}L")
            origin = f" ({', '.join(tags)})" if tags else ""
            blocked = "BLOCKED" if ep.get("blockers") else ""
            print(f"{ep['id']:8s}  stage={ep['current_stage']:4s}  "
                  f"{blocked:7s}  {inc}{origin}")
        return 0

    return run_one_invocation()


# ----------------------------------------------------------------------
# Manual-topic injection (Option B from the design doc)
# ----------------------------------------------------------------------

# The schema fields the operator must supply in the injection JSON.
# These match what S02-S12 actually read off `incident` downstream —
# if any are missing, later stages would crash with a KeyError or
# produce nonsense (e.g. blank year_anchor breaks the recency hints
# in the script-generation prompt). Better to fail loudly at inject
# time than ten stages later.
_MANUAL_REQUIRED_FIELDS = (
    "company_name",
    "founder_or_protagonist",
    "year_anchor",
    "story_kind",
    "hq_country",
    "hero",
    "conflict",
)

# Optional pins. If present in the JSON they go onto the episode
# record as `archetype_pin` / `narrator_pin` / `visual_style_pin` and
# S01 honors them; otherwise S01 picks via the cooldown engine.
_MANUAL_OPTIONAL_PINS = ("archetype", "narrator", "visual_style")


def _inject_topic_cmd(
    path_str: str,
    *,
    skip_validation: bool,
    preview_mode: bool = False,
) -> int:
    """CLI entry point for --inject-topic. Reads + validates the JSON,
    then enqueues a manual episode. Prints the new episode ID and the
    next stage that will run."""
    import json
    from pathlib import Path

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        print(f"--inject-topic: file not found: {path}", file=sys.stderr)
        return 2

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"--inject-topic: JSON parse error: {e}", file=sys.stderr)
        return 2

    if not isinstance(raw, dict):
        print("--inject-topic: top-level JSON must be an object", file=sys.stderr)
        return 2

    # Split incident fields from pin fields so we don't pollute the
    # incident dict with assignment metadata.
    pins = {k: raw.pop(k) for k in _MANUAL_OPTIONAL_PINS if k in raw}

    missing = [f for f in _MANUAL_REQUIRED_FIELDS if not raw.get(f)]
    if missing:
        print(
            "--inject-topic: missing required field(s): "
            + ", ".join(missing)
            + ". Required: " + ", ".join(_MANUAL_REQUIRED_FIELDS),
            file=sys.stderr,
        )
        return 2

    if not isinstance(raw.get("year_anchor"), int):
        print("--inject-topic: year_anchor must be an integer", file=sys.stderr)
        return 2

    try:
        eid = enqueue_manual_episode(
            incident=raw,
            archetype=pins.get("archetype"),
            narrator=pins.get("narrator"),
            visual_style=pins.get("visual_style"),
            skip_validation=skip_validation,
            preview_mode=preview_mode,
        )
    except ValueError as e:
        print(f"--inject-topic: {e}", file=sys.stderr)
        return 2

    pin_str = ""
    if pins:
        pin_str = " pins=" + ",".join(f"{k}={v}" for k, v in pins.items())
    val_str = " (validation skipped)" if skip_validation else ""
    pv_str = " (preview_mode)" if preview_mode else ""
    print(
        f"injected {eid}: {raw['company_name']} "
        f"[{raw.get('hq_country', '??')}, {raw.get('year_anchor')}, "
        f"{raw.get('story_kind')}]{pin_str}{val_str}{pv_str}"
    )
    return 0


# ----------------------------------------------------------------------
# Batch B: --approve and --rerender
# ----------------------------------------------------------------------

def _approve_cmd(episode_id: str) -> int:
    """Clear any needs_human gate on `episode_id` so the next cron
    tick can pick the stage back up. Used after operator reviews a
    brand-safety flag file or a beat sheet."""
    cfg = load_config()
    lock_path = cfg.state_dir / "locks" / "orchestrator.lock"
    stale = cfg.orchestrator["stale_lock_hours"] * 3600
    try:
        with file_lock(lock_path, stale_seconds=stale):
            queue = load_queue()
            ep = find_episode(queue, episode_id)
            if ep is None:
                print(f"--approve: no such episode {episode_id}",
                      file=sys.stderr)
                return 2
            if not (ep.get("blockers") or any(
                s.get("status") == "needs_human"
                for s in (ep.get("stages") or {}).values()
            )):
                print(f"--approve: {episode_id} has no blockers / no "
                      f"needs_human stages — nothing to clear.")
                return 0
            cleared = clear_blockers(queue, episode_id)
            save_queue(queue)
            if cleared:
                print(f"approved {episode_id}: blockers cleared; "
                      f"current_stage={ep['current_stage']}")
                return 0
            print(f"--approve: nothing changed for {episode_id}",
                  file=sys.stderr)
            return 2
    except RuntimeError as e:
        print(f"--approve: lock contention: {e}", file=sys.stderr)
        return 2


def _rerender_cmd(
    episode_id: str,
    beat_id: str,
    *,
    from_edited_prompt: bool = False,
) -> int:
    """Re-run S09's FLUX render path for a single beat. Loads the
    episode workspace, locates the beat in beat_sheet.json, archives
    any existing render under 03_assets/quarantine/, then re-renders
    via the same code path S09 uses. Optionally re-reads the prompt
    from beat_sheet.json (when --from-edited-prompt is set).

    Lives in the orchestrator (not in S09) because it's an out-of-
    band operation — the queue's stage status is not touched; the
    episode stays at whatever stage it was at."""
    cfg = load_config()
    lock_path = cfg.state_dir / "locks" / "orchestrator.lock"
    stale = cfg.orchestrator["stale_lock_hours"] * 3600
    try:
        with file_lock(lock_path, stale_seconds=stale):
            queue = load_queue()
            ep = find_episode(queue, episode_id)
            if ep is None:
                print(f"--rerender: no such episode {episode_id}",
                      file=sys.stderr)
                return 2

            # Delegate the actual re-render work to the stage module so
            # the FLUX call site stays in one place.
            from .stages import s09_flux_render as s09
            try:
                ok = s09.rerender_single_beat(
                    ep, beat_id,
                    from_edited_prompt=from_edited_prompt,
                )
            except FileNotFoundError as e:
                print(f"--rerender: {e}", file=sys.stderr)
                return 2

            if not ok:
                print(f"--rerender: failed for {episode_id}/{beat_id}",
                      file=sys.stderr)
                return 2

            save_queue(queue)
            print(f"rerendered {episode_id}/{beat_id} successfully")
            return 0
    except RuntimeError as e:
        print(f"--rerender: lock contention: {e}", file=sys.stderr)
        return 2


# ----------------------------------------------------------------------
# Batch E: --set-video-id
# ----------------------------------------------------------------------

def _set_video_id_cmd(episode_id: str, video_id: str) -> int:
    """Bind a published YouTube video_id to an episode record so S14
    can fetch its performance. Also stamps `published_at` so the
    feedback loop has a date anchor."""
    from datetime import datetime, timezone
    cfg = load_config()
    lock_path = cfg.state_dir / "locks" / "orchestrator.lock"
    stale = cfg.orchestrator["stale_lock_hours"] * 3600
    try:
        with file_lock(lock_path, stale_seconds=stale):
            queue = load_queue()
            ep = find_episode(queue, episode_id)
            if ep is None:
                print(f"--set-video-id: no such episode {episode_id}",
                      file=sys.stderr)
                return 2
            ep["youtube_video_id"] = video_id.strip()
            ep["published_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            save_queue(queue)
            print(f"bound {episode_id} → youtube://{video_id}")
            return 0
    except RuntimeError as e:
        print(f"--set-video-id: lock contention: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(cli())
