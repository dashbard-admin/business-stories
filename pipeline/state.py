"""State management — queue, locks, rolling-window dedup, per-episode workspaces.

Queue lives in `state/episode_queue.json`. Reads/writes are serialized
with a file-based advisory lock so concurrent cron invocations cannot
clobber each other. `used_topics.json` holds the dedup set across all
historical episodes.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import load_config


SCHEMA_VERSION = 1


# -------------------------- locking --------------------------

@contextmanager
def file_lock(path: Path, stale_seconds: int = 6 * 3600) -> Iterator[None]:
    """Cross-process advisory lock with stale-lock reclamation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        f.seek(0)
        existing = f.read().strip()
        if existing:
            try:
                ts = float(existing)
                if (time.time() - ts) > stale_seconds:
                    pass  # stale; we'll overwrite below
            except ValueError:
                pass
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another process holds {path}")
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))
        f.flush()
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


# -------------------------- queue --------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _queue_path() -> Path:
    return load_config().state_dir / "episode_queue.json"


def _used_topics_path() -> Path:
    return load_config().state_dir / "used_topics.json"


STAGE_ORDER = [f"S{i}" for i in range(1, 14)]   # S1 .. S13
                                                # S13 added Batch D
                                                # 2026-05-27 (packaging:
                                                # title variants,
                                                # thumbnails, shorts).


def _init_queue() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "episodes": [],
        "rolling_window": {
            "archetypes": [],
            "narrators": [],
            "visual_styles": [],
            # ISO 3166-1 alpha-2 country code of the protagonist
            # company's HQ. Pushed by S01 on every successful
            # topic commit; consumed by pipeline.trends.non_us_required
            # to enforce the 1-in-N non-US ratio.
            "countries": [],
        },
    }


def load_queue() -> dict[str, Any]:
    path = _queue_path()
    if not path.exists():
        return _init_queue()
    raw = path.read_text()
    try:
        queue = json.loads(raw)
    except json.JSONDecodeError as e:
        # The queue file is operator-editable (S14 binding, manual
        # blocker clears, hand-tuning). Trailing commas are the most
        # common hand-edit mistake; Python's strict json parser
        # rejects them. Recover and re-save the cleaned form so the
        # next load is fast.  Added 2026-05-28.
        if "trailing comma" not in str(e).lower():
            raise
        import logging, re
        logger = logging.getLogger("hermes.state")
        logger.warning(
            "queue %s had trailing-comma JSON syntax (line %d col %d); "
            "auto-cleaning and re-saving",
            path, e.lineno, e.colno,
        )
        cleaned = re.sub(r",(\s*[\]}])", r"\1", raw)
        try:
            queue = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            # Bad in a different way — surface the original error so
            # the operator can hand-fix.
            raise json.JSONDecodeError(
                f"queue {path} unparseable even after trailing-comma "
                f"recovery: {e2.msg}", cleaned, e2.pos
            ) from e2
        # Persist the cleaned version atomically.
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(queue, f, indent=2, sort_keys=False)
        os.replace(tmp, path)
    _migrate_queue_in_place(queue)
    return queue


def _migrate_queue_in_place(queue: dict[str, Any]) -> None:
    """Forward-migrate an on-disk queue loaded from an older pipeline
    version so it has every stage key the current STAGE_ORDER expects.

    Hit in production after Batch D added S13: existing episode
    records had stages={S1..S12}, and next_pending_episode tried
    ep["stages"]["S13"] → KeyError. This walks every episode and
    inserts any missing stage as `pending`. Idempotent.

    Also backfills missing rolling_window keys (Batch A added
    `countries`) so a queue saved before that batch still loads.

    Added 2026-05-28.
    """
    # Per-episode stages.
    for ep in queue.get("episodes", []):
        stages = ep.setdefault("stages", {})
        for sid in STAGE_ORDER:
            if sid not in stages:
                stages[sid] = {"status": "pending", "ts": None}
        # If current_stage is "DONE" but new stages got appended,
        # treat the earliest pending NEW stage as the next runnable.
        cs = ep.get("current_stage")
        if cs == "DONE":
            for sid in STAGE_ORDER:
                if stages[sid].get("status") == "pending":
                    ep["current_stage"] = sid
                    break

    # Rolling-window keys.
    rw = queue.setdefault("rolling_window", {})
    for key in ("archetypes", "narrators", "visual_styles", "countries"):
        rw.setdefault(key, [])


def save_queue(queue: dict[str, Any]) -> None:
    path = _queue_path()
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(queue, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def load_used_topics() -> set[str]:
    path = _used_topics_path()
    if not path.exists():
        return set()
    with path.open() as f:
        return set(json.load(f))


def save_used_topics(items: set[str]) -> None:
    path = _used_topics_path()
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(sorted(items), f, indent=2)
    os.replace(tmp, path)


def add_used_topic(name: str) -> None:
    s = load_used_topics()
    s.add(name.strip().lower())
    save_used_topics(s)


def topic_already_used(name: str) -> bool:
    return name.strip().lower() in load_used_topics()


# -------------------------- episode helpers --------------------------

def new_episode_record(episode_id: str) -> dict[str, Any]:
    return {
        "id": episode_id,
        "slug": None,
        "incident": None,
        "archetype": None,
        "narrator": None,
        "visual_style": None,
        "current_stage": "S1",
        "stages": {s: {"status": "pending", "ts": None} for s in STAGE_ORDER},
        "blockers": [],
        "created_at": _now(),
    }


def enqueue_episodes(
    n: int,
    *,
    preview_mode: bool = False,
    narrator_pin: str | None = None,
    archetype_pin: str | None = None,
    visual_style_pin: str | None = None,
) -> list[str]:
    """Add `n` empty episode records to the queue. Returns new IDs.

    `preview_mode=True` flags each new record so S06 generates only
    Act 0 + Act 5 (added Batch B 2026-05-26) — a tone-check render
    that takes ~10 min of compute instead of 3-4 hours.

    `narrator_pin` / `archetype_pin` / `visual_style_pin` (added Batch
    G 2026-05-28) lock the corresponding assignment dimension on
    every new record so S01 honors them instead of asking the
    cooldown engine. Use the CLI `--narrator N5` flag to set
    narrator_pin from the command line.
    """
    queue = load_queue()
    existing = {e["id"] for e in queue["episodes"]}
    next_idx = 1
    while f"EP_{next_idx:03d}" in existing:
        next_idx += 1
    new_ids = []
    for i in range(n):
        eid = f"EP_{next_idx + i:03d}"
        rec = new_episode_record(eid)
        if preview_mode:
            rec["preview_mode"] = True
        if narrator_pin:
            rec["narrator_pin"] = narrator_pin
        if archetype_pin:
            rec["archetype_pin"] = archetype_pin
        if visual_style_pin:
            rec["visual_style_pin"] = visual_style_pin
        queue["episodes"].append(rec)
        new_ids.append(eid)
    save_queue(queue)
    return new_ids


def enqueue_manual_episode(
    incident: dict[str, Any],
    *,
    archetype: str | None = None,
    narrator: str | None = None,
    visual_style: str | None = None,
    skip_validation: bool = False,
    preview_mode: bool = False,
) -> str:
    """Add ONE episode record with the incident pre-filled by the
    operator (not the LLM). S01 detects `incident_origin == "manual"`
    on the record and short-circuits the LLM pick — it still runs
    A/N/V assignment (unless any of the three is pinned here) and
    optionally the demand-validation gate (skipped iff
    skip_validation=True).

    Returns the new episode ID.

    The caller (the orchestrator CLI) is responsible for schema-
    validating `incident` before reaching here. This function only
    enforces that company_name is non-empty so the queue stays
    consistent.
    """
    name = (incident.get("company_name") or "").strip()
    if not name:
        raise ValueError("manual incident: company_name is required")

    queue = load_queue()
    existing = {e["id"] for e in queue["episodes"]}
    idx = 1
    while f"EP_{idx:03d}" in existing:
        idx += 1
    eid = f"EP_{idx:03d}"

    rec = new_episode_record(eid)
    rec["incident"] = dict(incident)
    rec["incident_origin"] = "manual"
    rec["skip_validation"] = bool(skip_validation)
    if preview_mode:
        rec["preview_mode"] = True
    if archetype:
        rec["archetype_pin"] = archetype
    if narrator:
        rec["narrator_pin"] = narrator
    if visual_style:
        rec["visual_style_pin"] = visual_style

    queue["episodes"].append(rec)
    save_queue(queue)
    return eid


def next_pending_episode(queue: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Find next episode with a runnable stage. Returns (episode, stage_id) or None."""
    for ep in queue["episodes"]:
        if ep.get("blockers"):
            continue
        for stage_id in STAGE_ORDER:
            stage = ep["stages"][stage_id]
            if stage["status"] == "pending":
                return ep, stage_id
            if stage["status"] == "needs_human":
                break
    return None


def mark_stage_done(queue: dict[str, Any], episode_id: str, stage_id: str) -> None:
    for ep in queue["episodes"]:
        if ep["id"] == episode_id:
            ep["stages"][stage_id] = {"status": "done", "ts": _now()}
            idx = STAGE_ORDER.index(stage_id)
            if idx + 1 < len(STAGE_ORDER):
                ep["current_stage"] = STAGE_ORDER[idx + 1]
            else:
                ep["current_stage"] = "DONE"
            return


def mark_stage_failed(
    queue: dict[str, Any], episode_id: str, stage_id: str, reason: str
) -> None:
    for ep in queue["episodes"]:
        if ep["id"] == episode_id:
            ep["stages"][stage_id] = {
                "status": "needs_human",
                "ts": _now(),
                "reason": reason,
            }
            ep["blockers"].append({"stage": stage_id, "reason": reason, "ts": _now()})
            return


def update_episode(queue: dict[str, Any], episode_id: str, **fields: Any) -> None:
    for ep in queue["episodes"]:
        if ep["id"] == episode_id:
            ep.update(fields)
            return


def find_episode(queue: dict[str, Any], episode_id: str) -> dict[str, Any] | None:
    """Return the episode record for `episode_id`, or None. Added Batch B
    2026-05-26 for the --approve and --rerender CLI flows."""
    for ep in queue["episodes"]:
        if ep["id"] == episode_id:
            return ep
    return None


def clear_blockers(
    queue: dict[str, Any],
    episode_id: str,
    *,
    stage_filter: str | None = None,
) -> bool:
    """Approve a stage past a `needs_human` gate. Clears blockers on
    `episode_id`, marks any gated stage as `done`, and advances
    `current_stage` to the NEXT stage in STAGE_ORDER so the
    orchestrator picks up at the following stage instead of looping.

    Bugfix 2026-05-28: previously this function reset `needs_human`
    stages back to `pending` AND set `current_stage` to the cleared
    stage's own ID — which made the orchestrator's next tick re-run
    the gated stage in an infinite loop. The brand-safety + S08
    in-flight gates exist precisely because the stage's WORK is
    already done (script.txt + brand_safety_flags.json present, or
    beat_sheet.json present) — `--approve` means "the artifact is
    OK, ship it" not "re-run this stage from scratch".

    If `stage_filter` is given, only clear blockers/advance for that
    stage. Returns True iff something was cleared."""
    ep = find_episode(queue, episode_id)
    if ep is None:
        return False
    cleared = False
    keep: list[dict[str, Any]] = []
    for b in (ep.get("blockers") or []):
        if stage_filter is None or b.get("stage") == stage_filter:
            cleared = True
            continue
        keep.append(b)
    ep["blockers"] = keep
    for sid, sval in (ep.get("stages") or {}).items():
        if sval.get("status") != "needs_human":
            continue
        if stage_filter is not None and sid != stage_filter:
            continue
        ep["stages"][sid] = {"status": "done", "ts": _now()}
        # Advance to the next stage in STAGE_ORDER. Mirrors the
        # logic in mark_stage_done().
        try:
            idx = STAGE_ORDER.index(sid)
        except ValueError:
            idx = -1
        if 0 <= idx < len(STAGE_ORDER) - 1:
            ep["current_stage"] = STAGE_ORDER[idx + 1]
        else:
            ep["current_stage"] = "DONE"
        cleared = True
    return cleared


def episode_workspace(episode_id: str, slug: str | None = None) -> Path:
    """Return the per-episode workspace path, creating sub-folders."""
    cfg = load_config()
    name = f"{episode_id}_{slug}" if slug else episode_id
    ws = cfg.episodes_dir / name
    for sub in [
        "00_research/raw",
        "00_research/extracted",
        "01_factcheck",
        "02_script",
        "03_assets/pd",
        "03_assets/flux",
        "03_assets/quarantine",
        "04_audio/chunks",
        "04_audio/music",
        "05_video/clips",
        "06_metadata",
    ]:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    return ws


def find_episode_workspace(episode_id: str) -> Path | None:
    """Locate workspace by episode_id, with or without slug."""
    cfg = load_config()
    for entry in cfg.episodes_dir.glob(f"{episode_id}*"):
        if entry.is_dir():
            return entry
    return None


# -------------------------- rolling window --------------------------

def push_rolling_window(
    queue: dict[str, Any],
    archetype: str,
    narrator: str,
    visual_style: str,
    keep: int = 6,
    *,
    country: str | None = None,
) -> None:
    """Append the assignment dimensions (and optionally the country)
    to the rolling-window state, trimming to the most recent `keep`
    entries per key. `country` is optional only because legacy callers
    that pre-date the country track may not pass it; S01 always does."""
    rw = queue["rolling_window"]
    entries: list[tuple[str, str]] = [
        ("archetypes", archetype),
        ("narrators", narrator),
        ("visual_styles", visual_style),
    ]
    if country is not None:
        entries.append(("countries", country))
    for key, val in entries:
        rw.setdefault(key, []).append(val)
        rw[key] = rw[key][-keep:]
