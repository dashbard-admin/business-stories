"""S14 — Performance writeback (Batch E 2026-05-27).

OUT-OF-BAND stage. Not in STAGE_DISPATCH / STAGE_ORDER. Triggered
manually (Q-E1 confirmed) via:

    python -m pipeline.hermes_orchestrator --analyse-performance

For every queue episode with `youtube_video_id` set, fetches the
latest performance metrics via pipeline.youtube_analytics, writes
them to 06_metadata/youtube_performance.json, and upserts the
summary into state/performance_history.json.

Subsequent S1 / S6 / S8 runs read the history via
pipeline.performance_summary.summarise_for_prompt() and inject the
patterns as prompt placeholders so future picks / scripts / beats
adapt to what worked.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from ..config import load_config
from ..performance_summary import upsert
from ..state import find_episode_workspace, load_queue
from ..youtube_analytics import YouTubeAnalytics, to_serialisable

logger = logging.getLogger("hermes.stage.s14")


def run() -> int:
    """Top-level entry called by the orchestrator. Returns 0 / non-0
    exit code. Differs from S1-S13's `run(episode, queue)` shape
    because S14 walks the entire queue itself."""
    cfg = load_config()
    if not cfg.youtube_analytics.get("enabled", False):
        print("--analyse-performance: cfg.youtube_analytics.enabled=false")
        return 0

    queue = load_queue()
    yt = YouTubeAnalytics()

    analysed = 0
    skipped = 0
    failed = 0

    for ep in queue["episodes"]:
        video_id = (ep.get("youtube_video_id") or "").strip()
        if not video_id:
            skipped += 1
            continue
        ws = find_episode_workspace(ep["id"])
        if ws is None:
            logger.warning("S14: no workspace for %s; skipping",
                           ep["id"])
            failed += 1
            continue

        perf = yt.fetch_episode(video_id)
        if perf is None:
            failed += 1
            continue

        # Write per-episode performance file.
        out = ws / "06_metadata" / "youtube_performance.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(to_serialisable(perf), indent=2))
        logger.info(
            "S14: %s — %d views, %.1f%% CTR, AVD %.0fs (%.0f%%)",
            ep["id"], perf.views, perf.ctr * 100,
            perf.avd_seconds, perf.avg_view_pct * 100,
        )

        # Build a per-episode history entry (much smaller than the
        # raw retention curve — that stays in the per-episode JSON).
        intent_retention = _intent_avg_retention(ws, perf)
        history_entry = {
            "episode_id": ep["id"],
            "video_id": video_id,
            "story_kind": (ep.get("incident") or {}).get("story_kind", ""),
            "hq_country": (ep.get("incident") or {}).get("hq_country", ""),
            "archetype": ep.get("archetype"),
            "narrator": ep.get("narrator"),
            "visual_style": ep.get("visual_style"),
            "fetched_at": perf.fetched_at,
            "views": perf.views,
            "ctr": perf.ctr,
            "avd_seconds": perf.avd_seconds,
            "avg_view_pct": perf.avg_view_pct,
            "peak_drop_at_seconds": perf.peak_drop_at_seconds,
            "intent_avg_retention": intent_retention,
        }
        upsert(history_entry)
        analysed += 1

    print(f"--analyse-performance: {analysed} analysed, "
          f"{skipped} skipped (no video_id), {failed} failed")
    return 0


# ----------------------------------------------------------------------
# Per-intent retention attribution
# ----------------------------------------------------------------------

def _intent_avg_retention(ws: Path, perf) -> dict[str, float]:
    """Cross-reference the retention curve against beat_sheet.json to
    compute average retention for each visual_intent. The curve gives
    retention by relative-position; we map each beat's timestamp window
    (from voice_timing.json) onto the curve, average within the window,
    and pool by visual_intent across all beats.

    Returns {intent: average_retention_0_to_1}.
    """
    bs_path = ws / "02_script" / "beat_sheet.json"
    vt_path = ws / "04_audio" / "voice_timing.json"
    if not bs_path.exists() or not vt_path.exists():
        return {}
    try:
        bs = json.loads(bs_path.read_text())
        vt = json.loads(vt_path.read_text())
    except Exception:
        return {}

    starts_by_id: dict[str, float] = {}
    ends_by_id: dict[str, float] = {}
    for b in vt.get("beats", []):
        bid = b.get("beat_id", "")
        if bid:
            starts_by_id[bid] = float(b.get("start_seconds", 0.0))
            ends_by_id[bid] = float(b.get("end_seconds", 0.0))

    curve = perf.retention_curve or []
    total_secs = vt.get("total_seconds", 0)
    if not curve or total_secs <= 0:
        return {}

    intent_buckets: dict[str, list[float]] = {}
    for b in bs.get("beats", []):
        bid = b.get("beat_id", "")
        intent = (b.get("visual_intent") or "").strip()
        if not intent or bid not in starts_by_id:
            continue
        start_pos = starts_by_id[bid] / total_secs
        end_pos = ends_by_id.get(bid, starts_by_id[bid]) / total_secs
        # Average retention within [start_pos, end_pos].
        window = [
            c["retention"] for c in curve
            if start_pos <= c["position"] <= end_pos
        ]
        if not window:
            continue
        intent_buckets.setdefault(intent, []).extend(window)

    return {
        intent: round(statistics.mean(values), 4)
        for intent, values in intent_buckets.items()
    }
