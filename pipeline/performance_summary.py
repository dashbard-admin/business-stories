"""Performance summary formatter (Batch E 2026-05-27).

Reads state/performance_history.json (aggregated by S14) and turns
the last N episodes' metrics into compact prompt-ready blocks
consumed by S1 / S6 / S8.

Per Q-E2 (confirmed): summarised pattern with up to 3 concrete
examples — not the full retention curves, but a synthesised
narrative the LLM can actually use:

  "Your last 5 scandal_postmortem episodes averaged 8.2% CTR vs.
   4.1% for origin episodes — prefer scandal_postmortem when the
   story supports it."

  "In recent episodes, viewers dropped sharply around the 3:40
   mark on beats with chart_abstraction visual_intent (concrete
   examples: EP_012, EP_018, EP_023). Avoid chart_abstraction in
   the middle third of the script."

  "Boardroom_meeting and product_reveal kept viewers above 60%
   retention; document_or_headline lost ~15% of viewers when
   placed before the 5-minute mark."

The summary is regenerated on every S1/S6/S8 run (cheap — pure
Python over a small JSON file). The user accepts that the
performance history must accumulate over several episodes before
the warnings have real signal.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

from .config import load_config

logger = logging.getLogger("hermes.performance_summary")


def _history_path() -> Path:
    return load_config().state_dir / "performance_history.json"


def load_history() -> list[dict[str, Any]]:
    """Return the per-episode performance history list. Empty when
    the file doesn't exist yet (no episodes published / analysed)."""
    path = _history_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return list(data.get("episodes") or [])
    except Exception as e:
        logger.warning("performance_history.json unreadable: %s", e)
        return []


def save_history(episodes: list[dict[str, Any]]) -> None:
    """Persist the full history list."""
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"episodes": episodes}, indent=2))


def upsert(entry: dict[str, Any]) -> None:
    """Insert-or-update by `episode_id`. S14 calls this."""
    history = load_history()
    by_id = {e.get("episode_id"): i for i, e in enumerate(history)}
    eid = entry.get("episode_id")
    if eid in by_id:
        history[by_id[eid]] = entry
    else:
        history.append(entry)
    save_history(history)


# ----------------------------------------------------------------------
# Summarisers — output strings for prompt injection
# ----------------------------------------------------------------------

def summarise_for_prompt(history: list[dict[str, Any]] | None = None,
                         *, k: int = 20) -> dict[str, str]:
    """Build prompt-ready strings from the last `k` episodes. Each
    string is a self-contained block the prompt-template can `.format()`
    directly. Empty strings when there's not enough history yet."""
    if history is None:
        history = load_history()
    if not history:
        return _empty_summary()

    cfg = load_config()
    window = int(cfg.youtube_analytics.get("summary_window", k))
    history = history[-window:]

    if len(history) < 2:
        # Below the minimum where averages mean anything.
        return _empty_summary()

    return {
        "top_performing_story_kinds": _summarise_story_kinds(history),
        "worst_performing_story_kinds": _summarise_worst_story_kinds(history),
        "retention_dip_warnings": _summarise_retention_dips(history),
        "visual_intents_that_retained": _summarise_intents_retained(history),
        "visual_intents_that_lost_viewers": _summarise_intents_lost(history),
    }


def _empty_summary() -> dict[str, str]:
    return {
        "top_performing_story_kinds": "(no performance history yet)",
        "worst_performing_story_kinds": "(no performance history yet)",
        "retention_dip_warnings": "(no performance history yet)",
        "visual_intents_that_retained": "(no performance history yet)",
        "visual_intents_that_lost_viewers": "(no performance history yet)",
    }


def _summarise_story_kinds(history: list[dict]) -> str:
    """Aggregate CTR + AVD-pct by story_kind. Return the top 2."""
    by_kind: dict[str, list[dict]] = {}
    for h in history:
        sk = (h.get("story_kind") or "").strip()
        if not sk:
            continue
        by_kind.setdefault(sk, []).append(h)
    if not by_kind:
        return "(no story_kind data)"

    rows: list[tuple[str, float, float, int]] = []
    for sk, eps in by_kind.items():
        ctrs = [float(e.get("ctr", 0)) for e in eps]
        avd_pcts = [float(e.get("avg_view_pct", 0)) for e in eps]
        rows.append((
            sk,
            statistics.mean(ctrs) if ctrs else 0.0,
            statistics.mean(avd_pcts) if avd_pcts else 0.0,
            len(eps),
        ))
    rows.sort(key=lambda r: r[1] * r[2], reverse=True)

    top = rows[:2]
    parts = []
    for sk, ctr, avd, n in top:
        parts.append(
            f"{sk}: avg CTR {ctr*100:.1f}%, avg AVD {avd*100:.0f}% "
            f"(n={n})"
        )
    return "Top-performing story_kinds in the last "\
        f"{len(history)} episodes: " + "; ".join(parts) + "."


def _summarise_worst_story_kinds(history: list[dict]) -> str:
    by_kind: dict[str, list[dict]] = {}
    for h in history:
        sk = (h.get("story_kind") or "").strip()
        if not sk:
            continue
        by_kind.setdefault(sk, []).append(h)
    if not by_kind:
        return "(no story_kind data)"

    rows: list[tuple[str, float, float, int]] = []
    for sk, eps in by_kind.items():
        ctrs = [float(e.get("ctr", 0)) for e in eps]
        avd_pcts = [float(e.get("avg_view_pct", 0)) for e in eps]
        rows.append((
            sk,
            statistics.mean(ctrs) if ctrs else 0.0,
            statistics.mean(avd_pcts) if avd_pcts else 0.0,
            len(eps),
        ))
    rows.sort(key=lambda r: r[1] * r[2])  # ascending

    worst = rows[:1]
    if not worst:
        return "(no story_kind data)"
    sk, ctr, avd, n = worst[0]
    return (
        f"Worst-performing story_kind: {sk} "
        f"(avg CTR {ctr*100:.1f}%, avg AVD {avd*100:.0f}%, n={n}). "
        f"Avoid this kind unless the story is genuinely outstanding."
    )


def _summarise_retention_dips(history: list[dict]) -> str:
    """Per Q-E2: 'summarised pattern with up to 3 concrete examples'.
    Find episodes whose peak retention drop fired in a specific time
    bucket (early/mid/late) and group them."""
    buckets: dict[str, list[str]] = {
        "0-3m (cold open / Act 1)": [],
        "3-7m (Act 2 / Act 3 start)": [],
        "7-12m (Act 3 / Act 3.5 midpoint)": [],
        "12-18m (Act 4 / Act 5)": [],
    }
    for h in history:
        drop_at = float(h.get("peak_drop_at_seconds", 0) or 0)
        eid = h.get("episode_id", "?")
        if drop_at <= 0:
            continue
        if drop_at < 180:
            buckets["0-3m (cold open / Act 1)"].append(eid)
        elif drop_at < 420:
            buckets["3-7m (Act 2 / Act 3 start)"].append(eid)
        elif drop_at < 720:
            buckets["7-12m (Act 3 / Act 3.5 midpoint)"].append(eid)
        else:
            buckets["12-18m (Act 4 / Act 5)"].append(eid)

    # Find the bucket with the most hits.
    busiest = max(buckets.items(), key=lambda kv: len(kv[1]))
    if not busiest[1]:
        return "(no retention-dip data yet)"
    examples = ", ".join(busiest[1][:3])
    return (
        f"Recent viewers consistently drop in the {busiest[0]} window "
        f"(concrete examples: {examples}). Strengthen hook cadence "
        f"and visual variety in that section."
    )


def _summarise_intents_retained(history: list[dict]) -> str:
    """Pool per-intent retention averages across episodes. Return the
    top 2 intents that kept viewers."""
    intent_ratios: dict[str, list[float]] = {}
    for h in history:
        for intent, ratio in (h.get("intent_avg_retention") or {}).items():
            intent_ratios.setdefault(intent, []).append(float(ratio))

    if not intent_ratios:
        return "(no visual_intent retention data yet)"

    rows = [
        (intent, statistics.mean(rs), len(rs))
        for intent, rs in intent_ratios.items()
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[:2]
    parts = [
        f"{intent} (avg retention {r*100:.0f}% across n={n})"
        for intent, r, n in top
    ]
    return "Top-retaining visual_intents: " + "; ".join(parts) + "."


def _summarise_intents_lost(history: list[dict]) -> str:
    intent_ratios: dict[str, list[float]] = {}
    for h in history:
        for intent, ratio in (h.get("intent_avg_retention") or {}).items():
            intent_ratios.setdefault(intent, []).append(float(ratio))
    if not intent_ratios:
        return "(no visual_intent retention data yet)"
    rows = [
        (intent, statistics.mean(rs), len(rs))
        for intent, rs in intent_ratios.items()
    ]
    rows.sort(key=lambda r: r[1])
    worst = rows[:2]
    parts = [
        f"{intent} (avg retention {r*100:.0f}% across n={n})"
        for intent, r, n in worst
    ]
    return (
        "Worst-retaining visual_intents: " + "; ".join(parts) +
        ". Consider replacing or relocating these intents."
    )
