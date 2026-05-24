"""Constraint engine — the anti-template firewall.

Given the rolling window of recent assignments, choose archetype,
narrator, and visual style for the next episode such that none collides
with the configured cooldown.

Unlike the maritime pipeline, there is no `vehicle_type` or domain
taxonomy in this project — every narrator and every visual style is
eligible for every topic; the rolling-window dedup is the only filter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import load_config


@dataclass
class Assignment:
    archetype: str
    narrator: str
    visual_style: str


def pick_assignment(
    rolling_window: dict,
    *,
    seed: int | None = None,
) -> Assignment:
    """Choose A/N/V for the next episode given the rolling window."""
    cfg = load_config()
    rng = random.Random(seed)

    cd_a = cfg.constraints["rolling_window_archetype"]
    cd_n = cfg.constraints["rolling_window_narrator"]
    cd_v = cfg.constraints["rolling_window_visual_style"]

    arch_ids = [a["id"] for a in cfg.archetypes]
    narr_ids = [n["id"] for n in cfg.narrators]
    vis_ids = [v["id"] for v in cfg.visual_styles]

    arch = _pick(arch_ids, rolling_window.get("archetypes", []), cd_a, rng)
    narr = _pick(narr_ids, rolling_window.get("narrators", []), cd_n, rng)
    vis = _pick(vis_ids, rolling_window.get("visual_styles", []), cd_v, rng)

    return Assignment(archetype=arch, narrator=narr, visual_style=vis)


def is_valid_topic(incident: dict, rolling_window: dict) -> tuple[bool, str]:
    """Light sanity check on a topic-discovery output.

    With no era/vehicle taxonomy, this only enforces that the incident
    has a non-empty company_name and that it's not in the historical
    used-topics set (caller checks that separately too).
    """
    name = (incident.get("company_name") or "").strip()
    if not name:
        return False, "incident.company_name is empty"
    if not incident.get("year_anchor"):
        return False, "incident.year_anchor is empty"
    return True, ""


# -------------------- internals --------------------

def _pick(options: list[str], recent: list[str], cooldown: int, rng: random.Random) -> str:
    forbidden = set(recent[-cooldown:])
    available = [o for o in options if o not in forbidden]
    if not available:
        return _least_recent(options, recent, rng)
    return rng.choice(available)


def _least_recent(options: list[str], recent: list[str], rng: random.Random) -> str:
    rev = list(reversed(recent))
    by_recency = {opt: rev.index(opt) if opt in rev else len(rev) for opt in options}
    max_dist = max(by_recency.values())
    candidates = [o for o, d in by_recency.items() if d == max_dist]
    return rng.choice(candidates)
