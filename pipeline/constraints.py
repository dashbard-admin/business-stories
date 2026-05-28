"""Constraint engine — the anti-template firewall.

Given the rolling window of recent assignments, choose archetype,
narrator, and visual style for the next episode such that none collides
with the configured cooldown.

Batch G 2026-05-28: pick_assignment now honors per-narrator
`suits_story_kinds` (declared in config.yaml's narrators block). When
`story_kind` is provided, narrators whose suits_story_kinds list
excludes that story_kind are filtered out BEFORE the cooldown filter.
A narrator with NO suits_story_kinds field is treated as universal
(legacy behaviour preserved for N1-N4). If filtering leaves zero
candidates AND zero are eligible by cooldown either, fall back to the
least-recently-used narrator across the unfiltered pool.

Archetype + visual_style picks are unchanged.
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
    story_kind: str | None = None,
) -> Assignment:
    """Choose A/N/V for the next episode given the rolling window.

    `story_kind` (added Batch G 2026-05-28) gates which narrators are
    eligible via the `suits_story_kinds` field in config.yaml. Pass
    None to disable the filter (universal pick — legacy behaviour).
    """
    cfg = load_config()
    rng = random.Random(seed)

    cd_a = cfg.constraints["rolling_window_archetype"]
    cd_n = cfg.constraints["rolling_window_narrator"]
    cd_v = cfg.constraints["rolling_window_visual_style"]

    arch_ids = [a["id"] for a in cfg.archetypes]
    narr_ids = _eligible_narrators(cfg.narrators, story_kind)
    vis_ids = [v["id"] for v in cfg.visual_styles]

    arch = _pick(arch_ids, rolling_window.get("archetypes", []), cd_a, rng)
    narr = _pick(narr_ids, rolling_window.get("narrators", []), cd_n, rng)
    vis = _pick(vis_ids, rolling_window.get("visual_styles", []), cd_v, rng)

    return Assignment(archetype=arch, narrator=narr, visual_style=vis)


def _eligible_narrators(
    narrators: list[dict],
    story_kind: str | None,
) -> list[str]:
    """Filter narrators by suits_story_kinds when a story_kind is
    given. Narrators with no suits_story_kinds field are universal
    (always eligible). Returns a list of narrator IDs; never empty
    — if filtering eliminates everyone, returns the full unfiltered
    list with a quiet fallback."""
    if not story_kind:
        return [n["id"] for n in narrators]
    sk = story_kind.strip().lower()
    eligible: list[str] = []
    for n in narrators:
        suits = n.get("suits_story_kinds")
        if not suits:
            eligible.append(n["id"])  # universal narrator
            continue
        if any(s.strip().lower() == sk for s in suits):
            eligible.append(n["id"])
    if not eligible:
        # Defensive fallback — should never happen since N1-N4 are
        # universal, but keep the engine running if config gets weird.
        import logging
        logging.getLogger("hermes.constraints").warning(
            "no narrator suits story_kind=%r; using full pool", story_kind,
        )
        return [n["id"] for n in narrators]
    return eligible


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
