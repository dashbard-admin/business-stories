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

    Batch H 2026-05-28: ALSO gates archetypes by suits_story_kinds
    (so e.g. A5 Underdog Comeback never gets picked for a
    scandal_postmortem story — Quibi forced exactly that bug). AND
    honors per-narrator `enabled: false` flag (operator can pin
    the channel to one voice during testing without deleting
    narrators from config).
    """
    cfg = load_config()
    rng = random.Random(seed)

    cd_a = cfg.constraints["rolling_window_archetype"]
    cd_n = cfg.constraints["rolling_window_narrator"]
    cd_v = cfg.constraints["rolling_window_visual_style"]

    arch_ids = _eligible_archetypes(cfg.archetypes, story_kind)
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
    """Filter narrators by `enabled` flag (Batch H 2026-05-28) and by
    suits_story_kinds (Batch G 2026-05-28). Narrators with
    `enabled: false` are dropped entirely. Narrators without a
    suits_story_kinds field are universal across story_kinds. If
    filtering produces an empty pool, fall back to all ENABLED
    narrators (ignoring the story_kind filter); if even THAT is
    empty (operator disabled everyone), fall back to the full
    unfiltered list with a warning.
    """
    enabled_narrators = [
        n for n in narrators if n.get("enabled", True)
    ]
    if not enabled_narrators:
        import logging
        logging.getLogger("hermes.constraints").warning(
            "no narrators have enabled=true; using full pool"
        )
        return [n["id"] for n in narrators]

    if not story_kind:
        return [n["id"] for n in enabled_narrators]
    sk = story_kind.strip().lower()
    eligible: list[str] = []
    for n in enabled_narrators:
        suits = n.get("suits_story_kinds")
        if not suits:
            eligible.append(n["id"])
            continue
        if any(s.strip().lower() == sk for s in suits):
            eligible.append(n["id"])
    if not eligible:
        import logging
        logging.getLogger("hermes.constraints").warning(
            "no enabled narrator suits story_kind=%r; using full "
            "enabled pool", story_kind,
        )
        return [n["id"] for n in enabled_narrators]
    return eligible


def _eligible_archetypes(
    archetypes: list[dict],
    story_kind: str | None,
) -> list[str]:
    """Filter archetypes by suits_story_kinds (Batch H 2026-05-28).
    Same semantics as _eligible_narrators: archetypes without
    suits_story_kinds are universal; if filtering empties the pool,
    fall back to all archetypes."""
    if not story_kind:
        return [a["id"] for a in archetypes]
    sk = story_kind.strip().lower()
    eligible: list[str] = []
    for a in archetypes:
        suits = a.get("suits_story_kinds")
        if not suits:
            eligible.append(a["id"])
            continue
        if any(s.strip().lower() == sk for s in suits):
            eligible.append(a["id"])
    if not eligible:
        import logging
        logging.getLogger("hermes.constraints").warning(
            "no archetype suits story_kind=%r; using full pool", story_kind,
        )
        return [a["id"] for a in archetypes]
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
