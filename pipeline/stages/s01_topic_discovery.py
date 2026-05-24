"""S01 — Topic Discovery.

Asks the writer LLM for a candidate business / brand story, validates
it against the historical exclusion list and rolling-window
constraints. Stores assignment (archetype/narrator/visual_style) into
the episode record. Creates the per-episode workspace.

Inputs:  episode record (likely empty), queue, used_topics set
Outputs: incident dict + assignment on episode; workspace directory
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from ..config import load_config
from ..constraints import is_valid_topic, pick_assignment
from ..llm import LLM
from ..state import (
    add_used_topic,
    episode_workspace,
    load_used_topics,
    topic_already_used,
    update_episode,
)

logger = logging.getLogger("hermes.stage.s01")


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    llm = LLM(role="writer")

    used = load_used_topics()
    current_year = datetime.now(timezone.utc).year
    max_year = current_year - 5  # business stories cool off faster than disasters

    template = (cfg.prompts_dir / "topic_discovery.txt").read_text()

    max_retries = cfg.orchestrator["max_topic_discovery_retries"]
    rejection_reasons: list[str] = []
    incident: dict | None = None

    rolling = queue.get("rolling_window") or {}
    recent_story_kinds = _recent_story_kinds(queue)

    for attempt in range(1, max_retries + 1):
        exclusion = "\n".join(f"  - {x}" for x in sorted(used)) or "  (none)"
        rejected_inline = "\n".join(
            f"  REJECTED previously this run: {r}" for r in rejection_reasons
        )
        prompt = template.format(
            current_year=current_year,
            max_year=max_year,
            used_topics_list=exclusion + ("\n" + rejected_inline if rejected_inline else ""),
            recent_story_kinds=", ".join(recent_story_kinds) or "(none)",
        )
        logger.info("topic discovery attempt %d", attempt)
        try:
            candidate = llm.complete_json(prompt, temperature=0.8)
        except Exception as e:
            rejection_reasons.append(f"LLM JSON parse failed: {e}")
            continue

        # gate 1: required fields
        name = (candidate.get("company_name") or "").strip()
        if not name:
            rejection_reasons.append("missing company_name")
            continue
        if not (candidate.get("founder_or_protagonist") or "").strip():
            rejection_reasons.append(f"{name}: missing founder_or_protagonist")
            continue
        if not (candidate.get("hero") or "").strip():
            rejection_reasons.append(f"{name}: missing hero field")
            continue
        if not (candidate.get("conflict") or "").strip():
            rejection_reasons.append(f"{name}: missing conflict field")
            continue

        # gate 2: dedup
        if topic_already_used(name):
            rejection_reasons.append(f"{name}: already used")
            continue

        # gate 3: recency (anchor year ≤ max_year)
        year = candidate.get("year_anchor")
        if not isinstance(year, int) or year > max_year:
            rejection_reasons.append(f"{name}: year_anchor {year} fails recency gate")
            continue

        # gate 4: risk markers
        risk = (candidate.get("demonetization_risk_notes") or "").lower()
        risky_terms = [
            "ongoing litigation", "active investigation",
            "minors", "explicit gore", "recent terrorism",
        ]
        if any(t in risk for t in risky_terms):
            rejection_reasons.append(f"{name}: risk flag matched ({risk[:60]})")
            continue

        # gate 5: basic structural validity
        ok, why = is_valid_topic(
            {"company_name": name, "year_anchor": year},
            rolling,
        )
        if not ok:
            rejection_reasons.append(f"{name}: {why}")
            continue

        incident = candidate
        break

    if incident is None:
        return (
            f"could not find a valid topic after {max_retries} attempts: "
            + "; ".join(rejection_reasons[-3:])
        )

    # Pick A/N/V. No domain taxonomy in this pipeline — any narrator
    # and any visual style is eligible; only rolling-window cooldown
    # gates the choice.
    assignment = pick_assignment(
        rolling,
        seed=hash(incident["company_name"]) & 0xffff,
    )

    slug = _slugify(incident["company_name"])
    logger.info(
        "selected: %s (year=%s, story=%s, hero=%s, archetype=%s, narrator=%s, style=%s)",
        incident["company_name"], incident.get("year_anchor"),
        incident.get("story_kind"), (incident.get("hero") or "")[:60],
        assignment.archetype, assignment.narrator, assignment.visual_style,
    )

    ws = episode_workspace(episode["id"], slug)
    (ws / "00_research" / "incident.json").write_text(
        json.dumps(incident, indent=2)
    )
    (ws / "00_research" / "assignment.json").write_text(
        json.dumps({
            "archetype": assignment.archetype,
            "narrator": assignment.narrator,
            "visual_style": assignment.visual_style,
        }, indent=2)
    )

    update_episode(
        queue, episode["id"],
        slug=slug,
        incident=incident,
        archetype=assignment.archetype,
        narrator=assignment.narrator,
        visual_style=assignment.visual_style,
    )

    add_used_topic(incident["company_name"])
    return None


def _recent_story_kinds(queue: dict, keep: int = 3) -> list[str]:
    """Pull story_kind values from the most recent N episodes that
    have an incident set, oldest→newest. Used for soft rotation."""
    eps = queue.get("episodes") or []
    out: list[str] = []
    for ep in eps:
        inc = ep.get("incident") or {}
        sk = (inc.get("story_kind") or "").strip()
        if sk:
            out.append(sk)
    return out[-keep:]


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] or "unnamed"
