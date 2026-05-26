"""S01 — Topic Discovery.

Asks the writer LLM for a candidate business / brand story, validates
it against:
  - historical exclusion list (used_topics.json)
  - rolling-window cooldown (archetype / narrator / visual_style)
  - basic schema (required fields, year recency, risk markers)
  - demand signals (SearXNG video saturation, news activity) via
    pipeline.trends
  - geographic-diversity floor (1-in-N non-US, configurable)
  - decline-story bias (prompt-side hint, not a hard gate)

On success, persists incident.json + assignment.json into the per-
episode workspace and pushes the assignment + country into the
rolling-window state so the next S01 run rotates correctly.

Inputs:  episode record (likely empty), queue, used_topics set
Outputs: incident dict + assignment on episode; workspace directory
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from ..browser import Browser
from ..config import load_config
from ..constraints import is_valid_topic, pick_assignment
from ..llm import LLM
from ..state import (
    add_used_topic,
    episode_workspace,
    load_used_topics,
    push_rolling_window,
    topic_already_used,
    update_episode,
)
from ..trends import non_us_required, validate_candidate

logger = logging.getLogger("hermes.stage.s01")


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    browser = Browser()
    tv_cfg = cfg.topic_validation

    used = load_used_topics()
    current_year = datetime.now(timezone.utc).year
    max_year = current_year - 5  # business stories cool off faster than disasters

    # Operator-injected topic short-circuit. When `--inject-topic` was
    # used to queue this episode, S01 skips the LLM call and the
    # rolling-window rotation hints entirely. We still run dedup, the
    # country normaliser, and (unless --no-validate was given) the
    # SearXNG saturation gate — those are about catching mistakes, not
    # about picking the topic.
    if episode.get("incident_origin") == "manual":
        return _run_manual(episode, queue, browser, tv_cfg, used)

    llm = LLM(role="writer")
    template = (cfg.prompts_dir / "topic_discovery.txt").read_text()

    max_retries = cfg.orchestrator["max_topic_discovery_retries"]
    rejection_reasons: list[str] = []
    incident: dict | None = None
    last_signals: dict = {}

    rolling = queue.get("rolling_window") or {}
    recent_story_kinds = _recent_story_kinds(queue)
    recent_countries = _recent_countries(rolling)

    # Pre-LLM: decide whether the next pick MUST be non-US to keep the
    # rolling-window non-US ratio on target. The result is folded into
    # the prompt as a hard hint, AND enforced post-LLM as a gate so an
    # ignored hint still gets caught.
    require_non_us = non_us_required(
        queue,
        ratio=float(tv_cfg.get("non_us_ratio", 0.33)),
        lookback=int(tv_cfg.get("non_us_ratio_lookback", 6)),
    )

    decline_hint = _decline_hint(tv_cfg)
    non_us_hint = _non_us_hint(require_non_us, tv_cfg, recent_countries)

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
            recent_countries=", ".join(recent_countries) or "(none)",
            decline_preference_hint=decline_hint,
            non_us_required_hint=non_us_hint,
        )
        logger.info("topic discovery attempt %d (require_non_us=%s)",
                    attempt, require_non_us)
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

        # gate 6: geographic diversity. If the rolling window says the
        # next pick must be non-US, enforce it as a hard reject so the
        # LLM gets feedback and tries again with a different country.
        country = _normalise_country(candidate.get("hq_country"))
        candidate["hq_country"] = country  # store normalised form back
        if require_non_us and country == "US":
            rejection_reasons.append(
                f"{name}: hq_country=US but rolling window requires non-US "
                f"(recent: {', '.join(recent_countries) or 'n/a'})"
            )
            continue

        # gate 7: demand validation via SearXNG. This is the costliest
        # gate (two network calls), so it runs last. Failure feeds back
        # into the rejection list so the LLM proposes a different
        # company on the next attempt.
        verdict = validate_candidate(candidate, tv_cfg, browser)
        last_signals = verdict.signals
        if not verdict.ok:
            rejection_reasons.append(f"{name}: {verdict.reason}")
            continue

        # Attach the demand signals to the candidate so downstream
        # stages (and the operator's eyes) can see why this topic was
        # accepted.
        candidate["validation_signals"] = verdict.signals
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

    _finalise_pick(episode, queue, incident, assignment, last_signals,
                   origin="llm")
    return None


# ----------------------------------------------------------------------
# Manual-injection path (operator pre-filled the incident via
# `hermes-orchestrator --inject-topic <file.json>`)
# ----------------------------------------------------------------------

def _run_manual(
    episode: dict,
    queue: dict,
    browser: Browser,
    tv_cfg: dict,
    used: set[str],
) -> str | None:
    """Handle an operator-injected episode. The incident JSON has
    already been schema-validated at inject time; this function still
    runs the cheap dedup + country-normalisation gates, optionally
    runs the SearXNG demand validation (skipped iff
    episode.skip_validation is True), and picks A/N/V — honoring any
    `archetype_pin` / `narrator_pin` / `visual_style_pin` set on the
    episode record.
    """
    incident = dict(episode.get("incident") or {})
    name = (incident.get("company_name") or "").strip()
    if not name:
        return "manual injection: company_name missing on episode record"

    # Re-run dedup. The operator might have queued the same topic
    # twice across runs without realising, or used_topics may have
    # picked it up via the auto-pilot in the meantime. Better to
    # fail loudly here than render a duplicate.
    if topic_already_used(name):
        return (
            f"manual injection: {name!r} already in used_topics.json — "
            f"clear it manually if you intend to re-cover this topic"
        )

    # Normalise country so the rolling-window non-US accounting
    # treats it the same as auto picks.
    incident["hq_country"] = _normalise_country(incident.get("hq_country"))

    signals: dict = {}
    if not episode.get("skip_validation"):
        verdict = validate_candidate(incident, tv_cfg, browser)
        signals = verdict.signals
        if not verdict.ok:
            return (
                f"manual injection demand-gate failed: {verdict.reason}. "
                f"Re-run with --no-validate if you want to render this "
                f"topic anyway."
            )
        incident["validation_signals"] = verdict.signals
    else:
        incident["validation_signals"] = {"skipped": True,
                                          "reason": "--no-validate"}

    rolling = queue.get("rolling_window") or {}
    base = pick_assignment(
        rolling,
        seed=hash(incident["company_name"]) & 0xffff,
    )
    # Honor per-episode pins. The CLI strips these from `incident` and
    # parks them on the episode record so they don't pollute the
    # incident-shape that downstream stages read.
    arch = episode.get("archetype_pin") or base.archetype
    narr = episode.get("narrator_pin") or base.narrator
    vis = episode.get("visual_style_pin") or base.visual_style

    from ..constraints import Assignment   # local import to keep top
    assignment = Assignment(archetype=arch, narrator=narr, visual_style=vis)

    _finalise_pick(episode, queue, incident, assignment, signals,
                   origin="manual")
    return None


# ----------------------------------------------------------------------
# Shared post-acceptance bookkeeping (LLM and manual paths)
# ----------------------------------------------------------------------

def _finalise_pick(
    episode: dict,
    queue: dict,
    incident: dict,
    assignment,
    signals: dict,
    *,
    origin: str,
) -> None:
    """Persist incident.json + assignment.json, update the queue
    record, push the rolling-window state, and add the topic to the
    used-topics set. Called by both run() (LLM path) and
    _run_manual()."""
    slug = _slugify(incident["company_name"])
    logger.info(
        "selected [%s]: %s [%s] (year=%s, story=%s, hero=%s, "
        "archetype=%s, narrator=%s, style=%s, signals=%s)",
        origin,
        incident["company_name"], incident.get("hq_country", "??"),
        incident.get("year_anchor"), incident.get("story_kind"),
        (incident.get("hero") or "")[:60],
        assignment.archetype, assignment.narrator, assignment.visual_style,
        signals,
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
            "origin": origin,
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

    # Update rolling window. push_rolling_window was previously never
    # called in this pipeline — the rolling-window dimensions stayed
    # empty across episodes and the cooldown filter therefore did
    # nothing. Fixed here as a side-effect of adding country tracking.
    push_rolling_window(
        queue,
        archetype=assignment.archetype,
        narrator=assignment.narrator,
        visual_style=assignment.visual_style,
        country=incident.get("hq_country") or "??",
    )

    add_used_topic(incident["company_name"])


# ----------------------------------------------------------------------
# Prompt hint construction
# ----------------------------------------------------------------------

def _decline_hint(tv_cfg: dict) -> str:
    """Build the decline-vs-rise editorial-line hint. Disabled when the
    operator turns prefer_decline_stories off in config."""
    if not tv_cfg.get("prefer_decline_stories", True):
        return ("No editorial preference set. Pick any story_kind that "
                "fits the topic.")
    return (
        "STRONGLY prefer decline-and-fall stories over rise-and-shine. "
        "The audience for business documentaries is overwhelmingly "
        "drawn to failures, scandals, and postmortems (WeWork, Theranos, "
        "Blockbuster, Pets.com, Enron, FTX, Wirecard, Lehman, "
        "Silicon Valley Bank). Of the seven story_kinds, prioritize "
        "in this order: scandal_postmortem > rise_and_fall > "
        "founder_drama > disruption > pivot > underdog_comeback > "
        "origin. An origin story is acceptable only if the founder's "
        "later arc is genuinely dramatic; otherwise pick a failure."
    )


def _non_us_hint(require: bool, tv_cfg: dict, recent_countries: list[str]) -> str:
    """Build the geographic-diversity hint. When required, instructs the
    LLM that this specific pick must be non-US. When not required, just
    nudges toward international diversity."""
    ratio = float(tv_cfg.get("non_us_ratio", 0.33))
    if require:
        return (
            f"REQUIRED for THIS pick: the company's hq_country must NOT "
            f"be 'US'. The recent rolling window has fallen below the "
            f"target non-US share ({ratio:.0%}). Propose a story whose "
            f"protagonist company was headquartered outside the United "
            f"States — UK (GB), Germany (DE), Japan (JP), South Korea "
            f"(KR), Sweden (SE), Netherlands (NL), France (FR), China "
            f"(CN), India (IN), Brazil (BR), Australia (AU), Canada "
            f"(CA), Israel (IL) — any of these are eligible. Examples: "
            f"Nokia (FI), Nortel (CA), Wirecard (DE), Toshiba (JP), "
            f"Saab (SE), Olivetti (IT), Air India (IN), TAM Linhas "
            f"Aéreas (BR), Carillion (GB), Steinhoff (ZA), Daewoo (KR), "
            f"Parmalat (IT), Bre-X Minerals (CA), Olympus (JP)."
        )
    return (
        f"No country requirement for this pick. The channel's target "
        f"non-US share is {ratio:.0%}; pick whichever country fits the "
        f"strongest story. Recent picks were from: "
        f"{', '.join(recent_countries) or '(none yet)'}."
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

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


def _recent_countries(rolling: dict, keep: int = 6) -> list[str]:
    """Last N HQ countries from the rolling-window state. Empty list
    until the first S01 run completes."""
    countries = (rolling.get("countries") or [])
    return countries[-keep:]


def _normalise_country(raw: str | None) -> str:
    """Best-effort normalisation of the LLM's hq_country output to a
    2-letter uppercase token. The prompt asks for ISO 3166-1 alpha-2,
    but LLMs sometimes emit names ('United States') or alpha-3 ('USA').
    A small alias table covers the common cases; anything else passes
    through as a 2-char upper-truncated token (or 'XX' if missing).
    """
    if not raw:
        return "XX"
    s = str(raw).strip().upper()
    # Common variants → ISO 3166-1 alpha-2. Covers full-name strings
    # the LLM sometimes emits despite the prompt asking for alpha-2,
    # plus the most-common alpha-3 codes (USA, GBR, DEU, JPN, etc.)
    # so a slip in the LLM's format doesn't degrade to "XX".
    aliases = {
        "USA": "US", "U.S.": "US", "U.S.A.": "US", "UNITED STATES": "US",
        "AMERICA": "US", "UNITED STATES OF AMERICA": "US",
        "UK": "GB", "U.K.": "GB", "GBR": "GB", "UNITED KINGDOM": "GB",
        "GREAT BRITAIN": "GB", "BRITAIN": "GB", "ENGLAND": "GB",
        "GERMANY": "DE", "DEU": "DE", "DEUTSCHLAND": "DE",
        "JAPAN": "JP", "JPN": "JP",
        "FRANCE": "FR", "FRA": "FR",
        "ITALY": "IT", "ITA": "IT",
        "SPAIN": "ES", "ESP": "ES",
        "NETHERLANDS": "NL", "NLD": "NL", "HOLLAND": "NL",
        "CHINA": "CN", "CHN": "CN",
        "INDIA": "IN", "IND": "IN",
        "BRAZIL": "BR", "BRA": "BR",
        "MEXICO": "MX", "MEX": "MX",
        "AUSTRALIA": "AU", "AUS": "AU",
        "CANADA": "CA", "CAN": "CA",
        "SWEDEN": "SE", "SWE": "SE",
        "SOUTH KOREA": "KR", "KOREA": "KR", "KOR": "KR",
        "ISRAEL": "IL", "ISR": "IL",
        "FINLAND": "FI", "FIN": "FI",
        "SWITZERLAND": "CH", "CHE": "CH",
        "RUSSIA": "RU", "RUS": "RU",
        "SOUTH AFRICA": "ZA", "ZAF": "ZA",
    }
    if s in aliases:
        return aliases[s]
    if len(s) == 2:
        return s
    return "XX"


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] or "unnamed"
