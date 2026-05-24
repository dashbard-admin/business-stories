"""S03 — Fact Extraction.

For each source, chunk-feed it to the extractor LLM with the
fact-extract prompt. Produces a raw facts file per source. Then runs
a second LLM pass over `location_founded` type facts to derive a
single best-estimate company HQ (city / state / country) — analogous
to the maritime pipeline's incident_location.json but with no need for
latitude/longitude precision.

Inputs:  00_research/source_inventory.json + per-source extracted text
Outputs: 01_factcheck/raw_facts.json  (all facts, attributed)
         01_factcheck/company_profile.json  (HQ consolidation)
"""

from __future__ import annotations

import json
import logging

from ..config import load_config
from ..llm import LLM
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s03")

CHUNK_WORDS = 4000


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    llm = LLM(role="extractor")
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    inv_path = ws / "00_research" / "source_inventory.json"
    if not inv_path.exists():
        return "no source inventory; S02 must run first"
    inventory = json.loads(inv_path.read_text())["sources"]

    template = (cfg.prompts_dir / "fact_extract.txt").read_text()

    all_facts: list[dict] = []
    incident_name = episode["incident"]["company_name"]

    for src in inventory:
        src_path = ws / src["local_path"]
        if not src_path.exists():
            logger.warning("source text missing: %s", src_path)
            continue
        text = src_path.read_text()

        # For paywall_title_only sources, the body is just a title +
        # snippet stub — feed it as one short chunk and let the
        # extractor pick at most one low-confidence fact.
        chunks = _chunk_text(text, CHUNK_WORDS)
        logger.info("extracting from %s (%d chunks, tier=%s)",
                    src["id"], len(chunks), src["tier"])

        for ci, chunk in enumerate(chunks):
            prompt_prefix = template.format(
                incident_name=incident_name,
                publisher=src.get("publisher", ""),
                tier=src.get("tier", "open_tier2"),
                source_date=src.get("date") or "unknown",
            )
            full_prompt = (
                f"{prompt_prefix}\n\nSOURCE TEXT CHUNK {ci+1} of {len(chunks)}:\n{chunk}"
            )
            try:
                facts = llm.complete_json(
                    full_prompt, temperature=0.3, max_tokens=3000
                )
            except Exception as e:
                logger.warning("extraction failed for %s chunk %d: %s",
                               src["id"], ci, e)
                continue

            if not isinstance(facts, list):
                logger.warning("non-array fact output from %s chunk %d",
                               src["id"], ci)
                continue

            for f in facts:
                if not isinstance(f, dict) or "fact_text" not in f:
                    continue
                f["source_id"] = src["id"]
                f["source_tier"] = src["tier"]
                f["source_publisher"] = src.get("publisher")
                all_facts.append(f)

    (ws / "01_factcheck").mkdir(exist_ok=True)
    (ws / "01_factcheck" / "raw_facts.json").write_text(
        json.dumps({"facts": all_facts}, indent=2)
    )
    logger.info("S03 complete: %d raw facts from %d sources",
                len(all_facts), len(inventory))

    # ---- Company HQ consolidation ----
    try:
        hq = _consolidate_hq(llm=llm, cfg=cfg, incident_name=incident_name,
                             facts=all_facts)
    except Exception as e:
        logger.warning("HQ consolidation raised: %s", e)
        hq = None
    if hq is not None:
        (ws / "01_factcheck" / "company_profile.json").write_text(
            json.dumps(hq, indent=2)
        )
        logger.info("S03 HQ: %s, %s, %s (conf=%s, method=%s)",
                    hq.get("city"), hq.get("state_or_region"),
                    hq.get("country"), hq.get("confidence"),
                    hq.get("method"))
    else:
        logger.info("S03 HQ: no consolidation result")

    min_facts_target = 20  # tuned lower than maritime — fewer claims per business story
    if len(all_facts) < min_facts_target:
        return (f"only {len(all_facts)} raw facts extracted "
                f"(need {min_facts_target}+) — increase source coverage")
    return None


def _consolidate_hq(*, llm, cfg, incident_name: str,
                    facts: list[dict]) -> dict | None:
    """Derive a single best-estimate HQ from facts of type
    `location_founded` or facts whose `hq_location` field is set.

    Two-step:
      1. If any fact has an explicit `hq_location` dict, return its
         contents directly (method="explicit_from_source").
      2. Otherwise gather location-founded facts and ask the LLM to
         consolidate via the company_hq_consolidate prompt.
    """
    # Step 1: explicit hq_location field.
    for f in facts:
        hq = f.get("hq_location")
        if isinstance(hq, dict) and (hq.get("city") or hq.get("country")):
            return {
                "city": (hq.get("city") or "").strip(),
                "state_or_region": (hq.get("state_or_region") or "").strip(),
                "country": (hq.get("country") or "").strip(),
                "confidence": "high",
                "method": "explicit_from_source",
                "source_id": f.get("source_id"),
                "supporting_statements": [
                    f.get("exact_quote") or f.get("fact_text", "")
                ],
            }

    # Step 2: LLM consolidation across location-founded facts.
    loc_facts = [
        f for f in facts
        if f.get("fact_type") in ("location_founded",) and (
            f.get("fact_text") or f.get("exact_quote")
        )
    ]
    if not loc_facts:
        return None

    lines: list[str] = []
    for f in loc_facts[:30]:
        txt = (f.get("fact_text") or f.get("exact_quote") or "").strip()
        if not txt:
            continue
        publisher = f.get("source_publisher") or f.get("source_id") or "unknown"
        tier = f.get("source_tier", "open_tier2")
        lines.append(f'- "{txt}" (source: {publisher}, tier {tier})')
    if not lines:
        return None
    statements_block = "\n".join(lines)

    try:
        template = (cfg.prompts_dir / "company_hq_consolidate.txt").read_text()
    except FileNotFoundError:
        logger.warning("company_hq_consolidate.txt prompt missing; skipping")
        return None

    full_prompt = template.format(
        incident_name=incident_name,
        statements=statements_block,
    )
    logger.info("S03 HQ: consolidating %d location-founded facts via LLM ...",
                len(lines))

    try:
        result = llm.complete_json(full_prompt, temperature=0.2, max_tokens=400)
    except Exception as e:
        logger.warning("HQ consolidation LLM call failed: %s", e)
        return None

    if not isinstance(result, dict):
        logger.warning("HQ consolidation: non-dict result %r", result)
        return None

    city = (result.get("city") or "").strip()
    country = (result.get("country") or "").strip()
    if not city and not country:
        return None

    return {
        "city": city,
        "state_or_region": (result.get("state_or_region") or "").strip(),
        "country": country,
        "confidence": (result.get("confidence") or "low").lower(),
        "method": result.get("method") or "llm_consolidated",
        "supporting_statements": result.get("supporting_statements") or [],
    }


def _chunk_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    out = []
    for i in range(0, len(words), max_words):
        out.append(" ".join(words[i:i + max_words]))
    return out
