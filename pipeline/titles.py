"""Title variant generator (Batch D 2026-05-27).

Generates N (default 10) candidate YouTube titles per episode using
the writer LLM, each tagged with a style hypothesis (curiosity_gap,
shock_value, outcome_first, named_person, question, contrarian,
number_anchored, before_after, time_anchored, character_voice).

Output goes to 06_metadata/titles.json with the structure:
  {
    "variants": [
      {
        "rank": 1,
        "text": "How a $42 late fee built a $200B company",
        "style_hypothesis": "curiosity_gap",
        "predicted_ctr_band": "high",
        "rationale": "specific number + outsized outcome"
      },
      ...
    ]
  }

The operator picks top 3 manually OR uses YouTube's native title-test
API to A/B-test them post-publish.

Real channels iterate. The single biggest discoverable CTR lever is
testing 5-10+ title variants vs. publishing the LLM's first pick.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .llm import LLM

logger = logging.getLogger("hermes.titles")


@dataclass
class TitleVariant:
    rank: int
    text: str
    style_hypothesis: str
    predicted_ctr_band: str
    rationale: str


def generate_variants(
    incident: dict[str, Any],
    beat_sheet: dict[str, Any] | None = None,
    *,
    n: int = 10,
) -> list[TitleVariant]:
    """Ask the writer LLM for `n` title variants. Returns ordered by
    the LLM's confidence (highest first). On any failure returns an
    empty list — caller (S13) logs and proceeds."""
    cfg = load_config()
    llm = LLM(role="writer")

    template_path = cfg.prompts_dir / "title_variants.txt"
    if not template_path.exists():
        logger.warning("title_variants.txt missing; skipping title generation")
        return []
    template = template_path.read_text()

    # Pull the dramatic-moment beat summaries to give the LLM
    # concrete numbers it can anchor titles on.
    beat_summary = ""
    if beat_sheet:
        beats = beat_sheet.get("beats", [])
        snippets = []
        for b in beats[:8]:
            sk = (b.get("script_text") or "")[:150]
            if sk:
                snippets.append(f"- {sk.strip()}")
        beat_summary = "\n".join(snippets)

    prompt = template.format(
        n=n,
        company_name=incident.get("company_name", ""),
        founder=incident.get("founder_or_protagonist", ""),
        year_anchor=incident.get("year_anchor", ""),
        story_kind=incident.get("story_kind", ""),
        hero=incident.get("hero", ""),
        conflict=incident.get("conflict", ""),
        one_line_pitch=incident.get("one_line_pitch", ""),
        beat_summary=beat_summary or "(no beat-summary available)",
    )

    try:
        result = llm.complete_json(prompt, temperature=0.85, max_tokens=4000)
    except Exception as e:
        logger.warning("title variants JSON parse failed: %s", e)
        return []

    variants_raw = result.get("variants") or result.get("titles") or []
    out: list[TitleVariant] = []
    for i, v in enumerate(variants_raw[:n], start=1):
        text = (v.get("text") or v.get("title") or "").strip()
        if not text:
            continue
        out.append(TitleVariant(
            rank=i,
            text=text,
            style_hypothesis=(v.get("style_hypothesis")
                              or v.get("style") or "unknown"),
            predicted_ctr_band=(v.get("predicted_ctr_band") or "medium"),
            rationale=(v.get("rationale") or v.get("reason") or "")[:240],
        ))
    return out


def write_variants(variants: list[TitleVariant], out_path: Path) -> None:
    """Persist to 06_metadata/titles.json. Always writes a file, even
    when the variants list is empty, so the operator sees the
    artifact slot regardless."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variants": [
            {
                "rank": v.rank,
                "text": v.text,
                "style_hypothesis": v.style_hypothesis,
                "predicted_ctr_band": v.predicted_ctr_band,
                "rationale": v.rationale,
            }
            for v in variants
        ],
        "count": len(variants),
    }
    out_path.write_text(json.dumps(payload, indent=2))
