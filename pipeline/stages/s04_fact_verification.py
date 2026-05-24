"""S04 — Fact Verification (merge + adversarial pass).

1. Use the critic LLM (Gemma-4) to group raw facts into deduplicated
   claims (the merge prompt).
2. For each merged claim, run an adversarial skeptic prompt (writer
   LLM as adversarial reviewer) to decide pass / reject /
   needs_browser_check / soft.
3. For `needs_browser_check`, hit the browser for independent
   corroboration; require an open_tier1 / open_tier2 hit to upgrade.

Inputs:  01_factcheck/raw_facts.json
Outputs: 01_factcheck/fact_ledger.json  (claims usable downstream)
"""

from __future__ import annotations

import json
import logging

from ..browser import Browser
from ..config import load_config
from ..llm import LLM
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s04")


# Mirrors the OPEN_TIER1 + OPEN_TIER2 sets in S02 but as a flat
# substring list — used to decide whether a corroborating web hit is
# strong enough to upgrade a "needs_browser_check" claim.
_AUTHORITATIVE_NEEDLES = (
    ".gov", ".mil", ".edu",
    "sec.gov", "courtlistener.com", "justice.gov", "govinfo.gov",
    "companieshouse.gov.uk",
    "wikipedia.org", "archive.org",
    "apnews.com", "reuters.com", "npr.org", "bbc.co.uk", "bbc.com",
    "pbs.org", "propublica.org",
    "theguardian.com", "theatlantic.com", "newyorker.com",
    "techcrunch.com", "arstechnica.com", "wired.com", "theverge.com",
)


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    critic = LLM(role="critic")
    skeptic = LLM(role="writer")
    browser = Browser()

    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    raw_path = ws / "01_factcheck" / "raw_facts.json"
    if not raw_path.exists():
        return "no raw_facts.json"
    raw_facts = json.loads(raw_path.read_text())["facts"]
    if not raw_facts:
        return "no raw facts to verify"

    incident_name = episode["incident"]["company_name"]

    # ---------- 1. merge (chunked) ----------
    # Run the merge prompt in batches of MERGE_CHUNK facts. Smaller
    # output per call dramatically reduces the rate of JSON-output
    # breakage we hit in production on Gemma-4 with 300+ facts (the
    # parser failed deep into a 12 KB response, around line 303).
    # Per-chunk failure is non-fatal: chunks that succeed contribute,
    # chunks that fail are logged and skipped.
    merge_template = (cfg.prompts_dir / "fact_merge.txt").read_text()
    MERGE_CHUNK = 80
    facts_for_merge = raw_facts[:300]
    merged = _merge_in_chunks(
        critic=critic,
        merge_template=merge_template,
        incident_name=incident_name,
        chunk=facts_for_merge,
        chunk_size=MERGE_CHUNK,
    )
    if not merged:
        return "merge produced zero usable claims (all chunks failed?)"

    # ---------- 2. adversarial verify ----------
    verify_template = (cfg.prompts_dir / "fact_verify.txt").read_text()
    verified: list[dict] = []

    for claim in merged:
        supporting_idx = claim.get("supporting_facts") or []
        if not isinstance(supporting_idx, list) or not supporting_idx:
            logger.info("dropped (no supporting_facts): %s",
                        (claim.get("canonical_statement") or "")[:80])
            continue

        # Build a small per-claim source-excerpt block so the skeptic
        # can see the underlying evidence.
        excerpts = []
        tiers_supporting = set()
        for idx in supporting_idx[:6]:
            if not isinstance(idx, int) or idx < 0 or idx >= len(raw_facts):
                continue
            f = raw_facts[idx]
            tier = f.get("source_tier") or "open_tier2"
            tiers_supporting.add(tier)
            excerpts.append(
                f"[{f.get('source_id','?')} tier={tier} pub={f.get('source_publisher','?')}]: "
                f"{f.get('fact_text','')[:240]}"
            )

        prompt = verify_template.format(
            incident_name=incident_name,
            claim_json=json.dumps({
                "claim_id": claim.get("claim_id"),
                "canonical_statement": claim.get("canonical_statement"),
                "fact_type": claim.get("fact_type"),
                "strongest_confidence": claim.get("strongest_confidence"),
                "tiers_supporting": sorted(tiers_supporting),
            }, indent=2),
            supporting_sources="\n".join(excerpts) or "(none)",
        )
        try:
            verdict_obj = skeptic.complete_json(
                prompt, temperature=0.4, max_tokens=600
            )
        except Exception as e:
            logger.warning("verify parse failed: %s", e)
            continue

        verdict = (verdict_obj.get("verdict") or "").lower()
        statement = (claim.get("canonical_statement") or "")[:80]

        if verdict == "pass":
            claim["verification"] = verdict_obj
            verified.append(claim)
        elif verdict == "soft":
            # Soft claims still flow downstream but with a flag so the
            # script writer can phrase them as opinion ("biographers
            # have suggested...").
            claim["verification"] = verdict_obj
            claim["soft"] = True
            verified.append(claim)
        elif verdict == "needs_browser_check":
            concerns = verdict_obj.get("suggested_search_terms") or []
            corroborated = False
            for q in concerns[:3]:
                q_full = f'"{q}" "{incident_name}"' if q else f'"{incident_name}"'
                try:
                    results = browser.search(q_full, n_results=5)
                except Exception:
                    results = []
                if any(_looks_like_authoritative(r.url) for r in results):
                    corroborated = True
                    break
            if corroborated:
                claim["verification"] = verdict_obj
                claim["verification"]["web_corroborated"] = True
                verified.append(claim)
            else:
                logger.info("dropped needs_browser_check (no corroboration): %s",
                            statement)
        elif verdict == "reject":
            reason = (verdict_obj.get("reasoning") or "no reason")[:120]
            logger.info("rejected by skeptic (%s): %s", reason, statement)
        else:
            logger.warning("unknown verdict %r; dropping: %s", verdict, statement)

    # ---------- 3. write ledger ----------
    ledger = {"incident": incident_name, "claims": verified}
    (ws / "01_factcheck" / "fact_ledger.json").write_text(
        json.dumps(ledger, indent=2)
    )

    min_facts = cfg.quality_gates["min_verified_facts"]
    logger.info("S04 complete: %d verified claims (need %d)",
                len(verified), min_facts)
    if len(verified) < min_facts:
        return (f"only {len(verified)} verified claims (need {min_facts}); "
                f"company lacks corroborated detail to safely write a script")
    return None


def _looks_like_authoritative(url: str) -> bool:
    u = (url or "").lower()
    return any(n in u for n in _AUTHORITATIVE_NEEDLES)


def _merge_in_chunks(
    *,
    critic,
    merge_template: str,
    incident_name: str,
    chunk: list[dict],
    chunk_size: int = 80,
) -> list[dict]:
    """Run the critic's merge prompt against `chunk` in slices of
    `chunk_size`, concatenating the resulting claim lists.

    Per-chunk transformations:
      - `supporting_facts` and `contradicting_facts` indices emitted
        by the LLM are local to the chunk it saw (0..chunk_size-1).
        We translate them back to absolute indices into the full
        raw_facts list by adding the chunk's start offset.
      - `claim_id` values are renumbered C001, C002, ... globally so
        downstream code can rely on sequential IDs without colliding
        across chunks.

    A chunk whose JSON output cannot be parsed (even after the repair
    pass in `complete_json`) is logged and skipped — the remaining
    chunks still contribute claims.
    """
    out: list[dict] = []
    next_claim_id = 1
    total_chunks = (len(chunk) + chunk_size - 1) // chunk_size
    for batch_idx, start in enumerate(range(0, len(chunk), chunk_size)):
        batch = chunk[start:start + chunk_size]
        prompt = merge_template.format(
            incident_name=incident_name,
            raw_facts_json=json.dumps(batch, indent=2),
        )
        logger.info(
            "S04 merge: batch %d/%d (facts %d..%d)",
            batch_idx + 1, total_chunks, start, start + len(batch) - 1,
        )
        try:
            batch_claims = critic.complete_json(
                prompt, temperature=0.3, max_tokens=4000,
            )
        except Exception as e:
            logger.warning("S04 merge batch %d failed: %s", batch_idx + 1, e)
            continue
        if not isinstance(batch_claims, list):
            logger.warning(
                "S04 merge batch %d returned non-list (%s); skipping",
                batch_idx + 1, type(batch_claims).__name__,
            )
            continue
        for claim in batch_claims:
            if not isinstance(claim, dict):
                continue
            # Re-base supporting / contradicting indices to absolute.
            supp = claim.get("supporting_facts") or []
            claim["supporting_facts"] = [
                idx + start
                for idx in supp
                if isinstance(idx, int) and 0 <= idx < len(batch)
            ]
            contra = claim.get("contradicting_facts") or []
            claim["contradicting_facts"] = [
                idx + start
                for idx in contra
                if isinstance(idx, int) and 0 <= idx < len(batch)
            ]
            # Renumber claim_id globally so downstream consumers see
            # a stable C001..CNNN sequence with no gaps.
            claim["claim_id"] = f"C{next_claim_id:03d}"
            next_claim_id += 1
            out.append(claim)
    logger.info("S04 merge: %d total claims from %d batches",
                len(out), total_chunks)
    return out
