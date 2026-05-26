"""LLM adapter for Qwen3.6 (writer + extractor) and Gemma-4 (critic).

Calls the local OpenAI-compatible inference gateway at
http://10.0.4.250:9000/v1 with API key "pass123". The same gateway
hosts the VLM — see vlm.py for the image branch.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import requests

from .config import load_config

logger = logging.getLogger("hermes.llm")

LLM_BASE = "http://10.0.4.250:9000/v1"
LLM_API_KEY = "pass123"


ROLE_TO_CONFIG_KEY = {
    "writer": "llm_writer",
    "critic": "llm_critic",
    "extractor": "llm_extractor",
}


@dataclass
class LLMResult:
    text: str
    model: str
    finish_reason: str = "stop"


class LLM:
    """Thin adapter over the local OpenAI-compatible inference gateway."""

    def __init__(self, role: str = "writer"):
        if role not in ROLE_TO_CONFIG_KEY:
            raise ValueError(f"unknown LLM role: {role}")
        cfg = load_config()
        self.role = role
        self.model_name: str = cfg.models[ROLE_TO_CONFIG_KEY[role]]
        self.mock_mode: bool = cfg.mock_mode

    # ------------------ public API ------------------

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> LLMResult:
        if self.mock_mode:
            return self._mock(prompt, max_tokens)
        return self._invoke_model(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
            seed=seed,
        )

    def complete_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        retries: int = 2,
    ) -> Any:
        """Parse the output as JSON, retrying on parse failure with an
        explicit reminder appended to the prompt.

        Three layered recoveries:
          1. Strict json.loads.
          2. Repair pass (`_repair_json`) for common LLM breakages —
             trailing prose, missing commas between objects, smart
             quotes, trailing commas, AND truncated arrays/objects
             (whichever complete top-level elements were emitted
             before the model ran out of token budget).
          3. Retry with an explicit reminder ONLY when the failure
             was NOT a truncation. If the model hit max_tokens
             (`finish_reason=='length'`) it will hit the same wall
             on the retry, so we skip straight to repair-or-give-up.
        """
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            res = self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
            text = _strip_code_fences(res.text)
            truncated = (res.finish_reason or "").lower() == "length"

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                # Repair pass — handles truncated output too.
                repaired = _repair_json(text)
                if repaired is not None and repaired != text:
                    try:
                        result = json.loads(repaired)
                        logger.info(
                            "JSON parse succeeded after repair (attempt %d%s)",
                            attempt, ", truncated" if truncated else "",
                        )
                        return result
                    except json.JSONDecodeError:
                        pass

                last_err = e
                if truncated:
                    # Don't burn a retry — the model will hit the
                    # token cap again with the same prompt. Bail
                    # immediately so the caller can downsize input.
                    logger.warning(
                        "LLM output truncated at max_tokens=%d "
                        "(finish_reason=length, %d chars emitted, "
                        "parse failed: %s) — skipping retries",
                        max_tokens, len(text), e,
                    )
                    break

                logger.warning("JSON parse failed (attempt %d): %s", attempt, e)
                prompt = (
                    prompt
                    + "\n\nIMPORTANT: Output ONLY valid JSON. No prose. No code fences.\n"
                    + f"Previous output was not valid JSON ({e})."
                )
        raise ValueError(f"could not parse JSON after {retries} retries: {last_err}")

    def complete_chunked(
        self,
        prompt_prefix: str,
        chunks: list[str],
        *,
        chunk_label: str = "CHUNK",
        max_tokens_per_chunk: int = 2048,
        temperature: float = 0.7,
    ) -> list[LLMResult]:
        """One call per chunk; used by S3 fact extraction over long sources."""
        results = []
        for i, ch in enumerate(chunks, start=1):
            full = f"{prompt_prefix}\n\n{chunk_label} {i} of {len(chunks)}:\n{ch}"
            results.append(
                self.complete(full, temperature=temperature,
                              max_tokens=max_tokens_per_chunk)
            )
        return results

    # ------------------ implementation hooks ------------------

    def _invoke_model(
        self,
        *,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: list[str],
        seed: int | None,
    ) -> LLMResult:
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        r = requests.post(
            f"{LLM_BASE}/chat/completions",
            json=payload, headers=headers, timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", "stop")
        return LLMResult(text=text, model=self.model_name, finish_reason=finish_reason)

    def _mock(self, prompt: str, max_tokens: int) -> LLMResult:
        """Deterministic mock for end-to-end pipeline testing."""
        h = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        pl = prompt.lower()

        # ----- ARRAY-shaped requests (S3 fact extract, S4 merge, S8 beats) -----
        if "json array" in pl or "array only" in pl:
            # S3 fact_extract.txt asks for an array of fact objects.
            if "fact_text" in pl or "fact_type" in pl:
                facts = [
                    {"fact_text": "Acme Corp was incorporated in Delaware in 1998.",
                     "fact_type": "founding_date",
                     "exact_quote": "incorporated in Delaware on March 12, 1998",
                     "source_paragraph_hint": "paragraph 1",
                     "confidence": "high",
                     "hq_location": {"city": "Palo Alto",
                                     "state_or_region": "California",
                                     "country": "United States"}},
                    {"fact_text": "Jordan Lee founded Acme Corp in a converted Palo Alto garage.",
                     "fact_type": "founder",
                     "exact_quote": "founded by Jordan Lee",
                     "source_paragraph_hint": "paragraph 1",
                     "confidence": "high"},
                    {"fact_text": "Acme Corp pivoted from enterprise software to consumer subscriptions in 2003.",
                     "fact_type": "pivotal_decision",
                     "exact_quote": "pivoted from enterprise software",
                     "source_paragraph_hint": "paragraph 2",
                     "confidence": "medium"},
                ]
                return LLMResult(text=json.dumps(facts),
                                 model=self.model_name + "-mock")
            # S4 fact_merge.txt asks for an array of claim objects.
            if "claim_id" in pl or "canonical_statement" in pl:
                claims = [
                    {"claim_id": "C001",
                     "canonical_statement": "Acme Corp was incorporated in Delaware in 1998.",
                     "fact_type": "founding_date",
                     "supporting_facts": [0],
                     "contradicting_facts": [],
                     "strongest_confidence": "high",
                     "tiers_supporting": ["open_tier1"]},
                    {"claim_id": "C002",
                     "canonical_statement": "Jordan Lee founded Acme Corp in Palo Alto.",
                     "fact_type": "founder",
                     "supporting_facts": [1],
                     "contradicting_facts": [],
                     "strongest_confidence": "high",
                     "tiers_supporting": ["open_tier1"]},
                ]
                return LLMResult(text=json.dumps(claims),
                                 model=self.model_name + "-mock")
            # S8 beat_sheet.txt asks for an array of beat plans.
            if "beat_id" in pl or "ken_burns_motion" in pl:
                # Need at least min_total_beats; in smoke mode that's 1.
                # Produce enough that real config also works (50+).
                beats = []
                for i in range(1, 6):
                    bid = f"BEAT_{i:02d}"
                    beats.append({
                        "beat_id": bid,
                        "estimated_seconds": 12.0,
                        "visual_intent": "office_environment",
                        "specific_visual_description": (
                            "wide cinematic comic panel of a converted "
                            "Palo Alto garage, golden hour light, single "
                            "founder silhouette at a workbench"
                        ),
                        "ken_burns_motion": "slow_zoom_in",
                        "public_domain_search_terms": ["palo alto garage", "1990s tech founder"],
                        "flux_fallback_prompt": (
                            "a founder at a workbench in a converted "
                            "garage, vintage computer monitor on the desk, "
                            "warm light"
                        ),
                        "sfx_cue": None,
                    })
                return LLMResult(text=json.dumps(beats),
                                 model=self.model_name + "-mock")
            return LLMResult(text="[]", model=self.model_name + "-mock")

        # ----- OBJECT-shaped requests (S1 topic, S3 HQ, S4 verify, S7 critique) -----
        if "json" in pl and ("output" in pl or "format" in pl):
            return LLMResult(
                text=json.dumps({
                    "_mock": True,
                    "_hash": h,
                    "_role": self.role,
                    # S1 topic-discovery shape
                    "company_name": "Acme Corp",
                    "founder_or_protagonist": "Jordan Lee",
                    "year_anchor": 1998,
                    "story_kind": "rise_and_fall",
                    "hq_country": "US",
                    "one_line_pitch": "Mock business story for pipeline testing.",
                    "hero": "Jordan Lee, the relentless outsider",
                    "conflict": "An incumbent giant and a market that wasn't ready",
                    "why_this_one": "Generated by mock_mode for end-to-end runs.",
                    "predicted_archetype_fit": "A2",
                    "estimated_source_quality": "medium",
                    "demonetization_risk_notes": "none — synthetic test data",
                    # S3 fact-extraction shape (catch-all)
                    "facts": [],
                    "fact_type": "founding_date",
                    "fact_text": "Acme Corp was incorporated in Delaware in 1998.",
                    "exact_quote": "incorporated in Delaware on March 12, 1998",
                    "confidence": "medium",
                    # S3 HQ-consolidation shape
                    "city": "Palo Alto",
                    "state_or_region": "California",
                    "country": "United States",
                    "method": "headquarters_announcement",
                    "supporting_statements": [],
                    # S4 verification shape
                    "verdict": "pass",
                    "reasoning": "Mock verification — synthetic data.",
                    # S7 critique shape
                    "rewrites": [],
                    # S8 beat sheet shape (catch-all)
                    "beats": [],
                    # S05 character iconography fallback
                    "iconography": (
                        "Long dark wavy hair past the shoulders, paired with "
                        "a casual black t-shirt and faded jeans. Bohemian, "
                        "loose-limbed posture — one hand often pressed to the "
                        "chest while speaking. Carries an old leather notebook "
                        "everywhere. Strikes a tall, expansive stance even in "
                        "small rooms; gestures wide. Always wears scuffed "
                        "boots, never sneakers."
                    ),
                }),
                model=self.model_name + "-mock",
            )

        # Prose responses: return a 2000-ish-word business script for
        # script-generation stages, beat-marker-tagged so S8 can parse.
        return LLMResult(
            text=(
                f"[MOCK {self.role} {h}]\n\n"
                "Acme Corp began in a converted garage in Palo Alto. ## BEAT 1 ##\n\n"
                "Jordan Lee, twenty-six and out of money, had two months of runway "
                "and a single prototype. ## BEAT 2 ##\n\n"
                "The incumbent — Globex — controlled ninety-three percent of the "
                "market and had no reason to notice. ## BEAT 3 ##\n\n"
                "What follows is a record of decisions, of timing, and of a market "
                "that — for reasons even the inquiry would only partially explain — "
                "tipped first slowly and then all at once. ## BEAT 4 ##\n"
            ),
            model=self.model_name + "-mock",
        )


# ------------------ helpers ------------------

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _repair_json(text: str) -> str | None:
    r"""Best-effort JSON repair for common LLM breakages.

    Targets the failure modes we have observed in production:

    1. Trailing prose after the JSON: the LLM closes the array/object
       cleanly, then keeps writing English ("And those are the merged
       claims!") — json.loads rejects the whole thing. We trim back
       to the last balanced `}` or `]`.

    2. Missing comma between adjacent objects in an array
       (``{...}\n  {...}`` instead of ``{...},\n  {...}``) — the
       "Expecting ',' delimiter" error we have hit at S4. We
       substitute ``}\s*{`` with ``},{`` and ``]\s*\[`` with ``],[``.

    3. Smart-quote injection (curly quotes) where the model should
       have emitted ASCII quotes. We normalize.

    4. Trailing comma before ``]`` or ``}`` — Python's parser is
       strict; we strip these.

    Returns the repaired string (which may or may not parse — caller
    re-tries json.loads on it). Returns None only if there is no
    plausible JSON structure to repair (no opening brace/bracket).
    """
    s = _strip_code_fences(text or "")
    if not s:
        return None

    # Find first opening brace / bracket and trim leading prose.
    first_obj = s.find("{")
    first_arr = s.find("[")
    candidates = [x for x in (first_obj, first_arr) if x >= 0]
    if not candidates:
        return None
    start = min(candidates)
    s = s[start:]
    outer_open = s[0]   # '{' or '['
    outer_close = "]" if outer_open == "[" else "}"

    # Walk the string tracking bracket depth. Record TWO things:
    #   - last_balanced: where the outer container fully closes
    #     (depth returns to zero). Used to trim trailing prose.
    #   - last_inner_close: the latest position where we closed a
    #     direct child of the outer container (depth went from
    #     2 to 1). Used to salvage TRUNCATED output — if the outer
    #     container never closes, we keep everything up to and
    #     including this position and append the missing closer.
    opens = "{["
    closes_for = {"}": "{", "]": "["}
    stack: list[str] = []
    in_string = False
    escape = False
    last_balanced = -1
    last_inner_close = -1
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in opens:
            stack.append(ch)
        elif ch in closes_for:
            if stack and stack[-1] == closes_for[ch]:
                stack.pop()
                if len(stack) == 1:
                    # Just closed a top-level child of the outer
                    # container. Safe truncation point.
                    last_inner_close = i
                elif not stack:
                    last_balanced = i

    if last_balanced >= 0:
        # Outer container closed cleanly somewhere; trim trailing prose.
        s = s[: last_balanced + 1]
    elif last_inner_close >= 0:
        # Outer container NEVER closed — output was truncated mid-array
        # (the actual production failure on Gemma-4 + S04 merge).
        # Salvage the complete top-level elements emitted before the
        # truncation point and close the outer container ourselves.
        # Drop everything after the last completed inner element
        # (which is usually a partial child that will not parse).
        salvaged = s[: last_inner_close + 1].rstrip().rstrip(",")
        s = salvaged + outer_close
    # else: no balanced inner element either — fall through to
    # comma/quote repairs, which probably won't save it.

    # Smart-quote normalization (string contents may legitimately
    # contain unicode quotes, but they break parsing only when they
    # stand in for the JSON delimiters themselves — that's almost
    # always at boundaries where ASCII " is required).
    s = (
        s.replace("“", '"').replace("”", '"')
         .replace("‘", "'").replace("’", "'")
    )

    # Missing-comma between adjacent objects / arrays.
    s = re.sub(r"\}\s*\{", "},{", s)
    s = re.sub(r"\]\s*\[", "],[", s)

    # Trailing commas before closers.
    s = re.sub(r",\s*([\]\}])", r"\1", s)

    return s


# Module-level imports for the repair helper (re isn't otherwise used
# in this module, so the import sits below the code that needs it).
import re  # noqa: E402
