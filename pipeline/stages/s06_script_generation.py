"""S06 — Script Generation.

Writes a ~2000-word business-story script using the locked
archetype/narrator/style and the verified fact ledger. Validates word
count + BEAT marker count before passing on. Runs the
forbidden-phrase lint with a single rewrite attempt on hit.

Inputs:  01_factcheck/fact_ledger.json  +  episode assignment
Outputs: 02_script/script.txt  +  02_script/script_meta.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from ..config import load_config
from ..llm import LLM
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s06")

BEAT_RE = re.compile(r"##\s*BEAT\s+(\d+)\s*##", re.IGNORECASE)
THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# Strip Kokoro-poisoning terminal sign-offs ("End of script", "THE END",
# "Fin.", "---END---") from the script tail.
TERMINAL_BOILERPLATE_RE = re.compile(
    r"""(?x)
    (?<=[.!?])
    [\s\n]*
    (?:
        [\-*=\[\(]+ \s* (?i:end) \s* [\-*=\]\)]+
        |
        (?i: end \s+ of \s+
             (?: script | narration | report | episode | story | transcript | text )
        )
        |
        (?i: the \s+ end )
        |
        (?i: fin )
    )
    [\s\-*=\.\!\?\]\)]*
    \s*$
    """,
)


def _clean(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    for _ in range(4):
        new_text = TERMINAL_BOILERPLATE_RE.sub("", text).rstrip()
        if new_text == text:
            break
        text = new_text
    return text


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    llm = LLM(role="writer")
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    ledger_path = ws / "01_factcheck" / "fact_ledger.json"
    if not ledger_path.exists():
        return "no fact ledger"
    ledger = json.loads(ledger_path.read_text())

    archetype = episode["archetype"]
    narrator = episode["narrator"]
    visual_style = episode["visual_style"]
    incident = episode["incident"]

    archetypes = yaml.safe_load(
        (cfg.style_profiles_dir / "archetypes.yaml").read_text()
    )
    narrators = yaml.safe_load(
        (cfg.style_profiles_dir / "narrators.yaml").read_text()
    )
    style_yaml = yaml.safe_load(
        (cfg.style_profiles_dir / f"{visual_style}.yaml").read_text()
    )

    arch = archetypes[archetype]
    arch_guidance = (
        f"Opening: {arch['opening_device']}\n"
        f"Middle: {arch['middle_structure']}\n"
        f"Closing: {arch['closing_device']}"
    )

    narr = narrators[narrator]
    narr_cfg = cfg.narrator_by_id(narrator)

    template = (cfg.prompts_dir / "script_generate.txt").read_text()

    target_words = cfg.production["target_words"]
    target_beats = (cfg.quality_gates["min_total_beats"]
                    + cfg.quality_gates["max_total_beats"]) // 2

    # Preview-mode short-circuit (Batch B 2026-05-26). Operator
    # invoked --preview when queueing this episode; render only
    # Act 0 + Act 5 so the operator can sanity-check tone, voice,
    # visual style, hook, and closing without burning the full
    # 3-4hr compute. The override is implemented as:
    #   1. Tight word window (~280-480 words).
    #   2. Small target_beats (~8).
    #   3. A prepended directive in the prompt telling the LLM to
    #      skip Acts 1, 2, 3, 3.5, 4 entirely.
    preview_mode = bool(episode.get("preview_mode"))
    preview_directive = ""
    if preview_mode:
        target_words = 360
        target_beats = 8
        preview_directive = (
            ">>> PREVIEW MODE — RENDER ONLY ACT 0 AND ACT 5 <<<\n"
            "This run is a tone-check, not a publishable episode. "
            "Skip Acts 1, 2, 3, 3.5, and 4 ENTIRELY. Generate ONLY:\n"
            "  - Act 0 (the hook, ~60 words, 1-2 beats)\n"
            "  - Act 5 (the closing, ~300 words, 5-6 beats)\n"
            "Place a literal line `## BEAT 2 ## [PREVIEW: Acts 1-4 "
            "skipped]` between Act 0's last beat and Act 5's first "
            "beat so the operator can see where the gap is. Total "
            "word count 280-480; total beat count 6-10. The hard-"
            "length-gate rules below are RELAXED for this mode — "
            "ignore them.\n\n"
        )
        logger.info("S06 preview-mode: targeting %d words / %d beats",
                    target_words, target_beats)

    # Load character iconography if S05's profile sub-step produced it.
    # Missing file is fine — the writer falls back to a neutral
    # placeholder and the prompt instructs it to plant visual cues
    # only when iconography is available.
    character_iconography = "(not available)"
    cp_path = ws / "01_factcheck" / "character_profile.json"
    if cp_path.exists():
        try:
            cp = json.loads(cp_path.read_text())
            icon = (cp.get("iconography") or "").strip()
            if icon:
                character_iconography = icon
        except Exception as e:
            logger.warning("character_profile.json unreadable: %s", e)

    min_w = cfg.quality_gates["min_script_words"]
    max_w = cfg.quality_gates["max_script_words"]
    if preview_mode:
        min_w = 280
        max_w = 480

    prompt = template.format(
        preview_mode_directive=preview_directive,
        incident_name=incident["company_name"],
        year=incident.get("year_anchor"),
        hero=incident.get("hero", ""),
        conflict=incident.get("conflict", ""),
        story_kind=incident.get("story_kind", ""),
        target_words=target_words,
        min_words=min_w,
        max_words=max_w,
        archetype_name=arch["name"],
        archetype_guidance=arch_guidance,
        narrator_name=narr["name"],
        narrator_tone=narr_cfg["tone"],
        narrator_id=narrator,
        narrator_full_instructions=narr["full_instructions"],
        visual_style_name=style_yaml["name"],
        character_iconography=character_iconography,
        fact_ledger_json=json.dumps(
            [{"id": c.get("claim_id"),
              "fact_type": c.get("fact_type"),
              "statement": c.get("canonical_statement"),
              "soft": c.get("soft", False)}
             for c in ledger.get("claims", [])],
            indent=2,
        ),
        target_beats=target_beats,
    )

    script = _generate_within_range(
        llm, prompt, min_w=min_w, max_w=max_w, target_w=target_words,
        max_attempts=8,
    )

    # forbidden-phrase lint with one rewrite attempt
    forbidden = _load_forbidden()
    hits = _find_forbidden(script, forbidden)
    if hits:
        logger.warning("forbidden phrases on chosen draft: %s", hits[:5])
        retry_prompt = (
            prompt
            + "\n\nADDITIONAL CONSTRAINT FOR THIS RETRY:\n"
            + "Your previous draft contained these forbidden phrases. "
            + "Rewrite the script with none of them present:\n"
            + "\n".join(f"  - {h}" for h in hits)
        )
        result = llm.complete(retry_prompt, temperature=0.7, max_tokens=12000)
        script = _clean(result.text)
        hits2 = _find_forbidden(script, forbidden)
        if hits2:
            return f"script still contains forbidden phrases after rewrite: {hits2[:3]}"

    # Length gate — last-mile expand/condense
    wc = len(script.split())
    if wc < min_w:
        logger.warning("S06 undershoot %d/%d, trying expand", wc, min_w)
        script = _expand_script(llm, script, min_w, target_words)
        wc = len(script.split())
        logger.info("S06 after expansion: %d words", wc)
    elif wc > max_w:
        logger.warning("S06 overshoot %d/%d, trying condense", wc, max_w)
        script = _condense_script(llm, script, max_w, target_words)
        wc = len(script.split())
        logger.info("S06 after condense: %d words", wc)

    # Beat normalization
    min_beats = cfg.quality_gates["min_total_beats"]
    max_beats = cfg.quality_gates["max_total_beats"]
    target_beats_count = (min_beats + max_beats) // 2
    beats = BEAT_RE.findall(script)
    if len(beats) > max_beats:
        logger.warning("S06 too many beats (%d > %d), consolidating to ~%d",
                       len(beats), max_beats, target_beats_count)
        script = _consolidate_beats(script, target_beats_count)
        beats = BEAT_RE.findall(script)
    elif len(beats) < min_beats:
        logger.warning("S06 too few beats (%d < %d), redistributing to ~%d",
                       len(beats), min_beats, target_beats_count)
        script = _redistribute_beats(script, target_beats_count)
        beats = BEAT_RE.findall(script)

    (ws / "02_script").mkdir(exist_ok=True)
    (ws / "02_script" / "script.txt").write_text(script)

    if _find_forbidden(script, forbidden):
        return f"forbidden phrase reintroduced by length retry: {_find_forbidden(script, forbidden)[:3]}"

    if wc < min_w or wc > max_w:
        return f"script word count {wc} outside {min_w}-{max_w} (after retry)"
    if len(beats) < min_beats:
        return f"only {len(beats)} BEAT markers (need {min_beats}) after redistribution"
    if len(beats) > max_beats:
        return f"too many BEAT markers ({len(beats)}) after consolidation; cap {max_beats}"

    (ws / "02_script" / "script_meta.json").write_text(json.dumps({
        "word_count": wc,
        "beat_count": len(beats),
        "archetype": archetype,
        "narrator": narrator,
        "visual_style": visual_style,
    }, indent=2))
    logger.info("S06 complete: %d words, %d beats", wc, len(beats))
    return None


# -------------------- forbidden phrase lint --------------------

def _load_forbidden() -> list[str]:
    path = Path(__file__).resolve().parent.parent / "lint" / "forbidden_phrases.txt"
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.lower())
    return out


def _find_forbidden(text: str, phrases: list[str]) -> list[str]:
    t = text.lower()
    return [p for p in phrases if p in t]


# -------------------- beat re-distribution --------------------

def _consolidate_beats(script: str, target_count: int) -> str:
    matches = list(BEAT_RE.finditer(script))
    n = len(matches)
    if n <= target_count:
        return script
    step = n / target_count
    keep = {int(i * step) for i in range(target_count)}
    parts: list[str] = []
    cursor = 0
    new_idx = 0
    for i, m in enumerate(matches):
        parts.append(script[cursor:m.start()])
        if i in keep:
            new_idx += 1
            parts.append(f"## BEAT {new_idx} ##")
        cursor = m.end()
    parts.append(script[cursor:])
    return "".join(parts)


def _redistribute_beats(script: str, target_count: int) -> str:
    cleaned = BEAT_RE.sub("", script).strip()
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", cleaned)

    para_breaks = [m.start() for m in re.finditer(r"\n\n", cleaned)]
    sent_breaks = [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"'])", cleaned)]

    if len(para_breaks) >= target_count:
        breaks = para_breaks
    else:
        breaks = sorted(set(para_breaks + sent_breaks))

    if len(breaks) < target_count:
        return _insert_beats_by_word(cleaned, target_count)

    step = len(breaks) / target_count
    keep = sorted({breaks[int(i * step)] for i in range(target_count)})

    parts: list[str] = []
    cursor = 0
    for idx, pos in enumerate(keep, start=1):
        parts.append(cleaned[cursor:pos])
        parts.append(f"\n\n## BEAT {idx} ##\n\n")
        cursor = pos
    parts.append(cleaned[cursor:])
    return "".join(parts)


def _insert_beats_by_word(text: str, target_count: int) -> str:
    words = text.split()
    n = len(words)
    if n < target_count:
        return text
    per_beat = max(1, n // target_count)
    parts: list[str] = []
    for i in range(target_count):
        start = i * per_beat
        end = (i + 1) * per_beat if i < target_count - 1 else n
        parts.append(f"## BEAT {i + 1} ##")
        parts.append(" ".join(words[start:end]))
    return "\n\n".join(parts)


# -------------------- length adjustment --------------------

def _generate_within_range(
    llm, base_prompt: str, *, min_w: int, max_w: int, target_w: int,
    max_attempts: int = 8,
) -> str:
    best_script: str | None = None
    best_distance = float("inf")
    last_wc: int | None = None

    for attempt in range(max_attempts):
        if attempt == 0:
            prompt = base_prompt
            temperature = 0.75
        else:
            pressure = (
                "*** LENGTH BUDGET — READ FIRST ***\n"
                f"Previous attempt produced {last_wc} words. "
                f"Target: {target_w}. Acceptable: {min_w}–{max_w}. "
                "You MUST land in that range. "
                "Allocate roughly: cold open 80, three forward teases "
                "~20 each, editorial closing 230, the remainder split "
                "evenly across BEAT chapters. Cut middle-act elaboration "
                "and adjective stacks if running long; add forensic "
                "detail from the ledger if running short.\n"
                "*** END LENGTH BUDGET ***\n\n"
            )
            prompt = pressure + base_prompt
            temperature = max(0.45, 0.75 - attempt * 0.1)

        logger.info("S06 attempt %d (temp=%.2f)", attempt + 1, temperature)
        result = llm.complete(prompt, temperature=temperature, max_tokens=12000)
        script = _clean(result.text)
        wc = len(script.split())
        last_wc = wc
        distance = max(0, min_w - wc, wc - max_w)
        in_range = min_w <= wc <= max_w
        logger.info("S06 attempt %d: %d words (dist=%d, in_range=%s)",
                    attempt + 1, wc, distance, in_range)
        if in_range:
            return script
        if distance < best_distance:
            best_script = script
            best_distance = distance

    if best_script is None:
        return ""
    logger.warning("S06 no attempt landed in [%d, %d]; using closest (dist=%d)",
                   min_w, max_w, best_distance)
    return best_script


def _expand_script(
    llm, script: str, target_min: int, target_words: int, max_attempts: int = 3,
) -> str:
    for attempt in range(max_attempts):
        current = len(script.split())
        if current >= target_min:
            return script
        needed = max(target_words - current, target_min - current + 100)
        intensity = [
            "",
            "PREVIOUS ATTEMPT WAS TOO SHORT. Be more generous with expansion. ",
            "URGENT: previous attempts failed. You MUST significantly expand. ",
        ][min(attempt, 2)]
        expand_prompt = (
            f"{intensity}"
            f"The script below is {current} words; we need {target_min}-"
            f"{target_words + 200}. Add about {needed} words by EXPANDING "
            "the existing draft. Do NOT rewrite from scratch. Do NOT "
            "invent facts beyond the ledger originally supplied.\n\n"
            "Areas you may expand:\n"
            "- Business / market context (industry conditions, "
            "competitor positioning, era technology) — only from ledger.\n"
            "- Founder-voice quoted material if present in the ledger.\n"
            "- Forensic detail in middle act (filings, deposition lines, "
            "specific dates) — only from ledger.\n"
            "- The editorial closing — one more concrete observation, "
            "no more than 50 additional words.\n\n"
            "PRESERVE EXACTLY:\n"
            "- The cold open (first paragraph, 60-100 words).\n"
            "- Every forward-tease sentence.\n"
            "- The editorial closing's final concrete image.\n"
            "- All ## BEAT N ## markers.\n\n"
            "Return the FULL revised script as plain text. No code "
            "fences, no preamble.\n\n"
            f"CURRENT SCRIPT ({current} words):\n---\n{script}\n---\n"
        )
        out = _clean(llm.complete(
            expand_prompt, temperature=0.6 + attempt * 0.1, max_tokens=12000,
        ).text)
        new_wc = len(out.split())
        if new_wc > current and new_wc > 500:
            logger.info("expand %d: %d -> %d", attempt + 1, current, new_wc)
            script = out
        else:
            logger.warning("expand %d no progress (%d words)", attempt + 1, new_wc)
    return script


def _condense_script(
    llm, script: str, target_max: int, target_words: int, max_attempts: int = 3,
) -> str:
    for attempt in range(max_attempts):
        current = len(script.split())
        if current <= target_max:
            return script
        excess = current - target_max
        intensity = [
            "",
            "PREVIOUS ATTEMPT WAS NOT SHORTER. Be more aggressive. ",
            "URGENT: previous attempts produced no shortening. You MUST cut "
            "at least 20% of the middle act. ",
        ][min(attempt, 2)]
        condense_prompt = (
            f"{intensity}"
            f"The script below is {current} words. Target {target_max-200}"
            f"-{target_max}. Remove approximately {excess + 100} words. "
            "Do NOT rewrite from scratch.\n\n"
            "PRESERVE EXACTLY:\n"
            "- The cold open (first paragraph, 60-100 words).\n"
            "- Every forward-tease sentence.\n"
            "- The editorial closing (final ~200 words).\n"
            "- All ## BEAT N ## markers (you may merge two adjacent "
            "beats if their content collapses into one paragraph).\n\n"
            "CUT FROM THE MIDDLE ACT by:\n"
            "- Removing adjective stacks.\n"
            "- Removing restatements of facts already given.\n"
            "- Removing parentheticals and asides.\n"
            "- Merging adjacent paragraphs covering the same beat.\n"
            "- Dropping sentences that don't advance the timeline.\n\n"
            "Return the FULL revised script as plain text. No code "
            "fences, no preamble.\n\n"
            f"CURRENT SCRIPT ({current} words):\n---\n{script}\n---\n"
        )
        out = _clean(llm.complete(
            condense_prompt, temperature=0.4 + attempt * 0.15, max_tokens=12000,
        ).text)
        new_wc = len(out.split())
        if new_wc < current and new_wc > 500:
            logger.info("condense %d: %d -> %d", attempt + 1, current, new_wc)
            script = out
        else:
            logger.warning("condense %d no progress (%d words)", attempt + 1, new_wc)
    return script
