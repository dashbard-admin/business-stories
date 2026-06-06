"""S06 — Script Generation.

Writes a compact business-story script using the locked
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
CALLOUT_RE = re.compile(
    r"\[\s*CALLOUT\s*:\s*[\"“‘]?([^\"”’\]]+)[\"”’]?\s*\]",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]")

ACT_SPECS = [
    ("ACT_0", "The Hook", 70, 2),
    ("ACT_1", "The Before", 260, 7),
    ("ACT_2", "The Bet", 230, 7),
    ("ACT_3", "The Crisis", 360, 10),
    ("ACT_3_5", "The Evidence", 180, 5),
    ("ACT_4", "The Pivot or Collapse", 280, 8),
    ("ACT_5", "The Lesson", 220, 6),
]

# Strip Kokoro-poisoning terminal sign-offs ("End of script", "THE END",
# "Fin.", "---END---") from the script tail.
#
# Batch H 2026-05-28: the original regex used a (?<=[.!?]) lookbehind
# that broke when a separator like `--`, `## BEAT 80 ##`, or a blank-
# line gap sat between the last sentence's period and the boilerplate.
# The Quibi script ended with "The end." after a BEAT 80 marker and
# the regex couldn't anchor. Two regexes now run in sequence: a strict
# one (period-anchored, for the common case) AND a loose tail-only
# one (matches just the trailing line(s) regardless of context).
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

# Loose tail-only sweep: any LINE at the end of the document that just
# says "the end" / "fin" / "end of script" / "[End]" / "---END---" gets
# stripped, regardless of what's directly before it. Applied AFTER the
# strict regex so the safer one runs first.
TERMINAL_TAIL_LOOSE_RE = re.compile(
    r"""(?xm)
    (?:
        # bare "the end", optionally bracketed/dashed
        ^ \s* [\-*=\[\(]* \s* (?i:the \s+ end) \s* [\-*=\]\)\.\!\?]* \s* $
        |
        # "fin" / "fin." / "---FIN---"
        ^ \s* [\-*=\[\(]* \s* (?i:fin) \s* [\-*=\]\)\.\!\?]* \s* $
        |
        # bracketed "end" with no inner content: [End] ---END--- (END)
        ^ \s* [\-*=\[\(]+ \s* (?i:end) \s* [\-*=\]\)]+ \s* $
        |
        # bare "end of script/narration/report/etc.", optionally
        # wrapped in brackets/dashes
        ^ \s* [\-*=\[\(]* \s*
            (?i: end \s+ of \s+
              (?: script | narration | report | episode | story
                  | transcript | text ))
            \s* [\-*=\]\)\.\!\?]* \s* $
    )
    \s* $
    """,
)


# Orphan beat marker — `## BEAT N` without a closing `##`.
# The Quibi script2 had the LLM emitting these (Markdown H2 syntax)
# alongside the canonical paired-delimiter form. Strip them in
# _clean() BEFORE _redistribute / _merge can act on miscounted beats.
# Added Batch I 2026-05-28.
ORPHAN_BEAT_RE = re.compile(
    r"""(?xm)
    ^                       # start of line
    [ \t]*                  # optional leading whitespace
    \#\#                    # opening ##
    \s*BEAT\s+\d+           # BEAT N
    (?!\s*\#\#)             # NOT followed by closing ##
    [ \t]*                  # optional trailing whitespace on the marker line
    $                       # end of line
    """,
    re.IGNORECASE,
)

# Stray bracketed-token leaks. The Batch H Quibi script2 had
# [SPONSOR_SLOT] markers throughout because the prompt's commented
# placeholder was read as an instruction. We strip these post-hoc
# in case future prompt edits re-introduce the same kind of leak.
# CALLOUT and PAUSE/EMPHASIS are legitimate so they're NOT stripped.
STRAY_TOKEN_RE = re.compile(
    r"""(?x)
    \[
    \s*
    (?: SPONSOR_SLOT | SPONSOR \s+ SLOT | SPONSOR \s+ READ
        | INTRO | OUTRO )
    \s*
    \]
    """,
    re.IGNORECASE,
)

MONTH_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\b\.?\s+\d{1,2}(?:st|nd|rd|th)?",
    re.IGNORECASE,
)

MONEY_NUM_RE = re.compile(
    r"\$\s*\d+(?:[.,]\d+)?\s*(?:million|billion|m|bn|b)?",
    re.IGNORECASE,
)
MONEY_WORD_RE = re.compile(
    r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|half|quarter|[-\s])+?)\s+"
    r"(million|billion)\s+dollars\b",
    re.IGNORECASE,
)
COUNT_WORD_RE = re.compile(
    r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|[-\s])+?)\s+"
    r"(users|customers|subscribers|creators|views|employees)\b",
    re.IGNORECASE,
)
DURATION_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|ninety|hundred)\s+"
    r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b|\btwenty[-\s](?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one|twenty-two|twenty-three|twenty-four|twenty-five|twenty-six)\b", re.IGNORECASE)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}


def _clean(text: str) -> str:
    """Strip <think> tags, code fences, stray placeholder tokens, and
    terminal boilerplate. Orphan beat-marker stripping lives OUTSIDE
    this function (in run()) so the dual-stream check can see the raw
    count first."""
    text = THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    # Strip stray placeholder tokens (SPONSOR_SLOT etc.) that leaked
    # from the prompt's commented placeholders. Added Batch I
    # 2026-05-28. CALLOUT / PAUSE / EMPHASIS markers are legitimate
    # and explicitly preserved.
    text = STRAY_TOKEN_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Up to 4 iterations: each pass strips one terminal sign-off so
    # stacked tails ("The end.\n\n[End of script]") all come off.
    for _ in range(4):
        new_text = TERMINAL_BOILERPLATE_RE.sub("", text).rstrip()
        if new_text != text:
            text = new_text
            continue
        # Strict regex didn't match — try the loose tail-only sweep.
        new_text = TERMINAL_TAIL_LOOSE_RE.sub("", text).rstrip()
        if new_text == text:
            break
        text = new_text
    return text


def _strip_orphan_beats(text: str) -> tuple[str, int]:
    """Remove orphan `## BEAT N` markers (no closing `##`). Returns
    (cleaned_text, n_stripped). Called from run() AFTER the dual-
    stream detector has had a chance to see the raw count."""
    matches = ORPHAN_BEAT_RE.findall(text)
    if not matches:
        return text, 0
    cleaned = ORPHAN_BEAT_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, len(matches)


def _detect_dual_stream(text: str) -> tuple[bool, int, int]:
    """Detect when the LLM emitted TWO parallel beat-marker streams
    (the Quibi script2 failure mode). Returns
    (is_dual_stream, valid_count, orphan_count). The dual-stream flag
    fires when BOTH counts are non-trivial and the orphan count is at
    least ~50% of the valid count — that's the LLM emitting a parallel
    Markdown-H2 numbering on top of the canonical paired-delimiter
    form. Added Batch I 2026-05-28."""
    valid = len(BEAT_RE.findall(text))
    orphans = len(ORPHAN_BEAT_RE.findall(text))
    is_dual = valid >= 5 and orphans >= max(3, valid // 2)
    return is_dual, valid, orphans


def _number_words_to_int(text: str) -> int | None:
    total = 0
    current = 0
    seen = False
    for raw in re.split(r"[\s-]+", text.lower()):
        word = raw.strip(" ,.")
        if not word:
            continue
        if word in NUMBER_WORDS:
            current += NUMBER_WORDS[word]
            seen = True
        elif word == "hundred":
            current = max(1, current) * 100
            seen = True
        elif word == "thousand":
            total += max(1, current) * 1000
            current = 0
            seen = True
        elif word in {"million", "billion"}:
            break
    value = total + current
    return value if seen and value > 0 else None


def _compact_number_phrase(text: str, unit: str | None = None) -> str | None:
    text = text.strip()
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    if m:
        value = float(m.group(0).replace(",", ""))
    else:
        parsed = _number_words_to_int(text)
        if parsed is None:
            return None
        value = float(parsed)
    if unit:
        suffix = "B" if unit.lower().startswith("b") else "M"
        return f"{int(value) if value.is_integer() else value:g}{suffix}"
    return f"{int(value) if value.is_integer() else value:g}"


def _callout_for_sentence(sentence: str) -> str | None:
    money = MONEY_NUM_RE.search(sentence)
    if money:
        raw = money.group(0).upper().replace(" ", "")
        raw = raw.replace("MILLION", "M").replace("BILLION", "B")
        raw = raw.replace("BN", "B")
        if len(raw) <= 20:
            return raw

    money_word = MONEY_WORD_RE.search(sentence)
    if money_word:
        compact = _compact_number_phrase(
            money_word.group(1), unit=money_word.group(2)
        )
        if compact:
            return f"${compact}"

    count_word = COUNT_WORD_RE.search(sentence)
    if count_word:
        compact = _compact_number_phrase(
            count_word.group(1), unit=(
                "billion" if "billion" in count_word.group(1).lower()
                else "million" if "million" in count_word.group(1).lower()
                else None
            ),
        )
        if compact:
            label = count_word.group(2).upper()
            return f"{compact} {label}"[:20]

    duration = DURATION_RE.search(sentence)
    if duration:
        value = duration.group(0).upper()
        value = re.sub(r"\bONE\b", "1", value)
        value = re.sub(r"\bTWO\b", "2", value)
        value = re.sub(r"\bTHREE\b", "3", value)
        value = re.sub(r"\bFOUR\b", "4", value)
        value = re.sub(r"\bFIVE\b", "5", value)
        value = re.sub(r"\bSIX\b", "6", value)
        if len(value) <= 20:
            return value

    month = MONTH_RE.search(sentence)
    if month:
        return month.group(0).upper().replace(".", "")[:20]

    year = YEAR_RE.search(sentence)
    if year:
        y = year.group(0)
        if y.lower().startswith("twenty"):
            tail = y.lower().replace("-", " ").split()[-1]
            year_map = {
                "ten": "2010", "eleven": "2011", "twelve": "2012",
                "thirteen": "2013", "fourteen": "2014",
                "fifteen": "2015", "sixteen": "2016",
                "seventeen": "2017", "eighteen": "2018",
                "nineteen": "2019", "twenty": "2020",
                "one": "2021", "two": "2022", "three": "2023",
                "four": "2024", "five": "2025", "six": "2026",
            }
            return year_map.get(tail)
        return y

    return None


def _ensure_callout_markers(
    script: str,
    *,
    min_total: int,
    target_total: int,
    max_total: int,
) -> tuple[str, dict]:
    existing = CALLOUT_RE.findall(script)
    desired = min(max_total, max(min_total, target_total))
    if len(existing) >= desired or max_total <= 0:
        return script, {"added": 0, "total": len(existing)}

    used_text = {c.strip().upper() for c in existing}
    used_beats = {
        int(m.group(1))
        for m in re.finditer(
            r"##\s*BEAT\s+(\d+)\s*##(?:(?!##\s*BEAT\s+\d+\s*##).)*"
            r"\[\s*CALLOUT\s*:",
            script,
            re.IGNORECASE | re.DOTALL,
        )
    }
    parts = list(BEAT_RE.finditer(script))
    inserts: list[tuple[int, str]] = []
    target = min(max_total, max(desired, len(existing)))

    for idx, marker in enumerate(parts):
        if len(existing) + len(inserts) >= target:
            break
        beat_num = int(marker.group(1))
        if beat_num in used_beats:
            continue
        start = marker.end()
        end = parts[idx + 1].start() if idx + 1 < len(parts) else len(script)
        beat_text = script[start:end]
        for sent in SENTENCE_RE.finditer(beat_text):
            sentence = sent.group(0).strip()
            if "[CALLOUT:" in sentence.upper():
                continue
            callout = _callout_for_sentence(sentence)
            if not callout:
                continue
            callout = callout.strip()[:20]
            if not re.search(r"[\d$]", callout):
                continue
            if callout.upper() in used_text:
                continue
            insert_at = start + sent.end()
            inserts.append((insert_at, f'\n[CALLOUT: "{callout}"]\n'))
            used_text.add(callout.upper())
            used_beats.add(beat_num)
            break

    if not inserts:
        return script, {"added": 0, "total": len(existing)}

    for pos, text in sorted(inserts, reverse=True):
        script = script[:pos].rstrip() + text + script[pos:]
    script = re.sub(r"(\[CALLOUT:[^\n]+\]\n)[ \t]+", r"\1", script)
    total = len(CALLOUT_RE.findall(script))
    return script, {"added": len(inserts), "total": total}


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

    # Batch E 2026-05-27 — performance feedback. Empty until
    # --analyse-performance accumulates data.
    from ..performance_summary import summarise_for_prompt
    perf = summarise_for_prompt()

    fact_claims = [
        {"id": c.get("claim_id"),
         "fact_type": c.get("fact_type"),
         "statement": c.get("canonical_statement"),
         # Batch H 2026-05-28: pass document_citation through to
         # the writer so Act 3.5 can cite filings/depositions
         # verbatim instead of producing generic analysis.
         "document_citation": c.get("document_citation"),
         "exact_quote": c.get("exact_quote"),
         "soft": c.get("soft", False)}
        for c in ledger.get("claims", [])
    ]
    fact_ledger_json = json.dumps(fact_claims, indent=2)

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
        fact_ledger_json=fact_ledger_json,
        target_beats=target_beats,
        retention_dip_warnings=perf["retention_dip_warnings"],
    )

    # Retry budget + temperature decay are operator-tunable per
    # config.production.{max_script_generation_attempts,
    # script_generation_temp_step}. Defaults match the historical
    # behaviour (8 attempts, 0.05 step → temp stays in the 0.45-0.75
    # band across the whole loop).
    #
    # Batch I.2 2026-05-28: forbidden-phrase check folded INTO the
    # retry loop (was a separate single-shot rewrite after the loop).
    # The loop now scores each draft by word-range fit AND clean-
    # phrase status; a draft that's in_range AND clean returns
    # immediately, otherwise the loop keeps trying. Selection
    # priority after exhausting attempts: in_range+forbidden >
    # clean+out-of-range > out_of_range+forbidden.
    forbidden = _load_forbidden()
    max_attempts = int(cfg.production.get("max_script_generation_attempts", 8))
    temp_step = float(cfg.production.get("script_generation_temp_step", 0.05))
    # Prompt-log path (added 2026-05-28). _generate_within_range
    # overwrites this on every attempt. After the loop returns, the
    # file holds the prompt that produced the chosen draft — useful
    # for diagnosing why a particular generation succeeded or failed.
    prompt_log_path = ws / "02_script" / "script_prompt.txt"
    staged_enabled = bool(cfg.production.get("script_act_by_act_enabled", True))
    script = ""
    if staged_enabled and not preview_mode:
        script = _generate_staged_within_range(
            llm,
            cfg=cfg,
            ws=ws,
            incident=incident,
            archetype_name=arch["name"],
            archetype_guidance=arch_guidance,
            narrator_name=narr["name"],
            narrator_tone=narr_cfg["tone"],
            narrator_id=narrator,
            narrator_full_instructions=narr["full_instructions"],
            visual_style_name=style_yaml["name"],
            character_iconography=character_iconography,
            fact_ledger_json=fact_ledger_json,
            retention_dip_warnings=perf["retention_dip_warnings"],
            target_words=target_words,
            min_words=min_w,
            max_words=max_w,
            target_beats=target_beats,
            forbidden=forbidden,
            max_attempts=max_attempts,
            temp_step=temp_step,
        )

    if not script:
        script = _generate_within_range(
            llm, prompt, min_w=min_w, max_w=max_w, target_w=target_words,
            max_attempts=max_attempts, temp_step=temp_step,
            forbidden=forbidden,
            prompt_log_path=prompt_log_path,
        )

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

    # Dual-stream safety check (Batch I 2026-05-28). The Quibi
    # script2 had the writer LLM emit BOTH paired-`## BEAT N ##`
    # markers AND orphan `## BEAT N` Markdown-H2 markers in parallel.
    # _redistribute_beats then added a third stream of properly-
    # formatted markers on top because BEAT_RE.findall() only counts
    # the paired form. The result was 80 markers in 1700 words of
    # tangled prose. Detection runs BEFORE we strip orphans so the
    # original counts are visible. If the LLM emitted significantly
    # more orphans than paired markers, the output is unsalvageable
    # by post-hoc auto-fix and goes to needs_human.
    is_dual, valid_count, orphan_count = _detect_dual_stream(script)
    if is_dual:
        (ws / "02_script").mkdir(exist_ok=True)
        (ws / "02_script" / "script.draft.dual-stream.txt").write_text(script)
        return (
            f"dual beat-marker stream detected: {valid_count} valid "
            f"`## BEAT N ##` + {orphan_count} orphan `## BEAT N` "
            f"markers in the LLM output. The writer is confused by "
            f"the prompt. Inspect "
            f"02_script/script.draft.dual-stream.txt then either "
            f"hand-fix the script and `--approve {episode['id']}`, "
            f"or re-run S6 after a prompt-template change."
        )
    # Below-threshold orphans (mild confusion only) get silently
    # stripped; log how many so the operator can spot a trend.
    if orphan_count:
        script, n_stripped = _strip_orphan_beats(script)
        logger.info(
            "S06: stripped %d orphan `## BEAT N` markers (LLM emitted "
            "%d valid + %d orphans, below dual-stream threshold)",
            n_stripped, valid_count, orphan_count,
        )

    # Beat normalization.
    # Bugfix 2026-05-28: the original block used cfg.quality_gates'
    # min/max unconditionally. When preview_mode is on, the script is
    # ONLY Act 0+5 (~360 words / ~8 beats), and redistributing those
    # ~3 beats up to ~80 splits the prose at word boundaries instead
    # of natural sentence boundaries — producing 5-words-per-beat
    # gibberish. Honor the preview-mode beat target so redistribution
    # stays sane.
    if preview_mode:
        min_beats = 6
        max_beats = 12
        target_beats_count = 8
    else:
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

    # Short-beat consolidation (Batch H 2026-05-28). The Quibi script
    # had ~13 beats with 1-2 sentences only (8-15 words each) — at
    # current narration speed those play too quickly for the
    # viewer to register the image, and they fragment the prose
    # rhythmically. Merge any beat shorter than 15 words into the
    # NEXT beat. Skips on preview-mode (already-tight pacing).
    if not preview_mode:
        script_before = script
        script = _merge_short_beats(script, min_words=15)
        if script != script_before:
            merged_beats = BEAT_RE.findall(script)
            logger.info("S06 short-beat merge: %d → %d beats "
                        "(absorbed sub-15-word beats)",
                        len(beats), len(merged_beats))
            beats = merged_beats

    # ----- post-retry substitution safety net (Batch J 2026-05-29) -----
    # The retry loop tries to avoid forbidden_phrases.txt hits by
    # re-generating, but a script can still ship with a phrase that
    # survived the budget. forbidden_substitutions.yaml maps each
    # surviving phrase to a safer replacement so the bad phrasing
    # never reaches S10 TTS / S12 captions. The substitution is the
    # LAST guardrail — preferred fix is always a clean re-generate.
    sub_table = _load_substitutions()
    if sub_table:
        script, sub_log = _apply_substitutions(script, sub_table)
        if sub_log:
            logger.info("S06 forbidden-phrase substitutions applied: %s",
                        ", ".join(f'{m}→{r}' for m, r in sub_log))

    # Callout repair. The prompt asks for 3-6 numeric callouts, but
    # act-by-act generation can under-deliver because each act sees
    # only its own slice. Add safe markers after existing numeric
    # sentences before S08 parses the beat sheet.
    if cfg.callouts.get("enabled", True):
        script, callout_info = _ensure_callout_markers(
            script,
            min_total=int(cfg.callouts.get("min_total", 3)),
            target_total=int(cfg.callouts.get("target_total", 5)),
            max_total=int(cfg.callouts.get("max_total", 6)),
        )
        if callout_info["added"]:
            logger.info(
                "S06 callout repair: added %d marker(s), total=%d",
                callout_info["added"], callout_info["total"],
            )

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
        "callout_count": len(CALLOUT_RE.findall(script)),
        "archetype": archetype,
        "narrator": narrator,
        "visual_style": visual_style,
    }, indent=2))
    logger.info("S06 complete: %d words, %d beats", wc, len(beats))
    return None


# -------------------- staged script generation --------------------

def _generate_staged_within_range(
    llm,
    *,
    cfg,
    ws: Path,
    incident: dict,
    archetype_name: str,
    archetype_guidance: str,
    narrator_name: str,
    narrator_tone: str,
    narrator_id: str,
    narrator_full_instructions: str,
    visual_style_name: str,
    character_iconography: str,
    fact_ledger_json: str,
    retention_dip_warnings: str,
    target_words: int,
    min_words: int,
    max_words: int,
    target_beats: int,
    forbidden: list[str],
    max_attempts: int,
    temp_step: float,
) -> str:
    """Retry the staged blueprint→acts path until it satisfies gates.

    Earlier staged generation tried once, then only ran local
    expand/condense repairs. This made `max_script_generation_attempts`
    apply to the old full-script fallback but not to the active staged
    path. Keep the same bounded retry budget here.
    """
    best_clean: tuple[int, str] | None = None
    best_any: tuple[int, str] | None = None
    last_wc: int | None = None
    failures: list[str] = []

    for attempt in range(max(1, max_attempts)):
        temperature = 0.62 if attempt == 0 else max(
            0.45, 0.62 - attempt * temp_step
        )
        length_directive = _staged_length_directive(
            attempt=attempt,
            last_wc=last_wc,
            min_words=min_words,
            max_words=max_words,
            target_words=target_words,
        )
        try:
            script = _generate_via_blueprint_and_acts(
                llm,
                cfg=cfg,
                ws=ws,
                incident=incident,
                archetype_name=archetype_name,
                archetype_guidance=archetype_guidance,
                narrator_name=narrator_name,
                narrator_tone=narrator_tone,
                narrator_id=narrator_id,
                narrator_full_instructions=narrator_full_instructions,
                visual_style_name=visual_style_name,
                character_iconography=character_iconography,
                fact_ledger_json=fact_ledger_json,
                retention_dip_warnings=retention_dip_warnings,
                target_words=target_words,
                min_words=min_words,
                max_words=max_words,
                target_beats=target_beats,
                forbidden=forbidden,
                attempt_index=attempt,
                act_temperature=temperature,
                length_directive=length_directive,
            )
        except Exception as e:
            failures.append(str(e)[:180])
            logger.warning(
                "S06 staged attempt %d failed: %s",
                attempt + 1, e,
            )
            continue

        wc = len(script.split())
        hits = _find_forbidden(script, forbidden)
        in_range = min_words <= wc <= max_words
        distance = 0 if in_range else min(abs(wc - min_words), abs(wc - max_words))
        logger.info(
            "S06 staged attempt %d: %d words (dist=%d, in_range=%s, "
            "forbidden_hits=%d)",
            attempt + 1, wc, distance, in_range, len(hits),
        )
        if in_range and not hits:
            return script
        if not hits and (best_clean is None or distance < best_clean[0]):
            best_clean = (distance, script)
        if best_any is None or distance < best_any[0]:
            best_any = (distance, script)
        last_wc = wc

    if best_clean is not None:
        logger.warning(
            "S06 staged retries exhausted; using closest clean draft "
            "(dist=%d)",
            best_clean[0],
        )
        return best_clean[1]
    if best_any is not None:
        logger.warning(
            "S06 staged retries exhausted; using closest draft with "
            "forbidden hits (dist=%d)",
            best_any[0],
        )
        return best_any[1]

    if failures:
        logger.warning(
            "S06 staged act-by-act generation failed on all attempts; "
            "falling back to full-script generation. Last failures: %s",
            " | ".join(failures[-3:]),
        )
    return ""


def _staged_length_directive(
    *,
    attempt: int,
    last_wc: int | None,
    min_words: int,
    max_words: int,
    target_words: int,
) -> str:
    if attempt <= 0 or last_wc is None:
        return (
            f"\n\nHARD LENGTH CONTRACT: final combined script must be "
            f"{min_words}-{max_words} words, with an ideal target near "
            f"{target_words}. Keep every act within its assigned budget. "
            "Do not compensate for short acts by bloating later acts."
        )
    if last_wc > max_words:
        excess = last_wc - max_words
        return (
            f"\n\nRETRY LENGTH CORRECTION: previous staged draft was "
            f"{last_wc} words, {excess} over the hard maximum of "
            f"{max_words}. Generate a materially shorter draft. Cut "
            "middle-act repetition, adjective stacks, repeated context, "
            "and explanatory asides. Preserve the hook, factual spine, "
            "beat markers, and final concrete image."
        )
    if last_wc < min_words:
        shortage = min_words - last_wc
        return (
            f"\n\nRETRY LENGTH CORRECTION: previous staged draft was "
            f"{last_wc} words, {shortage} under the hard minimum of "
            f"{min_words}. Add concise ledger-grounded context and "
            "forensic detail to the middle acts. Do not pad with generic "
            "moralizing or repeated setup."
        )
    return (
        f"\n\nRETRY QUALITY CORRECTION: previous staged draft was "
        f"{last_wc} words, inside the length range, but failed another "
        "gate. Preserve the length while removing forbidden phrases and "
        "keeping all facts ledger-grounded."
    )


def _scaled_act_specs(target_words: int, target_beats: int) -> list[dict]:
    base_words = sum(spec[2] for spec in ACT_SPECS)
    base_beats = sum(spec[3] for spec in ACT_SPECS)
    word_scale = target_words / max(1, base_words)
    beat_scale = target_beats / max(1, base_beats)
    specs: list[dict] = []
    used_words = 0
    used_beats = 0
    for idx, (act_id, title, base_w, base_b) in enumerate(ACT_SPECS):
        is_last = idx == len(ACT_SPECS) - 1
        words = target_words - used_words if is_last else max(60, round(base_w * word_scale))
        beats = target_beats - used_beats if is_last else max(1, round(base_b * beat_scale))
        specs.append({
            "act_id": act_id,
            "title": title,
            "target_words": int(words),
            "target_beats": int(beats),
        })
        used_words += int(words)
        used_beats += int(beats)
    return specs


def _generate_via_blueprint_and_acts(
    llm,
    *,
    cfg,
    ws: Path,
    incident: dict,
    archetype_name: str,
    archetype_guidance: str,
    narrator_name: str,
    narrator_tone: str,
    narrator_id: str,
    narrator_full_instructions: str,
    visual_style_name: str,
    character_iconography: str,
    fact_ledger_json: str,
    retention_dip_warnings: str,
    target_words: int,
    min_words: int,
    max_words: int,
    target_beats: int,
    forbidden: list[str],
    attempt_index: int = 0,
    act_temperature: float = 0.62,
    length_directive: str = "",
) -> str:
    """Generate script as outline first, then one act at a time.

    This lowers variance versus asking the writer for a full script in
    one pass. The old full-script generator remains the
    fallback when this staged path returns an unusable draft.
    """
    act_specs = _scaled_act_specs(target_words, target_beats)
    common = {
        "incident_name": incident["company_name"],
        "year": incident.get("year_anchor"),
        "hero": incident.get("hero", ""),
        "conflict": incident.get("conflict", ""),
        "story_kind": incident.get("story_kind", ""),
        "target_words": target_words,
        "min_words": min_words,
        "max_words": max_words,
        "target_beats": target_beats,
        "act_specs_json": json.dumps(act_specs, indent=2),
        "archetype_name": archetype_name,
        "archetype_guidance": archetype_guidance,
        "narrator_name": narrator_name,
        "narrator_tone": narrator_tone,
        "narrator_id": narrator_id,
        "narrator_full_instructions": narrator_full_instructions,
        "visual_style_name": visual_style_name,
        "character_iconography": character_iconography,
        "fact_ledger_json": fact_ledger_json,
        "retention_dip_warnings": retention_dip_warnings,
    }

    blueprint_template = (cfg.prompts_dir / "script_blueprint.txt").read_text()
    blueprint_prompt = blueprint_template.format(**common)
    if length_directive:
        blueprint_prompt = f"{blueprint_prompt.rstrip()}\n{length_directive}\n"
    (ws / "02_script").mkdir(exist_ok=True)
    (ws / "02_script" / "script_blueprint_prompt.txt").write_text(blueprint_prompt)
    if attempt_index > 0:
        (ws / "02_script" / f"script_blueprint_prompt_attempt_{attempt_index + 1:02d}.txt").write_text(
            blueprint_prompt
        )
    logger.info(
        "S06 staged generation attempt %d: creating blueprint",
        attempt_index + 1,
    )
    blueprint = llm.complete_json(
        blueprint_prompt, temperature=0.35, max_tokens=6000,
    )
    blueprint = _normalize_blueprint(blueprint, act_specs)
    (ws / "02_script" / "script_blueprint.json").write_text(
        json.dumps(blueprint, indent=2)
    )

    act_template = (cfg.prompts_dir / "script_act_generate.txt").read_text()
    act_texts: list[str] = []
    prior_summary = ""
    first_beat = 1
    prompt_log_parts: list[str] = []
    for spec in act_specs:
        act_id = spec["act_id"]
        act_blueprint = _blueprint_for_act(blueprint, act_id)
        last_beat = first_beat + int(spec["target_beats"]) - 1
        act_prompt = act_template.format(
            **common,
            act_id=act_id,
            act_title=spec["title"],
            act_target_words=spec["target_words"],
            act_target_beats=spec["target_beats"],
            beat_start=first_beat,
            beat_end=last_beat,
            act_blueprint_json=json.dumps(act_blueprint, indent=2),
            prior_summary=prior_summary or "(this is the opening act)",
            forbidden_phrases="\n".join(f"  - {p}" for p in forbidden),
        )
        if length_directive:
            act_prompt = f"{act_prompt.rstrip()}\n{length_directive}\n"
        prompt_log_parts.append(f"\n\n===== {act_id} PROMPT =====\n\n{act_prompt}")
        logger.info(
            "S06 staged generation attempt %d: %s target=%d words/%d beats",
            attempt_index + 1, act_id, spec["target_words"], spec["target_beats"],
        )
        result = llm.complete(
            act_prompt, temperature=act_temperature, max_tokens=4500,
        )
        act_text = _clean(result.text)
        act_text = _strip_non_requested_act_headers(act_text)
        act_texts.append(act_text)
        prior_summary = _summarize_tail(act_text)
        first_beat = last_beat + 1

    (ws / "02_script" / "script_act_prompts.txt").write_text(
        "".join(prompt_log_parts).strip()
    )
    if attempt_index > 0:
        (ws / "02_script" / f"script_act_prompts_attempt_{attempt_index + 1:02d}.txt").write_text(
            "".join(prompt_log_parts).strip()
        )
    script = "\n\n".join(part.strip() for part in act_texts if part.strip())
    script = _renumber_beats(script)
    script, sub_log = _apply_substitutions(script, _load_substitutions())
    if sub_log:
        logger.info(
            "S06 staged substitutions applied: %s",
            ", ".join(f"{m}->{r}" for m, r in sub_log),
        )

    wc = len(script.split())
    hits = _find_forbidden(script, forbidden)
    if hits:
        script = _rewrite_forbidden_sentences(llm, script, hits, forbidden)
        script, _ = _apply_substitutions(script, _load_substitutions())
        hits = _find_forbidden(script, forbidden)
        wc = len(script.split())

    if wc > max_words and wc - max_words <= 450:
        script = _condense_script(llm, script, max_words, target_words, max_attempts=2)
    elif wc < min_words and min_words - wc <= 450:
        script = _expand_script(llm, script, min_words, target_words, max_attempts=2)

    logger.info(
        "S06 staged generation attempt %d complete candidate: %d words, "
        "%d beats, %d forbidden hits",
        attempt_index + 1,
        len(script.split()), len(BEAT_RE.findall(script)),
        len(_find_forbidden(script, forbidden)),
    )
    return script


def _normalize_blueprint(raw, act_specs: list[dict]) -> dict:
    if isinstance(raw, dict) and isinstance(raw.get("acts"), list):
        acts = raw["acts"]
    elif isinstance(raw, list):
        acts = raw
    else:
        acts = []
    by_id = {
        str((a or {}).get("act_id") or "").upper().replace(".", "_"): a
        for a in acts if isinstance(a, dict)
    }
    normalized = {"acts": []}
    for spec in act_specs:
        act = dict(by_id.get(spec["act_id"], {}))
        act.setdefault("act_id", spec["act_id"])
        act.setdefault("title", spec["title"])
        act.setdefault("target_words", spec["target_words"])
        act.setdefault("target_beats", spec["target_beats"])
        act.setdefault("beats", [])
        act.setdefault("facts_to_use", [])
        act.setdefault("hook_or_turn", "")
        normalized["acts"].append(act)
    return normalized


def _blueprint_for_act(blueprint: dict, act_id: str) -> dict:
    for act in blueprint.get("acts") or []:
        if str(act.get("act_id")) == act_id:
            return act
    return {"act_id": act_id}


def _strip_non_requested_act_headers(text: str) -> str:
    # The video pipeline only wants BEAT markers. Drop markdown act
    # headings if the model emits them despite the prompt.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,6}\s*ACT\b", stripped, re.IGNORECASE):
            continue
        if re.match(r"^ACT\s+\d", stripped, re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _summarize_tail(text: str, max_words: int = 90) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:])


def _renumber_beats(script: str) -> str:
    idx = 0

    def repl(_m: re.Match[str]) -> str:
        nonlocal idx
        idx += 1
        return f"## BEAT {idx} ##"

    out = BEAT_RE.sub(repl, script)
    if idx == 0 and out.strip():
        out = f"## BEAT 1 ##\n\n{out.strip()}"
    return out.strip()


def _rewrite_forbidden_sentences(
    llm,
    script: str,
    hits: list[str],
    forbidden: list[str],
) -> str:
    """Targeted cleanup for forbidden phrases that substitutions miss."""
    if not hits:
        return script
    prompt = (
        "Rewrite only the sentences containing forbidden phrases in the "
        "script below. Preserve all facts, all ## BEAT N ## markers, all "
        "CALLOUT markers, and the overall word count. Do not rewrite the "
        "whole script.\n\n"
        "Forbidden phrases found:\n"
        + "\n".join(f"  - {h}" for h in hits[:20])
        + "\n\nFull forbidden list:\n"
        + "\n".join(f"  - {p}" for p in forbidden)
        + "\n\nReturn the FULL cleaned script only.\n\n"
        f"SCRIPT:\n---\n{script}\n---\n"
    )
    try:
        out = _clean(llm.complete(
            prompt, temperature=0.35, max_tokens=12000,
        ).text)
    except Exception as e:
        logger.warning("S06 forbidden sentence rewrite failed: %s", e)
        return script
    if len(out.split()) < 500:
        logger.warning("S06 forbidden sentence rewrite returned too little text")
        return script
    return out


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


# -------------------- forbidden-phrase substitution (Batch J) --------------------

def _load_substitutions() -> list[tuple[str, str]]:
    """Load `pipeline/lint/forbidden_substitutions.yaml` into a list of
    (match_lower, replacement) tuples. Returns [] if the file is
    missing or malformed — substitution is purely additive and should
    never block S06 on a parse failure."""
    path = (Path(__file__).resolve().parent.parent
            / "lint" / "forbidden_substitutions.yaml")
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        logger.warning("forbidden_substitutions.yaml parse failed: %s", e)
        return []
    pairs: list[tuple[str, str]] = []
    for entry in (data.get("substitutions") or []):
        m = (entry or {}).get("match")
        r = (entry or {}).get("replace")
        if isinstance(m, str) and isinstance(r, str) and m.strip():
            pairs.append((m.lower(), r))
    return pairs


def _apply_substitutions(
    text: str, table: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """Apply case-insensitive substring substitutions. Returns the
    rewritten text plus a log of (match, replacement) pairs that
    actually fired. Leading-capital matches keep their leading
    capital in the replacement so sentence-initial hits don't drop
    to lowercase mid-sentence."""
    log: list[tuple[str, str]] = []
    out = text
    for match_lower, repl in table:
        # Case-insensitive find. We rebuild `out` repeatedly because
        # a substitution can change the indices.
        pattern = re.compile(re.escape(match_lower), re.IGNORECASE)
        if not pattern.search(out):
            continue

        def _sub(m: re.Match[str]) -> str:
            original = m.group(0)
            # Preserve leading capitalization: if the original starts
            # uppercase, capitalize the replacement's first letter.
            if original[:1].isupper() and repl:
                return repl[:1].upper() + repl[1:]
            return repl

        new_out, n = pattern.subn(_sub, out)
        if n > 0:
            out = new_out
            log.append((match_lower, repl))
    return out, log


# -------------------- beat re-distribution --------------------

def _merge_short_beats(script: str, *, min_words: int = 15) -> str:
    """Walk beats in order. Any beat whose content has fewer than
    `min_words` words is absorbed into the next beat — the marker is
    dropped, the content stays. The trailing beat (no successor) keeps
    its content but the marker stays too so we never lose final-act
    closure. Beat numbers are RE-NUMBERED 1..N at the end.

    Added Batch H 2026-05-28 to fix the Quibi-style fragmentation
    (13 of 80 beats had ≤15 words, playing for 4-7s on screen each).
    """
    matches = list(BEAT_RE.finditer(script))
    if len(matches) <= 1:
        return script

    # Build (marker_text, content_text, marker_start, content_end)
    # tuples. The "content" of beat i is the text between marker i's
    # END and marker (i+1)'s START — or to end-of-string for the last.
    segments: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(script)
        content = script[content_start:content_end]
        segments.append((m.group(0), content))

    # Walk and merge: when current beat's content < min_words AND it's
    # not the last beat, skip emitting THIS marker — its content gets
    # prepended to the next beat's content.
    out_parts: list[str] = []
    carry: str = ""
    for idx, (_marker, content) in enumerate(segments):
        is_last = idx == len(segments) - 1
        merged_content = carry + content
        wc = len(merged_content.split())
        if wc < min_words and not is_last:
            # Don't emit a marker for this short beat; carry its
            # content into the next iteration.
            carry = merged_content
            continue
        # Emit a fresh marker (renumbering happens below) + content.
        out_parts.append(("__MARKER__", merged_content))
        carry = ""

    # Renumber the kept markers 1..N.
    rebuilt: list[str] = []
    n = 1
    for tag, content in out_parts:
        rebuilt.append(f"## BEAT {n} ##")
        rebuilt.append(content.rstrip())
        rebuilt.append("\n\n")
        n += 1
    return "".join(rebuilt).rstrip()


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
    """Insert ## BEAT N ## markers at sentence boundaries so the
    rendered beat count lands near `target_count` without ever
    splitting mid-sentence or mid-word.

    Hard floor: every beat must contain at least MIN_WORDS_PER_BEAT
    (20) words. At 160 wpm that's ~7.5 seconds of narration per
    beat — the minimum needed for a viewer to register the image
    on screen. If the script is too short to hit `target_count`
    while respecting the floor, we return FEWER beats and let the
    downstream min_total_beats gate surface "script too short" as
    a needs_human rather than producing gibberish.

    Bugfix 2026-05-28: was previously falling back to
    `_insert_beats_by_word` which sliced at word boundaries,
    producing 5-word "On December first, two thousand // twenty,
    Quibi Holdings LLC ceased" splits. That fallback is removed.
    """
    cleaned = BEAT_RE.sub("", script).strip()
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", cleaned)

    # Build the sentence list with their character spans.
    sent_starts: list[int] = []
    sent_ends: list[int] = []
    cursor = 0
    for m in re.finditer(r"[^.!?]*[.!?]+[\"')\s]*", cleaned, re.DOTALL):
        s, e = m.start(), m.end()
        if s < cursor:
            continue
        sent_starts.append(s)
        sent_ends.append(e)
        cursor = e
    if not sent_starts:
        # No detectable sentences — nothing safe to split. Return as
        # one beat; downstream gate will catch the script as too
        # short.
        return f"## BEAT 1 ##\n\n{cleaned}"

    MIN_WORDS_PER_BEAT = 20

    # Greedy sentence-pack: walk sentences in order, accumulate
    # into the current beat until it has >= target_words_per_beat
    # words, then close it and start the next. The last sentence
    # always closes whatever beat it lands in.
    total_words = len(cleaned.split())
    target_words_per_beat = max(
        MIN_WORDS_PER_BEAT, total_words // max(1, target_count)
    )

    parts: list[str] = []
    beat_idx = 0
    current_start = 0
    current_word_count = 0
    for s_idx, (s, e) in enumerate(zip(sent_starts, sent_ends)):
        sentence_text = cleaned[s:e]
        sentence_word_count = len(sentence_text.split())
        if current_word_count == 0:
            # Start a new beat at this sentence's start position.
            beat_idx += 1
            parts.append(f"## BEAT {beat_idx} ##\n\n")
            current_start = s
        current_word_count += sentence_word_count
        is_last_sentence = (s_idx == len(sent_starts) - 1)
        if current_word_count >= target_words_per_beat or is_last_sentence:
            parts.append(cleaned[current_start:e].strip())
            parts.append("\n\n")
            current_word_count = 0

    return "".join(parts).strip()


# -------------------- length adjustment --------------------

def _generate_within_range(
    llm, base_prompt: str, *, min_w: int, max_w: int, target_w: int,
    max_attempts: int = 8,
    temp_step: float = 0.05,
    forbidden: list[str] | None = None,
    prompt_log_path: "Path | None" = None,
) -> str:
    """Generate a script that lands inside [min_w, max_w] words AND
    contains NONE of the `forbidden` phrases, retrying up to
    `max_attempts` times. Each retry adds a length-budget prefix nudge
    + (if the previous draft had forbidden hits) a forbidden-list
    pressure block, and lowers temperature by `temp_step` (floored at
    0.45).

    Both knobs are operator-tunable via config.production from
    2026-05-28; defaults preserve creativity across retries with a
    0.05 step. Batch I.2 2026-05-28: forbidden-phrase retry merged
    into the same budget so we don't need a separate single-shot
    rewrite after the loop.

    Selection priority for the returned script (best to worst):
      1. in_range AND clean → return immediately
      2. in_range AND has-forbidden (track as best_in_range)
      3. out_of_range AND clean (track as best_clean)
      4. out_of_range AND has-forbidden (track as best_any)
    After the loop, return the highest-priority candidate seen.
    """
    forbidden = forbidden or []

    best_in_range: str | None = None
    best_in_range_hits: list[str] = []
    best_in_range_prompt: str | None = None
    best_clean: str | None = None
    best_clean_distance = float("inf")
    best_clean_prompt: str | None = None
    best_any: str | None = None
    best_any_distance = float("inf")
    best_any_prompt: str | None = None

    last_wc: int | None = None
    last_hits: list[str] = []

    def _write_prompt(p: str) -> None:
        """Overwrite the prompt log on disk so the operator can inspect
        the actual prompt that produced the most recent attempt (or,
        after the loop returns, the prompt that produced the returned
        draft)."""
        if prompt_log_path is None:
            return
        try:
            prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_log_path.write_text(p)
        except OSError as e:
            logger.warning("S06 could not write prompt log %s: %s",
                           prompt_log_path, e)

    for attempt in range(max_attempts):
        # Build the pressure block.
        pressure_parts: list[str] = []
        if attempt > 0:
            if last_wc is not None:
                pressure_parts.append(
                    "*** LENGTH BUDGET — READ FIRST ***\n"
                    f"Previous attempt produced {last_wc} words. "
                    f"Target: {target_w}. Acceptable: {min_w}–{max_w}. "
                    "You MUST land in that range. "
                    "Allocate roughly: cold open 80, three forward teases "
                    "~20 each, editorial closing 230, the remainder split "
                    "evenly across BEAT chapters. Cut middle-act "
                    "elaboration and adjective stacks if running long; "
                    "add forensic detail from the ledger if running "
                    "short.\n"
                    "*** END LENGTH BUDGET ***\n"
                )
            if last_hits:
                pressure_parts.append(
                    "*** FORBIDDEN PHRASE NUDGE ***\n"
                    "Your previous draft contained these forbidden "
                    "phrases:\n"
                    + "\n".join(f"  - {h}" for h in last_hits[:10])
                    + "\n\nAvoid them on this attempt AND any of the "
                    "full forbidden list (case-insensitive substring "
                    "match):\n"
                    + "\n".join(f"  - {p}" for p in forbidden)
                    + "\n\nIf you find yourself wanting to write any of "
                    "these, use a concrete fact from the ledger instead "
                    "— a specific number, name, date, or document "
                    "reference.\n"
                    "*** END FORBIDDEN PHRASE NUDGE ***\n"
                )

        prompt = (
            "\n".join(pressure_parts) + "\n" + base_prompt
            if pressure_parts else base_prompt
        )
        temperature = 0.80 if attempt == 0 else max(
            0.45, 0.80 - attempt * temp_step
        )

        # Write the prompt to disk BEFORE the LLM call so the operator
        # can inspect even a stuck/crashed generation.
        _write_prompt(prompt)

        logger.info("S06 attempt %d (temp=%.2f)", attempt + 1, temperature)
        result = llm.complete(
            prompt, temperature=temperature, max_tokens=12000,
        )
        script = _clean(result.text)
        sub_table = _load_substitutions()
        if sub_table:
            script, sub_log = _apply_substitutions(script, sub_table)
            if sub_log:
                logger.info(
                    "S06 attempt %d substitutions: %s",
                    attempt + 1,
                    ", ".join(f"{m}->{r}" for m, r in sub_log),
                )
        wc = len(script.split())
        hits = _find_forbidden(script, forbidden) if forbidden else []
        last_wc = wc
        last_hits = hits
        distance = max(0, min_w - wc, wc - max_w)
        in_range = min_w <= wc <= max_w
        is_clean = not hits

        logger.info(
            "S06 attempt %d: %d words (dist=%d, in_range=%s, "
            "forbidden_hits=%d%s)",
            attempt + 1, wc, distance, in_range, len(hits),
            f" {hits[:5]}" if hits else "",
        )

        # Best-case: in_range AND clean → return now. The prompt file
        # already holds THIS attempt's prompt from the _write_prompt
        # call above — no further write needed.
        if in_range and is_clean:
            return script

        # Otherwise update each best-tracker as appropriate. For
        # best_in_range: seed on first in_range draft regardless of
        # hit count, then keep the one with FEWEST forbidden hits.
        # Each tracker remembers the prompt that produced it so the
        # fallback path can rewrite the prompt log to match.
        if in_range:
            if best_in_range is None or len(hits) < len(best_in_range_hits):
                best_in_range = script
                best_in_range_hits = hits
                best_in_range_prompt = prompt
        if is_clean and distance < best_clean_distance:
            best_clean = script
            best_clean_distance = distance
            best_clean_prompt = prompt
        if distance < best_any_distance:
            best_any = script
            best_any_distance = distance
            best_any_prompt = prompt

    # Selection priority after the loop:
    # 1. in_range + has-forbidden (we tried to clean it but couldn't —
    #    word-count was satisfied so we ship the cleanest in-range)
    # 2. clean + out_of_range (closer to range)
    # 3. anything else (best by raw distance)
    # In each case we rewrite the prompt log so it matches the
    # candidate we're actually returning (otherwise the file would
    # hold the LAST attempt's prompt, not the chosen one).
    if best_in_range is not None:
        logger.warning(
            "S06 no attempt was in-range AND clean; using best in-range "
            "with %d forbidden hits: %s",
            len(best_in_range_hits), best_in_range_hits[:5],
        )
        if best_in_range_prompt is not None:
            _write_prompt(best_in_range_prompt)
        return best_in_range
    if best_clean is not None:
        logger.warning(
            "S06 no in-range attempt; using best clean draft "
            "(dist=%d, no forbidden)", best_clean_distance,
        )
        if best_clean_prompt is not None:
            _write_prompt(best_clean_prompt)
        return best_clean
    if best_any is not None:
        logger.warning(
            "S06 no clean attempt; using closest-to-range draft "
            "(dist=%d, may contain forbidden phrases)", best_any_distance,
        )
        if best_any_prompt is not None:
            _write_prompt(best_any_prompt)
        return best_any
    return ""


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
