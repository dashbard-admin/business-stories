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
              # Batch H 2026-05-28: pass document_citation through to
              # the writer so Act 3.5 can cite filings/depositions
              # verbatim instead of producing generic analysis.
              "document_citation": c.get("document_citation"),
              "exact_quote": c.get("exact_quote"),
              "soft": c.get("soft", False)}
             for c in ledger.get("claims", [])],
            indent=2,
        ),
        target_beats=target_beats,
        retention_dip_warnings=perf["retention_dip_warnings"],
    )

    # Retry budget + temperature decay are operator-tunable per
    # config.production.{max_script_generation_attempts,
    # script_generation_temp_step}. Defaults match the historical
    # behaviour (8 attempts, 0.05 step → temp stays in the 0.45-0.75
    # band across the whole loop).
    max_attempts = int(cfg.production.get("max_script_generation_attempts", 8))
    temp_step = float(cfg.production.get("script_generation_temp_step", 0.05))
    script = _generate_within_range(
        llm, prompt, min_w=min_w, max_w=max_w, target_w=target_words,
        max_attempts=max_attempts, temp_step=temp_step,
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
    # 120 wpm those play for 4-7 seconds on screen, too fast for the
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
    (20) words. At 120 wpm that's ~10 seconds of narration per
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
) -> str:
    """Generate a script that lands inside [min_w, max_w] words, retrying
    up to `max_attempts` times. Each retry adds a length-budget prefix
    nudge and lowers temperature by `temp_step` (floored at 0.45).
    Both knobs are operator-tunable via config.production from
    2026-05-28; defaults preserve creativity across retries with
    a 0.05 step (the original 0.10 step crashed to the floor by
    attempt 4)."""
    best_script: str | None = None
    best_distance = float("inf")
    last_wc: int | None = None

    for attempt in range(max_attempts):
        if attempt == 0:
            prompt = base_prompt
            temperature = 0.80
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
            temperature = max(0.45, 0.80 - attempt * temp_step)

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
