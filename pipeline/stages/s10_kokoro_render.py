"""S10 — Kokoro TTS Rendering.

Strips BEAT markers from the script, applies pronunciation overrides
and number-to-word conversion, renders chunked WAVs via the Kokoro
adapter, then concatenates into a single voice_full.wav. Writes a
voice_timing.json that maps each beat to start/end timestamp.

Inputs:  02_script/script.txt + 02_script/beat_sheet.json
Outputs: 04_audio/chunks/*.wav + voice_full.wav + voice_timing.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from ..config import load_config
from ..state import find_episode_workspace
from ..tts import KOKORO_SAMPLE_RATE, make_tts

logger = logging.getLogger("hermes.stage.s10")

BEAT_RE = re.compile(r"##\s*BEAT\s+(\d+)\s*##", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d{1,6}\b")
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
PAUSE_RE = re.compile(r"\[PAUSE\s+(\d+(?:\.\d+)?)s\]", re.IGNORECASE)
EMPHASIS_RE = re.compile(r"\[EMPHASIS\]\s*", re.IGNORECASE)
# Batch J 2026-05-29: strip `[CALLOUT: "TEXT"]` markers before TTS.
# S08's beat-sheet parser also strips them from its per-beat
# `script_text` field, but Kokoro reads `02_script/script.txt` raw,
# so the markers reach the synthesizer and get spoken verbatim as
# "callout dec 1 comma 2020" if we don't strip here too. The regex
# tolerates straight + curly quotes and any whitespace around the
# colon — same character classes as s08's _CALLOUT_RE.
CALLOUT_STRIP_RE = re.compile(
    r"\[\s*CALLOUT\s*:\s*[\"“‘]?[^\"”’\]]+[\"”’]?\s*\]",
    re.IGNORECASE,
)


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    script_path = ws / "02_script" / "script.txt"
    if not script_path.exists():
        return "no script.txt"
    raw_script = script_path.read_text()

    # Strip CALLOUT and EMPHASIS markers BEFORE computing beat
    # positions so beat_positions and speech_only stay consistent —
    # mid-loop strips would drift the per-beat char_pos relative to
    # the post-strip n_chars and skew voice_timing.json. The bracketed
    # CALLOUT text lives in beat_sheet.json (set by S08) and is
    # composited as a Pillow overlay by S12; it must NOT be read
    # aloud (Batch J 2026-05-29 — Kokoro was speaking "callout dec one
    # comma twenty twenty" verbatim).
    raw_script = CALLOUT_STRIP_RE.sub("", raw_script)
    raw_script = EMPHASIS_RE.sub("", raw_script)
    # Collapse any double-spaces the strips leave behind, but preserve
    # newlines so BEAT_RE.match() still anchors to line starts cleanly.
    raw_script = re.sub(r"[ \t]+", " ", raw_script)

    # Capture beat positions while stripping BEAT markers.
    beat_positions: list[tuple[int, int]] = []
    speech_chars: list[str] = []
    beat_counter = 0
    i = 0
    while i < len(raw_script):
        m = BEAT_RE.match(raw_script, i)
        if m:
            beat_counter += 1
            beat_positions.append((beat_counter, len(speech_chars)))
            i = m.end()
            continue
        speech_chars.append(raw_script[i])
        i += 1
    speech_only = "".join(speech_chars)

    # Apply pronunciation overrides
    overrides_path = (
        Path(__file__).resolve().parent.parent / "lexicon" / "pronunciation_overrides.yaml"
    )
    if overrides_path.exists():
        try:
            data = yaml.safe_load(overrides_path.read_text()) or {}
        except Exception as e:
            logger.warning("pronunciation overrides parse failed: %s", e)
            data = {}
        # Support two shapes:
        #   (a) flat {term: sub} dict
        #   (b) {overrides: [{match: ..., sub: ...}]}
        if isinstance(data, dict) and "overrides" in data:
            for o in data["overrides"] or []:
                m = o.get("match")
                s = o.get("sub")
                if m and s:
                    speech_only = re.sub(rf"\b{re.escape(m)}\b", s, speech_only)
        elif isinstance(data, dict):
            for term, sub in data.items():
                if isinstance(term, str) and isinstance(sub, str) and term.strip():
                    speech_only = re.sub(rf"\b{re.escape(term)}\b", sub, speech_only)

    # Convert years and standalone numbers to words
    speech_only = _convert_numbers(speech_only)

    # Convert [PAUSE Ns] markers to punctuation prosody hints
    speech_only = PAUSE_RE.sub(_pause_to_punct, speech_only)

    # Render
    chunks_dir = ws / "04_audio" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    kokoro = make_tts(narrator_id=episode["narrator"])

    chunk_results = kokoro.synthesize_script(
        speech_only, chunks_dir, max_words_per_chunk=300
    )
    if not chunk_results:
        return "TTS produced no chunks"

    # Concatenate with crossfade
    crossfade_ms = 80
    crossfade_samples = int(KOKORO_SAMPLE_RATE * crossfade_ms / 1000)

    voice_chunks: list[np.ndarray] = []
    for r in chunk_results:
        audio, sr = sf.read(str(r.wav_path), dtype="float32")
        if sr != KOKORO_SAMPLE_RATE:
            logger.warning("unexpected sample rate %d in %s", sr, r.wav_path)
        voice_chunks.append(audio)

    full = _crossfade_concat(voice_chunks, crossfade_samples)
    voice_full = ws / "04_audio" / "voice_full.wav"
    sf.write(str(voice_full), full, KOKORO_SAMPLE_RATE, subtype="PCM_16")
    voice_total_seconds = len(full) / KOKORO_SAMPLE_RATE
    logger.info("voice_full.wav: %.1fs", voice_total_seconds)

    # Per-beat timing
    n_chars = max(1, len(speech_only))
    beat_timing = []
    for idx, (bnum, char_pos) in enumerate(beat_positions):
        start = (char_pos / n_chars) * voice_total_seconds
        if idx + 1 < len(beat_positions):
            end_char = beat_positions[idx + 1][1]
            end = (end_char / n_chars) * voice_total_seconds
        else:
            end = voice_total_seconds
        beat_timing.append({
            "beat_id": f"BEAT_{bnum:02d}",
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
        })

    (ws / "04_audio" / "voice_timing.json").write_text(json.dumps({
        "total_seconds": voice_total_seconds,
        "sample_rate": KOKORO_SAMPLE_RATE,
        "beats": beat_timing,
        "narrator": episode["narrator"],
    }, indent=2))

    tgt = cfg.production["target_duration_seconds"]
    tol = cfg.production["duration_tolerance_seconds"]
    if abs(voice_total_seconds - tgt) > tol:
        logger.warning("voice duration %.1fs outside target %d±%d",
                       voice_total_seconds, tgt, tol)

    logger.info("S10 complete: %d beats, %.1fs voice",
                len(beat_timing), voice_total_seconds)
    return None


# ------------------ helpers ------------------

def _crossfade_concat(chunks: list[np.ndarray], n: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0].copy()
    for nxt in chunks[1:]:
        if len(out) < n or len(nxt) < n:
            out = np.concatenate([out, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
        tail = out[-n:] * fade_out + nxt[:n] * fade_in
        out = np.concatenate([out[:-n], tail, nxt[n:]])
    return out.astype(np.float32)


def _convert_numbers(text: str) -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text

    def year_replace(m: re.Match[str]) -> str:
        y = int(m.group(0))
        try:
            return num2words(y, to="year")
        except Exception:
            return m.group(0)

    text = YEAR_RE.sub(year_replace, text)

    def num_replace(m: re.Match[str]) -> str:
        n = int(m.group(0))
        if n > 99999:
            return m.group(0)
        try:
            return num2words(n)
        except Exception:
            return m.group(0)

    text = NUMBER_RE.sub(num_replace, text)
    return text


def _pause_to_punct(m: re.Match[str]) -> str:
    seconds = float(m.group(1))
    if seconds < 1.5:
        return ", "
    if seconds < 3:
        return " ... "
    return " ... ... "
