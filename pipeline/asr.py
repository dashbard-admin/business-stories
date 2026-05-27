"""ASR adapter (Batch D 2026-05-27) — whisper.cpp wrapper for Shorts
subtitles. Used by pipeline/shorts.py to generate word-level captions
on the vertical Short cuts.

Whisper.cpp is the recommended backend (D1 confirmed):
  - Free, local, no API key
  - Reasonable accuracy at the base.en model size
  - Runs ~30s per 30-second Short on a 2024 Mac

If whisper.cpp isn't installed on PATH (binary name configurable via
cfg.asr.binary), the adapter logs a warning and returns None — the
Shorts get generated WITHOUT subtitles (just the cut + audio). The
operator can install whisper.cpp later and re-run.

Mock mode returns a canned subtitle list so tests don't need a
binary.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import load_config

logger = logging.getLogger("hermes.asr")


@dataclass
class Segment:
    start_seconds: float
    end_seconds: float
    text: str


def _binary_available(name: str) -> bool:
    return shutil.which(name) is not None


def transcribe(
    wav_path: Path,
    *,
    model: str | None = None,
) -> list[Segment] | None:
    """Run whisper.cpp on `wav_path` and return word-level segments,
    or None if whisper.cpp is unavailable or the call fails.

    Returns segments via the --output-json flag (whisper.cpp >= 1.5).
    """
    cfg = load_config()
    asr_cfg = cfg.asr

    if cfg.mock_mode:
        # Mock: one second of "Mock subtitle" per ~1s of audio.
        try:
            import soundfile as sf
            info = sf.info(str(wav_path))
            duration = info.frames / info.samplerate
        except Exception:
            duration = 30.0
        out: list[Segment] = []
        t = 0.0
        while t < duration:
            out.append(Segment(
                start_seconds=t,
                end_seconds=min(t + 1.5, duration),
                text="Mock subtitle",
            ))
            t += 1.5
        return out

    binary = asr_cfg.get("binary", "whisper-cli")
    if not _binary_available(binary):
        logger.warning(
            "ASR: %s not on PATH; Shorts will be generated without "
            "subtitles. Install whisper.cpp (https://github.com/"
            "ggerganov/whisper.cpp) and retry to enable.", binary,
        )
        return None

    model = model or asr_cfg.get("model", "base.en")
    json_out = wav_path.with_suffix(".whisper.json")

    cmd = [
        binary,
        "-m", str(asr_cfg.get("model_path") or model),
        "-f", str(wav_path),
        "--output-json-full",
        "--output-file", str(json_out.with_suffix("")),
        "--no-prints",
    ]
    try:
        subprocess.run(
            cmd, check=True,
            capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("whisper.cpp failed: %s — stderr=%s",
                       e.returncode, (e.stderr or "")[:300])
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("whisper.cpp invocation failed: %s", e)
        return None

    if not json_out.exists():
        logger.warning("whisper.cpp produced no JSON at %s", json_out)
        return None

    try:
        data = json.loads(json_out.read_text())
    except Exception as e:
        logger.warning("whisper.cpp JSON unparseable: %s", e)
        return None

    segs: list[Segment] = []
    for s in data.get("transcription") or []:
        # whisper.cpp emits offsets in milliseconds
        t = (s.get("offsets") or {})
        start = float(t.get("from", 0)) / 1000.0
        end = float(t.get("to", 0)) / 1000.0
        text = (s.get("text") or "").strip()
        if text:
            segs.append(Segment(start, end, text))
    return segs
