"""ASR adapter (Batch D 2026-05-27) — whisper.cpp wrapper.

Used by pipeline/shorts.py to generate word-level captions on the
vertical Short cuts, and by S12 to align long-form subtitles/callouts
to the rendered voice track when local ASR is available.

Whisper.cpp is the recommended backend (D1 confirmed):
  - Free, local, no API key
  - Reasonable accuracy at the base.en model size
  - Runs ~30s per 30-second Short on a 2024 Mac

If whisper.cpp isn't installed on PATH (binary name configurable via
cfg.asr.binary), the adapter logs a warning and returns None. Callers
fall back to their non-ASR timing path.

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
    wav_path = Path(wav_path)

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
            "ASR: %s not on PATH; falling back to estimated subtitle/"
            "callout timings. Install whisper.cpp (https://github.com/"
            "ggerganov/whisper.cpp) and retry to enable ASR alignment.",
            binary,
        )
        return None

    if not wav_path.exists():
        logger.warning("ASR: input audio not found: %s", wav_path)
        return None

    model_path = _resolve_model_path(asr_cfg, cfg.root, model)
    if not model_path.exists():
        logger.warning(
            "ASR: whisper.cpp model not found: %s. Download one, e.g. "
            "models/whisper/ggml-base.en.bin, or set asr.model_path.",
            model_path,
        )
        return None

    json_out = wav_path.with_suffix(".whisper.json")

    cmd = [
        binary,
        "-m", str(model_path),
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
        stderr = (e.stderr or "").strip()
        logger.warning(
            "whisper.cpp failed: %s — stderr=%s",
            e.returncode, stderr[-1000:],
        )
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


def _resolve_model_path(
    asr_cfg: dict,
    root: Path,
    model: str | None,
) -> Path:
    configured = str(asr_cfg.get("model_path") or "").strip()
    if configured:
        raw = configured.replace("${root}", str(root))
        path = Path(raw).expanduser()
        return path if path.is_absolute() else root / path

    model_name = str(model or asr_cfg.get("model") or "base.en").strip()
    if model_name.endswith(".bin") or "/" in model_name:
        path = Path(model_name).expanduser()
        return path if path.is_absolute() else root / path

    candidates = [
        root / "models" / "whisper" / f"ggml-{model_name}.bin",
        root / "models" / "whisper" / model_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
