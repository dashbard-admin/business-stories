"""Kokoro TTS adapter.

Calls the local mlx-audio Kokoro server at
http://127.0.0.1:8001/v1/audio/speech (OpenAI-compatible API) and writes
per-chunk WAVs into the directory S10 hands us. Narrator voice + speed
come from `config.yaml`'s `narrators` block via `cfg.narrator_by_id()`.

In mock mode, synthesizes silent WAVs at the same sample rate so the
rest of the pipeline can run end-to-end without the GPU server.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import load_config

TTS_BASE = "http://127.0.0.1:8001/v1/audio/speech"
KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"
KOKORO_SAMPLE_RATE = 24000

logger = logging.getLogger("hermes.tts")


@dataclass
class TTSChunk:
    wav_path: Path
    text: str
    voice: str


class Kokoro:
    """Narrator-aware adapter for the local Kokoro server."""

    def __init__(self, narrator_id: str):
        cfg = load_config()
        narrator = cfg.narrator_by_id(narrator_id)
        self._mock: bool = cfg.mock_mode
        self.narrator_id: str = narrator_id
        self.voice: str = narrator["voice"]
        self.speed: float = float(narrator.get("speed", 1.0))

    def synthesize_script(
        self,
        text: str,
        output_dir: Path | str,
        *,
        max_words_per_chunk: int = 300,
    ) -> list[TTSChunk]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        chunks = _chunk_text(text, max_words_per_chunk)
        if not chunks:
            return []

        results: list[TTSChunk] = []
        for i, chunk in enumerate(chunks):
            wav = out_dir / f"chunk_{i:03d}.wav"
            if self._mock:
                _write_silent_wav(wav, _estimate_duration(chunk, self.speed))
            else:
                self._render_one(chunk, wav)
            results.append(TTSChunk(wav_path=wav, text=chunk, voice=self.voice))
        return results

    def _render_one(self, text: str, out: Path) -> None:
        payload = {
            "model": KOKORO_MODEL,
            "input": text,
            "voice": self.voice,
            "speed": self.speed,
            "response_format": "wav",
        }
        logger.info("kokoro: %d words, voice=%s, speed=%.2f -> %s",
                    len(text.split()), self.voice, self.speed, out.name)
        r = requests.post(TTS_BASE, json=payload, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(
                f"kokoro server returned HTTP {r.status_code}: {r.text[:300]}"
            )
        out.write_bytes(r.content)


# -------------------- helpers --------------------

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _chunk_text(text: str, max_words: int) -> list[str]:
    """Pack sentences into chunks up to `max_words`. Long sentences pass
    through whole — splitting mid-sentence would break prosody."""
    sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for sent in sentences:
        n = len(sent.split())
        if buf and buf_words + n > max_words:
            chunks.append(" ".join(buf))
            buf, buf_words = [sent], n
        else:
            buf.append(sent)
            buf_words += n
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _estimate_duration(text: str, speed: float) -> float:
    # 120 wpm matches the cadence the writer prompt is calibrated
    # against (Act 0-5 timing schedule in script_generate.txt).
    words = max(1, len(text.split()))
    base_seconds = (words / 120.0) * 60.0
    return max(0.5, base_seconds / max(0.5, speed))


def _write_silent_wav(path: Path, seconds: float) -> None:
    import wave
    n_frames = int(round(seconds * KOKORO_SAMPLE_RATE))
    silence = b"\x00\x00" * n_frames  # PCM_16 mono
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(KOKORO_SAMPLE_RATE)
        w.writeframes(silence)
