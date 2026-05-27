"""ElevenLabs TTS adapter — wired but disabled by default.

Drop-in replacement for the Kokoro adapter at the same call site.
S10 selects between backends via `cfg.tts.backend ∈ {kokoro, elevenlabs}`.
ElevenLabs is the perceived-quality step change (premium voice
synthesis vs. Kokoro's mechanical edge); enable when you're ready to
pay $5-22/mo and want the channel to read like a real documentary.

Setup:
  1. Put ELEVENLABS_API_KEY in .env
  2. Set cfg.tts.backend: elevenlabs in config.yaml
  3. Set cfg.tts.elevenlabs.enabled: true
  4. Optionally pin per-narrator voice IDs in cfg.tts.elevenlabs.voice_id_map

The default voice_id ("Adam" preset) is the safe starting point. The
real-time API at /text-to-speech/{voice_id} returns MP3 by default;
we request `wav` (audio/wav) so S10's downstream pipeline doesn't
need to convert.

Falls back gracefully:
  - On HTTP failure or missing key, the adapter logs and re-raises;
    S10 catches and the orchestrator transitions the stage to
    needs_human. The operator can flip tts.backend back to kokoro
    in config and re-run.
  - In mock_mode, returns silent WAVs at 24kHz (same shape as the
    Kokoro mock).

Added Batch D 2026-05-27.
"""

from __future__ import annotations

import logging
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import load_config

logger = logging.getLogger("hermes.elevenlabs")

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # "Adam" preset
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
SAMPLE_RATE = 24000  # matches Kokoro for downstream symmetry


@dataclass
class ElevenLabsChunk:
    wav_path: Path
    text: str
    voice: str


def _resolve_api_key() -> str:
    """Read ELEVENLABS_API_KEY from env. Falls back to config.tts.
    elevenlabs.api_key only as a last resort (and warns since
    config-committed keys trip GitHub secret scanning)."""
    for var in ("ELEVENLABS_API_KEY",):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    cfg = load_config()
    tts = cfg.tts.get("elevenlabs", {})
    val = (tts.get("api_key") or "").strip()
    if val:
        logger.warning(
            "ElevenLabs API key in config.yaml — move it to .env; "
            "GitHub secret-scanning will reject pushes."
        )
    return val


class ElevenLabsTTS:
    """ElevenLabs adapter with the same surface as Kokoro."""

    def __init__(self, narrator_id: str):
        cfg = load_config()
        narrator = cfg.narrator_by_id(narrator_id)
        el_cfg = cfg.tts.get("elevenlabs") or {}
        self._mock: bool = cfg.mock_mode
        self.narrator_id: str = narrator_id

        # Per-narrator voice mapping (operator-set in config). Falls
        # back to the global default voice_id, then to "Adam".
        voice_map = el_cfg.get("voice_id_map") or {}
        self.voice_id: str = (
            voice_map.get(narrator_id)
            or el_cfg.get("voice_id")
            or DEFAULT_VOICE_ID
        )
        self.model_id: str = el_cfg.get("model_id") or DEFAULT_MODEL_ID
        self.speed: float = float(narrator.get("speed", 1.0))
        self._api_key = _resolve_api_key()

    @property
    def voice(self) -> str:
        """Match Kokoro's `voice` attribute name so existing callers
        that log/serialize it don't break."""
        return self.voice_id

    def synthesize_script(
        self,
        text: str,
        output_dir: Path | str,
        *,
        max_words_per_chunk: int = 300,
    ) -> list[ElevenLabsChunk]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        chunks = _chunk_text(text, max_words_per_chunk)
        if not chunks:
            return []

        results: list[ElevenLabsChunk] = []
        for i, chunk in enumerate(chunks):
            wav = out_dir / f"chunk_{i:03d}.wav"
            if self._mock:
                _write_silent_wav(wav, _estimate_duration(chunk, self.speed))
            else:
                self._render_one(chunk, wav)
            results.append(ElevenLabsChunk(
                wav_path=wav, text=chunk, voice=self.voice_id,
            ))
        return results

    def _render_one(self, text: str, out: Path) -> None:
        if not self._api_key:
            raise RuntimeError(
                "ElevenLabs API key not found — set ELEVENLABS_API_KEY in .env"
            )
        url = f"{ELEVENLABS_BASE}/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            # ElevenLabs doesn't take a speed parameter directly;
            # voice_settings can dial in stability + similarity.
            # We use defaults the operator can tune later.
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        logger.info("elevenlabs: %d words, voice=%s, model=%s -> %s",
                    len(text.split()), self.voice_id, self.model_id,
                    out.name)
        r = requests.post(url, headers=headers, json=payload, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(
                f"elevenlabs API returned HTTP {r.status_code}: "
                f"{r.text[:300]}"
            )
        out.write_bytes(r.content)


# -------------------- helpers (mirror tts.py) --------------------

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _chunk_text(text: str, max_words: int) -> list[str]:
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
    words = max(1, len(text.split()))
    base_seconds = (words / 120.0) * 60.0
    return max(0.5, base_seconds / max(0.5, speed))


def _write_silent_wav(path: Path, seconds: float) -> None:
    n_frames = int(round(seconds * SAMPLE_RATE))
    silence = b"\x00\x00" * n_frames
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(silence)
