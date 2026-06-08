"""Chatterbox TTS adapter for the oMLX audio gateway.

The local production gateway exposes an OpenAI-style endpoint:

    POST /v1/audio/speech

This adapter mirrors the Kokoro/ElevenLabs surface used by S10 and
Shorts while allowing optional narrator-specific reference audio for
voice cloning.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import load_config

logger = logging.getLogger("hermes.chatterbox")

DEFAULT_BASE_URL = "http://10.0.4.250:9000/v1/audio/speech"
DEFAULT_MODEL_ID = "chatterbox-turbo-4bit"
SAMPLE_RATE = 24000


@dataclass
class ChatterboxChunk:
    wav_path: Path
    text: str
    voice: str


class ChatterboxTTS:
    """Narrator-aware Chatterbox adapter with optional voice refs."""

    def __init__(self, narrator_id: str):
        cfg = load_config()
        narrator = cfg.narrator_by_id(narrator_id)
        cb_cfg = cfg.tts.get("chatterbox") or {}
        self._cfg_root = cfg.root
        self._mock: bool = cfg.mock_mode
        self.narrator_id = narrator_id
        self.base_url: str = cb_cfg.get("base_url") or DEFAULT_BASE_URL
        self.model_id: str = cb_cfg.get("model_id") or DEFAULT_MODEL_ID
        self.response_format: str = cb_cfg.get("response_format") or "wav"
        self.timeout_seconds: float = float(cb_cfg.get("timeout_seconds", 300))
        self.max_words_per_chunk: int = int(cb_cfg.get("max_words_per_chunk", 180))
        self.speed: float = float(cb_cfg.get("speed") or narrator.get("speed", 1.0) or 1.0)
        self.language: str = cb_cfg.get("language") or ""
        self.voice: str = self._resolve_voice(cb_cfg, narrator)
        self.instructions: str = self._resolve_map_value(
            cb_cfg.get("instructions_map") or {},
            cb_cfg.get("instructions") or "",
        )
        self.ref_audio_path, self.ref_text = self._resolve_voice_reference(cb_cfg)
        self.temperature = cb_cfg.get("temperature")
        self.top_k = cb_cfg.get("top_k")
        self.top_p = cb_cfg.get("top_p")
        self.repetition_penalty = cb_cfg.get("repetition_penalty")
        self.max_tokens = cb_cfg.get("max_tokens")
        self._api_key = self._resolve_api_key(cb_cfg)

    def synthesize_script(
        self,
        text: str,
        output_dir: Path | str,
        *,
        max_words_per_chunk: int = 300,
    ) -> list[ChatterboxChunk]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        chunk_limit = max(1, min(int(max_words_per_chunk), self.max_words_per_chunk))
        chunks = _chunk_text(text, chunk_limit)
        if not chunks:
            return []

        results: list[ChatterboxChunk] = []
        ref_audio_b64 = _load_ref_audio(self.ref_audio_path) if self.ref_audio_path else ""
        for i, chunk in enumerate(chunks):
            wav = out_dir / f"chunk_{i:03d}.wav"
            if self._mock:
                _write_silent_wav(wav, _estimate_duration(chunk, self.speed))
            else:
                self._render_one(chunk, wav, ref_audio_b64=ref_audio_b64)
            results.append(ChatterboxChunk(wav_path=wav, text=chunk, voice=self.voice))
        return results

    def _render_one(self, text: str, out: Path, *, ref_audio_b64: str = "") -> None:
        payload = self._payload(text, ref_audio_b64=ref_audio_b64)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        logger.info(
            "chatterbox: %d words, voice=%s, model=%s, speed=%.2f -> %s",
            len(text.split()), self.voice or "default", self.model_id,
            self.speed, out.name,
        )
        r = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"chatterbox server returned HTTP {r.status_code}: {r.text[:400]}"
            )
        if not r.content:
            raise RuntimeError("chatterbox server returned an empty audio payload")
        out.write_bytes(r.content)

    def _payload(self, text: str, *, ref_audio_b64: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "input": text,
            "response_format": self.response_format,
        }
        optional = {
            "voice": self.voice or None,
            "instructions": self.instructions or None,
            "ref_audio": ref_audio_b64 or None,
            "ref_text": self.ref_text or None,
            "language": self.language or None,
            "speed": self.speed,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
        }
        payload.update({k: v for k, v in optional.items() if v not in (None, "")})
        return payload

    def _resolve_voice(self, cb_cfg: dict[str, Any], narrator: dict[str, Any]) -> str:
        voice_map = cb_cfg.get("voice_map") or {}
        return (
            voice_map.get(self.narrator_id)
            or cb_cfg.get("voice")
            or narrator.get("voice")
            or ""
        )

    def _resolve_map_value(self, mapping: dict[str, Any], fallback: str) -> str:
        value = mapping.get(self.narrator_id) if isinstance(mapping, dict) else None
        return str(value if value is not None else fallback).strip()

    def _resolve_voice_reference(self, cb_cfg: dict[str, Any]) -> tuple[Path | None, str]:
        ref_map = cb_cfg.get("voice_ref_map") or {}
        entry = ref_map.get(self.narrator_id) if isinstance(ref_map, dict) else None
        ref_audio = cb_cfg.get("ref_audio") or ""
        ref_text = cb_cfg.get("ref_text") or ""
        if isinstance(entry, str):
            ref_audio = entry
        elif isinstance(entry, dict):
            ref_audio = entry.get("path") or entry.get("ref_audio") or ref_audio
            ref_text = entry.get("ref_text") or ref_text
        path = self._resolve_path(ref_audio) if ref_audio else None
        ref_text = str(ref_text or "").strip()
        if path and not ref_text:
            ref_text = str(cb_cfg.get("ref_text_fallback") or "").strip()
            if ref_text:
                logger.warning(
                    "chatterbox: narrator %s uses ref_audio without exact ref_text; "
                    "using configured ref_text_fallback",
                    self.narrator_id,
                )
        return path, ref_text

    def _resolve_path(self, value: str) -> Path:
        raw = str(value).replace("${root}", str(self._cfg_root))
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self._cfg_root / path
        return path

    @staticmethod
    def _resolve_api_key(cb_cfg: dict[str, Any]) -> str:
        env_name = cb_cfg.get("api_key_env") or "OMLX_API_KEY"
        api_key = (os.environ.get(env_name) or os.environ.get("OMLX_API_KEY") or "").strip()
        if api_key:
            return api_key
        return str(cb_cfg.get("api_key") or "pass123").strip()


# -------------------- helpers --------------------

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


def _load_ref_audio(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Chatterbox reference audio not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _estimate_duration(text: str, speed: float) -> float:
    words = max(1, len(text.split()))
    cfg = load_config()
    wpm = max(1.0, float(cfg.production.get("wpm_effective", 150)))
    base_seconds = (words / wpm) * 60.0
    return max(0.5, base_seconds / max(0.5, speed))


def _write_silent_wav(path: Path, seconds: float) -> None:
    n_frames = int(round(seconds * SAMPLE_RATE))
    silence = b"\x00\x00" * n_frames
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(silence)
