"""Qwen3-TTS VoiceDesign adapter for the local oMLX gateway.

The model `Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` is exposed by oMLX
through the OpenAI-style `/v1/audio/speech` route. In testing on
10.0.4.250, the useful voice-control field is `instructions`; `voice`
alone is ignored/invalid for this VoiceDesign model and returns HTTP
500. This adapter therefore treats per-narrator natural-language
instructions as the voice identity.
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

logger = logging.getLogger("hermes.qwen3_tts")

DEFAULT_BASE_URL = "http://10.0.4.250:9000/v1/audio/speech"
DEFAULT_MODEL_ID = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
SAMPLE_RATE = 24000


@dataclass
class Qwen3TTSChunk:
    wav_path: Path
    text: str
    voice: str


class Qwen3TTS:
    """VoiceDesign adapter with the same surface as Kokoro."""

    def __init__(self, narrator_id: str):
        cfg = load_config()
        narrator = cfg.narrator_by_id(narrator_id)
        q_cfg = cfg.tts.get("qwen3") or {}
        self._mock: bool = cfg.mock_mode
        self.narrator_id: str = narrator_id
        self.base_url: str = q_cfg.get("base_url") or DEFAULT_BASE_URL
        self.model_id: str = q_cfg.get("model_id") or DEFAULT_MODEL_ID
        self.response_format: str = q_cfg.get("response_format") or "wav"
        self.temperature = q_cfg.get("temperature")
        self.top_p = q_cfg.get("top_p")
        self.timeout_seconds: int = int(q_cfg.get("timeout_seconds") or 300)
        self.send_voice_field: bool = bool(q_cfg.get("send_voice_field", False))
        self.speed: float = float(narrator.get("speed", 1.0))
        self.kokoro_voice: str = narrator.get("voice") or narrator_id
        self.voice_instruction: str = self._resolve_voice_instruction(
            q_cfg, narrator_id, narrator,
        )
        self._api_key = _resolve_api_key(q_cfg)

    @property
    def voice(self) -> str:
        return self.voice_instruction

    def synthesize_script(
        self,
        text: str,
        output_dir: Path | str,
        *,
        max_words_per_chunk: int = 300,
    ) -> list[Qwen3TTSChunk]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        chunks = _chunk_text(text, max_words_per_chunk)
        if not chunks:
            return []

        results: list[Qwen3TTSChunk] = []
        for i, chunk in enumerate(chunks):
            wav = out_dir / f"chunk_{i:03d}.wav"
            if self._mock:
                _write_silent_wav(wav, _estimate_duration(chunk, self.speed))
            else:
                self._render_one(chunk, wav)
            results.append(Qwen3TTSChunk(
                wav_path=wav,
                text=chunk,
                voice=self.voice_instruction,
            ))
        return results

    def _render_one(self, text: str, out: Path) -> None:
        payload: dict[str, object] = {
            "model": self.model_id,
            "input": text,
            "instructions": self.voice_instruction,
            "speed": self.speed,
            "response_format": self.response_format,
        }
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.send_voice_field:
            # oMLX's Qwen3 VoiceDesign path is instruction-driven, but
            # this switch is useful for future server builds that also
            # interpret the OpenAI-compatible `voice` field.
            payload["voice"] = self.kokoro_voice

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        logger.info(
            "qwen3_tts: %d words, narrator=%s, speed=%.2f -> %s",
            len(text.split()), self.narrator_id, self.speed, out.name,
        )
        r = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"Qwen3-TTS returned HTTP {r.status_code}: {r.text[:500]}"
            )
        out.write_bytes(r.content)

    @staticmethod
    def _resolve_voice_instruction(
        q_cfg: dict,
        narrator_id: str,
        narrator: dict,
    ) -> str:
        instruction_map = q_cfg.get("voice_instruction_map") or {}
        mapped = (instruction_map.get(narrator_id) or "").strip()
        if mapped:
            return mapped
        fallback = (q_cfg.get("voice_instruction") or "").strip()
        if fallback:
            return fallback
        tone = (narrator.get("tone") or "clear documentary").strip()
        return (
            f"Speak in a {tone} narrator voice. Use natural pacing, "
            "clean articulation, and documentary-style authority."
        )


def _resolve_api_key(q_cfg: dict) -> str:
    env_names = [
        q_cfg.get("api_key_env") or "",
        "OMLX_API_KEY",
        "QWEN3_TTS_API_KEY",
    ]
    for env_name in env_names:
        if not env_name:
            continue
        value = (os.environ.get(env_name) or "").strip()
        if value:
            return value
    # The local gateway has historically used this development token.
    return "pass123"


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
