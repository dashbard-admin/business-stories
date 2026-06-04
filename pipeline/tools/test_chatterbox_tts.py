"""Generate Chatterbox TTS samples through the oMLX audio endpoint.

This is a test-only tool, not a production pipeline backend.

Examples:
  python3 -m pipeline.tools.test_chatterbox_tts --voice female --target-wpm 160
  python3 -m pipeline.tools.test_chatterbox_tts --voice male --target-wpm 140 --target-wpm 180
  python3 -m pipeline.tools.test_chatterbox_tts --ref-audio /path/on/server/voice.wav --ref-text "Reference transcript"

The oMLX route is OpenAI-style:
  POST /v1/audio/speech
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Any

import requests

from pipeline.ffmpeg_builder import time_stretch_audio


DEFAULT_BASE_URL = "http://10.0.4.250:9000/v1/audio/speech"
DEFAULT_MODEL = "Chatterbox-Multilingual-MLX-v2-Q4"
DEFAULT_TEXT = (
    "A small team thought six seconds could change the internet. "
    "For a while, they were right. Then the product that felt magical "
    "started pulling itself apart."
)

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m pipeline.tools.test_chatterbox_tts",
        description="Generate Chatterbox TTS samples via oMLX.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OMLX_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        help="Voice string to send. Repeat to test several voices.",
    )
    parser.add_argument(
        "--instructions",
        action="append",
        default=[],
        help="Optional style/instruction string. Repeat to test several.",
    )
    parser.add_argument(
        "--ref-audio",
        action="append",
        default=[],
        help=(
            "Optional reference audio string/path accepted by oMLX. "
            "If running this script from another machine, this path must "
            "still be meaningful to the oMLX server."
        ),
    )
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--response-format", default="wav")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/chatterbox_tts_tests"))
    parser.add_argument("--max-words-per-chunk", type=int, default=80)
    parser.add_argument("--concat-pause-ms", type=int, default=180)
    parser.add_argument(
        "--target-wpm",
        action="append",
        type=float,
        default=[],
        help="Write a pitch-preserving ffmpeg-stretched full WAV at this WPM.",
    )
    args = parser.parse_args()

    text = args.text_file.read_text(encoding="utf-8").strip() if args.text_file else args.text
    text = _clean_text(text)
    variants = _build_variants(args)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    api_key = args.api_key or os.environ.get(args.api_key_env, "") or "pass123"

    for idx, variant in enumerate(variants, start=1):
        variant_name = f"sample_{idx:02d}"
        sample_dir = out_dir / variant_name
        chunks_dir = sample_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks = _chunk_text(text, args.max_words_per_chunk)
        chunk_paths: list[Path] = []
        t0 = time.time()

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_path = chunks_dir / f"chunk_{chunk_idx:03d}.wav"
            payload = _payload(args, variant, chunk_text)
            try:
                _request_audio(
                    args.base_url,
                    api_key,
                    payload,
                    chunk_path,
                    timeout=args.timeout_seconds,
                )
            except Exception as e:
                rec = {
                    "sample": variant_name,
                    "ok": False,
                    "error": repr(e),
                    "payload": _redact_payload(payload),
                }
                manifest.append(rec)
                print(f"FAIL {variant_name} chunk {chunk_idx:03d}: {e}")
                break
            chunk_paths.append(chunk_path)
        else:
            raw_full = sample_dir / "raw_full.wav"
            _concat_wavs(chunk_paths, raw_full, pause_ms=args.concat_pause_ms)
            words = _word_count(text)
            raw_info = _wav_info(raw_full)
            rec = {
                "sample": variant_name,
                "ok": True,
                "model": args.model,
                "base_url": args.base_url,
                "voice": variant.get("voice"),
                "instructions": variant.get("instructions"),
                "ref_audio": variant.get("ref_audio"),
                "ref_text": args.ref_text or None,
                "language": args.language or None,
                "speed": args.speed,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_tokens": args.max_tokens,
                "text": text,
                "word_count": words,
                "chunk_count": len(chunk_paths),
                "chunk_paths": [str(p) for p in chunk_paths],
                "raw_full_path": str(raw_full),
                "raw_actual_wpm": _actual_wpm(words, raw_info),
                "elapsed_seconds": round(time.time() - t0, 3),
                **{f"raw_{k}": v for k, v in raw_info.items()},
                "targets": [],
            }
            for target_wpm in args.target_wpm:
                rec["targets"].append(
                    _write_target_wpm_variant(raw_full, words, target_wpm)
                )
            manifest.append(rec)
            print(
                f"OK {variant_name}: {raw_full} "
                f"duration={raw_info.get('duration_seconds')}s "
                f"wpm={rec.get('raw_actual_wpm')} "
                f"sha={raw_info.get('sha256', '')[:12]}"
            )
            for target in rec["targets"]:
                if target.get("path"):
                    print(
                        f"   target {target['target_wpm']:g}wpm: "
                        f"{target['path']} duration={target.get('duration_seconds')}s "
                        f"actual_wpm={target.get('actual_wpm')}"
                    )
                else:
                    print(f"   target {target['target_wpm']:g}wpm failed: {target.get('error')}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


def _build_variants(args: argparse.Namespace) -> list[dict[str, str | None]]:
    voices: list[str | None] = args.voice or [None]
    instructions: list[str | None] = args.instructions or [None]
    ref_audios: list[str | None] = args.ref_audio or [None]
    variants: list[dict[str, str | None]] = []
    for voice in voices:
        for instruction in instructions:
            for ref_audio in ref_audios:
                variants.append({
                    "voice": voice,
                    "instructions": instruction,
                    "ref_audio": ref_audio,
                })
    return variants


def _payload(args: argparse.Namespace, variant: dict[str, str | None], text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "input": text,
        "response_format": args.response_format,
    }
    optional = {
        "voice": variant.get("voice"),
        "instructions": variant.get("instructions"),
        "ref_audio": variant.get("ref_audio"),
        "ref_text": args.ref_text or None,
        "language": args.language or None,
        "speed": args.speed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    return payload


def _request_audio(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    out: Path,
    *,
    timeout: float,
) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:600]}")
    out.write_bytes(r.content)
    if out.stat().st_size <= 44:
        raise RuntimeError(f"audio response too small: {out.stat().st_size} bytes")


def _chunk_text(text: str, max_words: int) -> list[str]:
    max_words = max(1, int(max_words))
    sentences = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for sent in sentences:
        n = _word_count(sent)
        if buf and buf_words + n > max_words:
            chunks.append(" ".join(buf))
            buf, buf_words = [sent], n
        else:
            buf.append(sent)
            buf_words += n
    if buf:
        chunks.append(" ".join(buf))
    return chunks or [text]


def _concat_wavs(paths: list[Path], out: Path, *, pause_ms: int) -> None:
    if not paths:
        raise RuntimeError("no WAV chunks to concatenate")
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    pause_frames = int(round(params.framerate * (max(0, pause_ms) / 1000.0)))
    silence = b"\x00" * pause_frames * params.sampwidth * params.nchannels
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        for idx, path in enumerate(paths):
            with wave.open(str(path), "rb") as r:
                if r.getnchannels() != params.nchannels or r.getsampwidth() != params.sampwidth:
                    raise RuntimeError(f"WAV format mismatch: {path}")
                frames = r.readframes(r.getnframes())
            if idx:
                w.writeframes(silence)
            w.writeframes(frames)


def _write_target_wpm_variant(raw_path: Path, words: int, target_wpm: float) -> dict[str, Any]:
    target_wpm = max(1.0, float(target_wpm))
    target_seconds = (words / target_wpm) * 60.0
    label = f"{target_wpm:g}".replace(".", "p")
    target_path = raw_path.with_name(f"{raw_path.stem}_{label}wpm.wav")
    try:
        time_stretch_audio(raw_path, target_path, target_seconds)
        info = _wav_info(target_path)
        return {
            "target_wpm": target_wpm,
            "path": str(target_path),
            "requested_duration_seconds": round(target_seconds, 3),
            "actual_wpm": _actual_wpm(words, info),
            **info,
        }
    except Exception as e:
        return {"target_wpm": target_wpm, "error": repr(e)}


def _wav_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    info: dict[str, Any] = {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    try:
        with wave.open(str(path), "rb") as w:
            info.update({
                "sample_rate": w.getframerate(),
                "channels": w.getnchannels(),
                "sample_width": w.getsampwidth(),
                "duration_seconds": round(w.getnframes() / w.getframerate(), 3),
            })
    except Exception as e:
        info["wav_error"] = repr(e)
    return info


def _actual_wpm(words: int, wav_info: dict[str, Any]) -> float | None:
    duration = wav_info.get("duration_seconds")
    if not duration:
        return None
    return round((words / float(duration)) * 60.0, 1)


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _clean_text(text: str) -> str:
    text = re.sub(r"\\([,.:;?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload)


if __name__ == "__main__":
    raise SystemExit(main())
