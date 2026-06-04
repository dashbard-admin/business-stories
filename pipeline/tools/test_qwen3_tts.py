"""Generate Qwen3-TTS VoiceDesign samples and a manifest.

Usage:
  python3 -m pipeline.tools.test_qwen3_tts --narrator N5
  python3 -m pipeline.tools.test_qwen3_tts --voice-instruction "Speak..."

The oMLX server route tested for this project is `/v1/audio/speech`.
For the VoiceDesign model, change voices by changing `instructions`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

from pipeline.config import load_config
from pipeline.ffmpeg_builder import time_stretch_audio
from pipeline.qwen3_tts import Qwen3TTS


DEFAULT_TEXT = (
    "A small team thought six seconds could change the internet. "
    "For a while, they were right. Then the very thing that made the "
    "product magical began to pull it apart."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m pipeline.tools.test_qwen3_tts",
        description="Generate Qwen3-TTS VoiceDesign samples.",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--narrator", action="append", default=[])
    parser.add_argument("--voice-instruction", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/qwen3_tts_tests"))
    parser.add_argument("--max-words-per-chunk", type=int, default=300)
    parser.add_argument(
        "--target-wpm",
        type=float,
        help=(
            "Also write a pitch-preserving ffmpeg-stretched WAV at this "
            "exact words-per-minute target."
        ),
    )
    args = parser.parse_args()

    text = args.text
    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8").strip()

    cfg = load_config()
    narrator_ids = args.narrator or [n["id"] for n in cfg.narrators if n.get("enabled")]
    if not narrator_ids:
        narrator_ids = ["N1"]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for narrator_id in narrator_ids:
        instructions = args.voice_instruction or [
            _instruction_for_narrator(narrator_id)
        ]
        for idx, instruction in enumerate(instructions, start=1):
            tts = Qwen3TTS(narrator_id)
            tts.voice_instruction = instruction
            sample_dir = out_dir / f"{narrator_id}_{idx:02d}"
            t0 = time.time()
            try:
                chunks = tts.synthesize_script(
                    text,
                    sample_dir,
                    max_words_per_chunk=args.max_words_per_chunk,
                )
            except Exception as e:
                manifest.append({
                    "narrator_id": narrator_id,
                    "instruction_index": idx,
                    "ok": False,
                    "error": repr(e),
                    "voice_instruction": instruction,
                })
                print(f"FAIL {narrator_id}_{idx:02d}: {e}")
                continue

            for chunk in chunks:
                info = _wav_info(chunk.wav_path)
                words = _word_count(chunk.text)
                rec = {
                    "narrator_id": narrator_id,
                    "instruction_index": idx,
                    "ok": True,
                    "path": str(chunk.wav_path),
                    "voice_instruction": instruction,
                    "text": chunk.text,
                    "word_count": words,
                    "raw_actual_wpm": _actual_wpm(words, info),
                    "elapsed_seconds": round(time.time() - t0, 3),
                    **info,
                }
                if args.target_wpm:
                    rec.update(_write_target_wpm_variant(
                        chunk.wav_path,
                        chunk.text,
                        args.target_wpm,
                    ))
                manifest.append(rec)
                print(
                    "OK "
                    f"{narrator_id}_{idx:02d}: "
                    f"{chunk.wav_path} "
                    f"duration={info.get('duration_seconds')}s "
                    f"wpm={rec.get('raw_actual_wpm')} "
                    f"sha={info.get('sha256', '')[:12]}"
                )
                if args.target_wpm and rec.get("target_wpm_path"):
                    print(
                        "   target "
                        f"{args.target_wpm:g}wpm: "
                        f"{rec['target_wpm_path']} "
                        f"duration={rec.get('target_duration_seconds')}s "
                        f"actual_wpm={rec.get('target_actual_wpm')}"
                    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


def _instruction_for_narrator(narrator_id: str) -> str:
    cfg = load_config()
    q_cfg = cfg.tts.get("qwen3") or {}
    instruction_map = q_cfg.get("voice_instruction_map") or {}
    mapped = (instruction_map.get(narrator_id) or "").strip()
    if mapped:
        return mapped
    return (q_cfg.get("voice_instruction") or "").strip()


def _wav_info(path: Path) -> dict:
    data = path.read_bytes()
    info = {
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


def _write_target_wpm_variant(
    raw_path: Path,
    text: str,
    target_wpm: float,
) -> dict:
    words = _word_count(text)
    target_wpm = max(1.0, float(target_wpm))
    target_seconds = (words / target_wpm) * 60.0
    wpm_label = f"{target_wpm:g}".replace(".", "p")
    target_path = raw_path.with_name(f"{raw_path.stem}_{wpm_label}wpm.wav")
    try:
        time_stretch_audio(raw_path, target_path, target_seconds)
        info = _wav_info(target_path)
        return {
            "target_wpm": target_wpm,
            "target_wpm_path": str(target_path),
            "target_duration_requested_seconds": round(target_seconds, 3),
            "target_duration_seconds": info.get("duration_seconds"),
            "target_actual_wpm": _actual_wpm(words, info),
            "target_bytes": info.get("bytes"),
            "target_sha256": info.get("sha256"),
        }
    except Exception as e:
        return {
            "target_wpm": target_wpm,
            "target_wpm_error": repr(e),
        }


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _actual_wpm(words: int, wav_info: dict) -> float | None:
    duration = wav_info.get("duration_seconds")
    if not duration:
        return None
    return round((words / float(duration)) * 60.0, 1)


if __name__ == "__main__":
    raise SystemExit(main())
