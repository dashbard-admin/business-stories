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
                rec = {
                    "narrator_id": narrator_id,
                    "instruction_index": idx,
                    "ok": True,
                    "path": str(chunk.wav_path),
                    "voice_instruction": instruction,
                    "text": chunk.text,
                    "elapsed_seconds": round(time.time() - t0, 3),
                    **info,
                }
                manifest.append(rec)
                print(
                    "OK "
                    f"{narrator_id}_{idx:02d}: "
                    f"{chunk.wav_path} "
                    f"duration={info.get('duration_seconds')}s "
                    f"sha={info.get('sha256', '')[:12]}"
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


if __name__ == "__main__":
    raise SystemExit(main())
