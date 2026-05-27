"""Shorts pipeline (Batch D 2026-05-27).

Generates N vertical (9:16) Short clips per episode from the most-
dramatic 30-second windows in the script. Long-form alone is slow
growth; Shorts as a discovery funnel pull 10-50x the views and
funnel new subscribers in.

Flow:
  1. Ask the writer LLM to identify the 3 most-dramatic 30-second
     windows (prompt: shorts_select.txt).
  2. For each window, cut the corresponding [start, end] segment
     out of 05_video/final.mp4.
  3. Crop/scale to 1080x1920 (9:16 vertical).
  4. Run ASR (whisper.cpp) over the cut audio to get word-level
     subtitles.
  5. Burn the subtitles as hard captions onto the vertical frame
     (Q-D1 confirmed: hard subtitles, behaviour disable-able via
     cfg.shorts.burn_subtitles).
  6. Output: 05_video/shorts/short_N.mp4 (1080x1920, 30s, with
     subtitles).
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asr import Segment, transcribe
from .config import load_config
from .ffmpeg_builder import require_ffmpeg
from .llm import LLM

logger = logging.getLogger("hermes.shorts")


@dataclass
class ShortWindow:
    rank: int
    start_seconds: float
    end_seconds: float
    title_hint: str
    reasoning: str


def pick_windows(
    *,
    incident: dict[str, Any],
    script: str,
    beat_sheet: dict[str, Any],
    voice_timing: dict[str, Any] | None,
    n: int = 3,
    target_seconds: float = 30.0,
) -> list[ShortWindow]:
    """Ask the writer LLM for `n` 30-second windows that work as
    standalone clips. Returns rank-ordered."""
    cfg = load_config()
    llm = LLM(role="writer")
    template_path = cfg.prompts_dir / "shorts_select.txt"
    if not template_path.exists():
        logger.warning("shorts_select.txt missing; skipping")
        return []
    template = template_path.read_text()

    beats = beat_sheet.get("beats", [])
    # Build a compact beat table the LLM can pick from.
    beat_lines: list[str] = []
    for b in beats:
        bid = b.get("beat_id", "")
        text = (b.get("script_text") or "")[:200].replace("\n", " ")
        beat_lines.append(f"{bid}: {text}")
    beats_dump = "\n".join(beat_lines)

    # Beat-id → start seconds, when available.
    starts_by_id: dict[str, float] = {}
    if voice_timing:
        for b in voice_timing.get("beats", []):
            bid = b.get("beat_id", "")
            if bid:
                starts_by_id[bid] = float(b.get("start_seconds", 0.0))

    prompt = template.format(
        n=n,
        target_seconds=int(target_seconds),
        company_name=incident.get("company_name", ""),
        hero=incident.get("hero", ""),
        conflict=incident.get("conflict", ""),
        story_kind=incident.get("story_kind", ""),
        beats_dump=beats_dump,
    )

    try:
        result = llm.complete_json(prompt, temperature=0.6, max_tokens=2000)
    except Exception as e:
        logger.warning("shorts JSON parse failed: %s", e)
        return []

    raw = result.get("windows") or result.get("clips") or []
    windows: list[ShortWindow] = []
    for i, w in enumerate(raw[:n], start=1):
        start_bid = w.get("start_beat_id") or w.get("from_beat_id")
        # Prefer the LLM-provided start_seconds, fall back to looking
        # up the start_beat_id in voice_timing.
        start = w.get("start_seconds")
        if start is None and start_bid:
            start = starts_by_id.get(start_bid, 0.0)
        if start is None:
            continue
        start = float(start)
        end = start + target_seconds
        windows.append(ShortWindow(
            rank=i,
            start_seconds=start,
            end_seconds=end,
            title_hint=(w.get("title_hint") or "")[:60],
            reasoning=(w.get("reasoning") or "")[:240],
        ))
    return windows


def cut_short(
    *,
    src_mp4: Path,
    out_mp4: Path,
    start_seconds: float,
    duration_seconds: float,
    burn_subtitles: bool,
    subtitles: list[Segment] | None,
    callout_text: str | None = None,
) -> bool:
    """Cut a 1080x1920 vertical Short from `src_mp4` starting at
    `start_seconds`. When `burn_subtitles` and `subtitles` are both
    provided, the captions are burned in via ffmpeg's drawtext.

    Falls back to a no-subtitle cut if subtitles are missing or
    whisper.cpp isn't installed.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Crop to 9:16 vertical from a 16:9 source: take the center
    # square (1080x1080) then pad to 1080x1920 with the same content
    # blurred above and below for a polished mobile look.
    #   scale=1080:-1,crop=1080:1080,...
    # Approach: copy the source clip's center column at 9:16.
    #
    # Simpler: scale to 1920 height keeping AR, then center-crop to
    # 1080 width. This works when the source is 16:9 (1920x1080) and
    # produces 1080x1920 directly.
    vf_parts = [
        "scale=-2:1920:flags=lanczos",
        "crop=1080:1920",
    ]

    # Hard subtitle burn-in. drawtext per-segment via enable=between(t,..).
    if burn_subtitles and subtitles:
        # whisper segments are absolute to the SOURCE wav (which we
        # extracted starting at start_seconds). We want timing
        # relative to the cut output, so subtract start_seconds.
        font_size = 60
        for seg in subtitles:
            s = max(0.0, seg.start_seconds - start_seconds)
            e = max(0.0, seg.end_seconds - start_seconds)
            if e <= s or e > duration_seconds:
                continue
            text = (seg.text or "").strip().replace("'", "")
            text = text.replace(":", "")  # drawtext escape
            if not text:
                continue
            vf_parts.append(
                f"drawtext=text='{text[:80]}':fontsize={font_size}"
                f":fontcolor=white:borderw=5:bordercolor=black"
                f":x=(w-text_w)/2:y=h-200"
                f":enable='between(t,{s:.2f},{e:.2f})'"
            )

    vf = ",".join(vf_parts)

    cmd = [
        require_ffmpeg(), "-y",
        "-ss", f"{start_seconds:.3f}",
        "-i", str(src_mp4),
        "-t", f"{duration_seconds:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=600)
        return out_mp4.exists() and out_mp4.stat().st_size > 1000
    except subprocess.CalledProcessError as e:
        logger.warning("ffmpeg Shorts cut failed: %s — stderr=%s",
                       e.returncode, (e.stderr or b"")[:400])
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("Shorts ffmpeg invocation failed: %s", e)
        return False


def extract_audio_for_window(
    src_mp4: Path, out_wav: Path,
    *, start_seconds: float, duration_seconds: float,
) -> bool:
    """Pull the audio of `[start, start+duration]` from src_mp4 into
    a mono 16kHz WAV (whisper.cpp's preferred input)."""
    cmd = [
        require_ffmpeg(), "-y",
        "-ss", f"{start_seconds:.3f}",
        "-i", str(src_mp4),
        "-t", f"{duration_seconds:.3f}",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=120)
        return out_wav.exists() and out_wav.stat().st_size > 0
    except Exception as e:
        logger.warning("extract_audio_for_window failed: %s", e)
        return False


def write_manifest(
    windows: list[ShortWindow], out_paths: list[Path],
    manifest_path: Path,
) -> None:
    """Write a JSON sidecar describing the Shorts that were emitted."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for w, p in zip(windows, out_paths):
        entries.append({
            "rank": w.rank,
            "start_seconds": round(w.start_seconds, 3),
            "end_seconds": round(w.end_seconds, 3),
            "title_hint": w.title_hint,
            "reasoning": w.reasoning,
            "path": str(p) if p else None,
        })
    manifest_path.write_text(json.dumps({"shorts": entries}, indent=2))
