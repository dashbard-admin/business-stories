"""Shorts pipeline (Batch D 2026-05-27).

Generates N vertical (9:16) Short clips per episode. The current
default is a teaser-first path: write a fresh 30-second viral script,
render fast standalone TTS, and cut quickly across reused beat images.
The older window-cutter helpers remain below for fallback/comparison.

Flow:
  1. Ask the writer LLM for one compressed viral teaser script
     (prompt: shorts_teaser_script.txt).
  2. Render one fast standalone TTS track, then time-stretch it to
     the configured WPM target.
  3. Generate English SRT cues from the teaser script.
  4. For each Short, reuse beat images as rapid 3-5 second vertical
     cuts and mux the teaser voice only. No music, no SFX.
  5. Optionally hard-burn the English captions.
  6. Output: 05_video/shorts/short_NN.mp4 + short_NN.srt +
     manifest.json.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asr import Segment, transcribe
from .config import load_config
from .ffmpeg_builder import get_duration_seconds, require_ffmpeg, time_stretch_audio
from .llm import LLM
from .tts import KOKORO_SAMPLE_RATE, make_tts

logger = logging.getLogger("hermes.shorts")


@dataclass
class ShortWindow:
    rank: int
    start_seconds: float
    end_seconds: float
    title_hint: str
    reasoning: str


@dataclass
class ShortTeaser:
    script: str
    title_hint: str
    hook_notes: str
    word_count: int


def generate_teaser_script(
    *,
    incident: dict[str, Any],
    script: str,
    beat_sheet: dict[str, Any],
    target_seconds: float = 30.0,
    target_wpm: float = 230.0,
) -> ShortTeaser | None:
    """Generate a fresh, compressed Shorts script from the full episode."""
    cfg = load_config()
    template_path = cfg.prompts_dir / "shorts_teaser_script.txt"
    if not template_path.exists():
        logger.warning("shorts_teaser_script.txt missing; skipping teaser")
        return None
    beats = beat_sheet.get("beats", [])
    beat_lines = []
    for b in beats:
        bid = b.get("beat_id", "")
        text = (b.get("script_text") or "")[:260].replace("\n", " ")
        if bid and text:
            beat_lines.append(f"{bid}: {text}")
    prompt = template_path.read_text().format(
        company_name=incident.get("company_name", ""),
        hero=incident.get("hero", ""),
        conflict=incident.get("conflict", ""),
        story_kind=incident.get("story_kind", ""),
        target_seconds=int(target_seconds),
        target_wpm=int(target_wpm),
        target_words_min=int(target_seconds * target_wpm / 60) - 12,
        target_words_max=int(target_seconds * target_wpm / 60) + 8,
        script=script[:24000],
        beats_dump="\n".join(beat_lines[:120]),
    )
    try:
        result = LLM(role="writer").complete_json(
            prompt, temperature=0.76, max_tokens=1800
        )
    except Exception as e:
        logger.warning("shorts teaser generation failed: %s", e)
        return None
    teaser = _clean_teaser_text(result.get("teaser_script") or "")
    if not teaser:
        return None
    return ShortTeaser(
        script=teaser,
        title_hint=(result.get("title_hint") or incident.get("company_name") or "Short")[:80],
        hook_notes=(result.get("hook_notes") or "")[:500],
        word_count=len(teaser.split()),
    )


def render_teaser_audio(
    *,
    teaser: ShortTeaser,
    narrator_id: str,
    out_dir: Path,
    target_seconds: float,
    target_wpm: float,
    tts_speed: float,
) -> tuple[Path, float]:
    """Render one fast Shorts voice track with no music or SFX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "teaser_chunks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tts = make_tts(narrator_id=narrator_id)
    if hasattr(tts, "speed"):
        tts.speed = float(tts_speed)
    chunks = tts.synthesize_script(
        teaser.script, raw_dir, max_words_per_chunk=160
    )
    if not chunks:
        raise RuntimeError("Shorts teaser TTS produced no chunks")

    concat_file = out_dir / "teaser_audio.concat.txt"
    concat_file.write_text(
        "".join(f"file '{c.wav_path.as_posix()}'\n" for c in chunks)
    )
    raw_wav = out_dir / "teaser_voice_raw.wav"
    final_wav = out_dir / "teaser_voice.wav"
    cmd = [
        require_ffmpeg(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-ar", str(KOKORO_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(raw_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)

    words = max(1, teaser.word_count)
    desired = (words / max(1.0, target_wpm)) * 60.0
    desired = min(float(target_seconds), max(10.0, desired))
    try:
        time_stretch_audio(raw_wav, final_wav, desired)
    except Exception as e:
        logger.warning("Shorts teaser time-stretch failed (%s); using raw TTS", e)
        final_wav.write_bytes(raw_wav.read_bytes())
    duration = get_duration_seconds(final_wav)
    logger.info(
        "shorts teaser TTS: %d words, target %.0f wpm -> %.1fs",
        words, target_wpm, duration,
    )
    return final_wav, duration


def build_teaser_subtitles(script: str, duration_seconds: float) -> list[Segment]:
    sentences = _split_sentences(script)
    total_words = sum(len(s.split()) for s in sentences) or 1
    cursor = 0.0
    out: list[Segment] = []
    for sentence in sentences:
        words = max(1, len(sentence.split()))
        seg_dur = float(duration_seconds) * (words / total_words)
        start = cursor
        end = min(float(duration_seconds), cursor + seg_dur)
        cursor = end
        out.append(Segment(start_seconds=start, end_seconds=end, text=sentence))
    return out


def write_srt(segments: list[Segment], path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.extend([
            str(i),
            f"{_ts_srt(seg.start_seconds)} --> {_ts_srt(seg.end_seconds)}",
            seg.text.strip(),
            "",
        ])
    path.write_text("\n".join(lines))


def collect_story_images(
    *,
    ws: Path,
    beat_sheet: dict[str, Any],
    count: int,
    offset: int = 0,
) -> list[Path]:
    flux_dir = ws / "03_assets" / "flux"
    beats = beat_sheet.get("beats") or []
    ordered: list[Path] = []
    for b in beats:
        bid = b.get("beat_id", "")
        if not bid:
            continue
        p = flux_dir / f"{bid}.png"
        if p.exists():
            ordered.append(p)
    if not ordered:
        ordered = sorted(flux_dir.glob("BEAT_*.png"))
    if not ordered:
        title = ws / "05_video" / "title_card.png"
        return [title] if title.exists() else []
    if len(ordered) <= count:
        return ordered
    step = max(1, len(ordered) // count)
    picked = [ordered[(offset + i * step) % len(ordered)] for i in range(count)]
    # Preserve order while removing duplicates; top up if the modulo repeated.
    unique: list[Path] = []
    for p in picked + ordered:
        if p not in unique:
            unique.append(p)
        if len(unique) >= count:
            break
    return unique


def build_teaser_short(
    *,
    image_paths: list[Path],
    audio_path: Path,
    out_mp4: Path,
    duration_seconds: float,
    subtitles: list[Segment] | None,
    burn_subtitles: bool,
    seconds_per_image: float,
) -> bool:
    """Create a vertical Short from still images plus standalone teaser audio."""
    if not image_paths:
        logger.warning("Shorts teaser has no images")
        return False
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_mp4.parent / f".{out_mp4.stem}_clips"
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    needed = max(1, int(round(float(duration_seconds) / max(1.0, seconds_per_image) + 0.49)))
    images = [image_paths[i % len(image_paths)] for i in range(needed)]
    base_dur = float(duration_seconds) / len(images)
    for i, img in enumerate(images, start=1):
        clip = work_dir / f"img_{i:02d}.mp4"
        vf = (
            "scale=-2:1920:flags=lanczos,"
            "crop=1080:1920,"
            "setsar=1"
        )
        cmd = [
            require_ffmpeg(), "-y",
            "-loop", "1",
            "-t", f"{base_dur:.3f}",
            "-i", str(img),
            "-vf", vf,
            "-r", "30",
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(clip),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           stdin=subprocess.DEVNULL, timeout=240)
            clip_paths.append(clip)
        except Exception as e:
            logger.warning("Shorts image clip failed for %s: %s", img.name, e)
    if not clip_paths:
        return False

    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in clip_paths)
    )
    temp_mp4 = work_dir / "with_audio.mp4"
    cmd = [
        require_ffmpeg(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-i", str(audio_path),
        "-t", f"{duration_seconds:.3f}",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(temp_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=300)
    except Exception as e:
        logger.warning("Shorts teaser mux failed: %s", e)
        return False

    if burn_subtitles and subtitles:
        vf = _drawtext_filter(subtitles, duration_seconds)
        cmd = [
            require_ffmpeg(), "-y",
            "-i", str(temp_mp4),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(out_mp4),
        ]
    else:
        cmd = [
            require_ffmpeg(), "-y",
            "-i", str(temp_mp4),
            "-c", "copy",
            str(out_mp4),
        ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=300)
        return out_mp4.exists() and out_mp4.stat().st_size > 1000
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Shorts teaser final encode failed: %s — stderr=%s",
            e.returncode,
            (e.stderr or b"")[:1000],
        )
        return False
    except Exception as e:
        logger.warning("Shorts teaser final encode failed: %s", e)
        return False


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
    exclude_callouts = bool(cfg.packaging.get("shorts_exclude_callout_beats", True))
    callout_beat_ids = {
        b.get("beat_id", "") for b in beats
        if exclude_callouts and (b.get("callouts") or [])
    }
    # Build a compact beat table the LLM can pick from.
    beat_lines: list[str] = []
    for b in beats:
        bid = b.get("beat_id", "")
        if bid in callout_beat_ids:
            continue
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
    callout_intervals: list[tuple[float, float, str]] = []
    if exclude_callouts and voice_timing:
        for b in voice_timing.get("beats", []):
            bid = b.get("beat_id", "")
            if bid in callout_beat_ids:
                callout_intervals.append((
                    float(b.get("start_seconds", 0.0)),
                    float(b.get("end_seconds", 0.0)),
                    bid,
                ))

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
    for w in raw:
        if len(windows) >= n:
            break
        start_bid = w.get("start_beat_id") or w.get("from_beat_id")
        if start_bid in callout_beat_ids:
            logger.info("shorts: skipped %s because it has callouts", start_bid)
            continue
        # Prefer the LLM-provided start_seconds, fall back to looking
        # up the start_beat_id in voice_timing.
        start = w.get("start_seconds")
        if start is None and start_bid:
            start = starts_by_id.get(start_bid, 0.0)
        if start is None:
            continue
        start = float(start)
        end = start + target_seconds
        if exclude_callouts and any(
            start < c_end and end > c_start
            for c_start, c_end, _bid in callout_intervals
        ):
            logger.info(
                "shorts: skipped window %.1f-%.1f because it overlaps "
                "a callout beat",
                start, end,
            )
            continue
        windows.append(ShortWindow(
            rank=len(windows) + 1,
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
        "-map", "0:v:0",
        "-map", "0:a:0?",
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


def write_teaser_manifest(
    *,
    teaser: ShortTeaser,
    out_paths: list[Path | None],
    srt_paths: list[Path | None],
    image_sets: list[list[Path]],
    manifest_path: Path,
    duration_seconds: float,
    target_wpm: float,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx, path in enumerate(out_paths, start=1):
        entries.append({
            "rank": idx,
            "mode": "teaser",
            "start_seconds": 0.0,
            "end_seconds": round(duration_seconds, 3),
            "title_hint": teaser.title_hint,
            "reasoning": teaser.hook_notes,
            "path": str(path) if path else None,
            "caption_path": str(srt_paths[idx - 1]) if idx - 1 < len(srt_paths) and srt_paths[idx - 1] else None,
            "target_wpm": target_wpm,
            "script_word_count": teaser.word_count,
            "images": [p.name for p in image_sets[idx - 1]] if idx - 1 < len(image_sets) else [],
        })
    manifest_path.write_text(json.dumps({
        "mode": "teaser",
        "teaser_script": teaser.script,
        "shorts": entries,
    }, indent=2))


def _clean_teaser_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"' ")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _drawtext_filter(subtitles: list[Segment], duration_seconds: float) -> str:
    parts = ["scale=1080:1920,setsar=1"]
    for seg in subtitles:
        start = max(0.0, float(seg.start_seconds))
        end = min(float(duration_seconds), float(seg.end_seconds))
        if end <= start:
            continue
        text = _drawtext_escape(seg.text.strip()[:96])
        if not text:
            continue
        parts.append(
            f"drawtext=text='{text}':fontsize=64"
            f":fontcolor=white:borderw=6:bordercolor=black"
            f":x=(w-text_w)/2:y=h-260"
            f":enable='between(t,{start:.2f},{end:.2f})'"
        )
    return ",".join(parts)


def _drawtext_escape(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace(",", "\\,")
    text = text.replace("%", "\\%")
    text = text.replace("\n", " ")
    return text


def _ts_srt(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
