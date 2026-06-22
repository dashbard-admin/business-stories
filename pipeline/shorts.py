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
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

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
    variant_rank: int = 1,
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
    variation_instruction = (
        "Short 1: use the default whole-story teaser angle. Open with "
        "the strongest central contradiction, then sell the full episode."
        if variant_rank <= 1 else
        f"Short {variant_rank}: write a brand-new alternate teaser, not a "
        "paraphrase of Short 1. Choose a different opening hook, focus on "
        "a different tension from the story, and vary the title_hint and "
        "curiosity hooks while still ending with a reason to watch the "
        "full episode."
    )
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
        variation_instruction=variation_instruction,
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
    enforce_wpm: bool = False,
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
    if enforce_wpm:
        desired = (words / max(1.0, target_wpm)) * 60.0
        desired = max(10.0, desired)
        try:
            time_stretch_audio(raw_wav, final_wav, desired)
        except Exception as e:
            logger.warning(
                "Shorts teaser time-stretch failed (%s); using raw TTS", e
            )
            final_wav.write_bytes(raw_wav.read_bytes())
    else:
        final_wav.write_bytes(raw_wav.read_bytes())
    duration = get_duration_seconds(final_wav)
    logger.info(
        "shorts teaser TTS: %d words, target %.0f wpm, speed %.2f, "
        "enforce_wpm=%s -> %.1fs",
        words, target_wpm, tts_speed, enforce_wpm, duration,
    )
    return final_wav, duration


def build_teaser_subtitles(
    script: str,
    duration_seconds: float,
    *,
    max_words: int = 6,
    max_chars: int = 44,
) -> list[Segment]:
    units = _split_caption_units(
        script, max_words=max_words, max_chars=max_chars
    )
    total_words = sum(len(s.split()) for s in units) or 1
    cursor = 0.0
    out: list[Segment] = []
    for unit in units:
        words = max(1, len(unit.split()))
        seg_dur = float(duration_seconds) * (words / total_words)
        start = cursor
        end = min(float(duration_seconds), cursor + seg_dur)
        cursor = end
        out.append(Segment(start_seconds=start, end_seconds=end, text=unit))
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


def shift_subtitles(
    segments: list[Segment],
    offset_seconds: float,
) -> list[Segment]:
    if offset_seconds <= 0:
        return list(segments)
    return [
        Segment(
            start_seconds=seg.start_seconds + offset_seconds,
            end_seconds=seg.end_seconds + offset_seconds,
            text=seg.text,
        )
        for seg in segments
    ]


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
    transition_seconds: float = 0.22,
    motion_strength: float = 0.10,
    title_card_path: Path | None = None,
    title_card_seconds: float = 0.0,
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
    transition = max(
        0.0,
        min(float(transition_seconds), max(1.0, seconds_per_image) / 2),
    )
    base_dur = (
        (float(duration_seconds) + transition * max(0, len(images) - 1))
        / len(images)
    )
    for i, img in enumerate(images, start=1):
        clip = work_dir / f"img_{i:02d}.mp4"
        frames = max(1, int(round(base_dur * 30)))
        vf = _shorts_motion_filter(
            index=i,
            duration_seconds=base_dur,
            motion_strength=motion_strength,
        )
        cmd = [
            require_ffmpeg(), "-y",
            "-loop", "1",
            "-i", str(img),
            "-vf", vf,
            "-r", "30",
            "-frames:v", str(frames),
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

    video_only = work_dir / "video_only.mp4"
    if not _join_short_clips(
        clip_paths=clip_paths,
        out_path=video_only,
        clip_seconds=base_dur,
        transition_seconds=transition,
    ):
        return False

    audio_offset = (
        max(0.0, float(title_card_seconds))
        if title_card_path and title_card_path.exists() else 0.0
    )
    video_for_mux = video_only
    if audio_offset > 0:
        title_clip = work_dir / "short_title_card.mp4"
        if _render_still_clip(
            image_path=title_card_path,
            out_path=title_clip,
            duration_seconds=audio_offset,
        ):
            joined = work_dir / "video_with_title.mp4"
            if _concat_hard_cut([title_clip, video_only], joined):
                video_for_mux = joined
            else:
                logger.warning("Shorts title prepend failed; continuing without it")
                audio_offset = 0.0
        else:
            logger.warning("Shorts title clip render failed; continuing without it")
            audio_offset = 0.0

    temp_mp4 = work_dir / "with_audio.mp4"
    total_duration = float(duration_seconds) + audio_offset
    cmd = [
        require_ffmpeg(), "-y",
        "-i", str(video_for_mux),
    ]
    if audio_offset > 0:
        cmd.extend(["-itsoffset", f"{audio_offset:.3f}"])
    cmd.extend([
        "-i", str(audio_path),
        "-t", f"{total_duration:.3f}",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        str(temp_mp4),
    ])
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=300)
    except Exception as e:
        logger.warning("Shorts teaser mux failed: %s", e)
        return False

    overlays = (
        _caption_overlays(subtitles, total_duration, work_dir=work_dir)
        if burn_subtitles and subtitles else []
    )
    if overlays:
        inputs: list[str] = ["-i", str(temp_mp4)]
        for overlay_path, _start, _end in overlays:
            inputs.extend(["-loop", "1", "-i", str(overlay_path)])
        filter_complex = _overlay_filter(overlays)
        cmd = [
            require_ffmpeg(), "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[v{len(overlays)}]",
            "-map", "0:a:0?",
            "-t", f"{total_duration:.3f}",
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
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        logger.warning(
            "Shorts teaser final encode failed: %s — stderr=%s",
            e.returncode,
            stderr[-2000:],
        )
        return False
    except Exception as e:
        logger.warning("Shorts teaser final encode failed: %s", e)
        return False


def render_short_title_card(
    *,
    image_path: Path,
    logo_path: Path | None,
    out_path: Path,
    incident: dict[str, Any],
    rank: int,
    enabled: bool = True,
) -> Path | None:
    if not enabled or not image_path.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as raw:
        img = _cover_vertical(raw.convert("RGB")).convert("RGBA")
    shade = Image.new("RGBA", img.size, (0, 0, 0, 118))
    img.alpha_composite(shade)

    company = str(incident.get("company_name") or "Company").strip()
    category = _story_outcome_category(incident)
    template = _short_title_template(
        company=company, category=category, rank=rank
    )
    logo = _load_short_logo(logo_path) if logo_path else None
    _draw_short_title_lines(img, template, logo=logo, company=company)
    img.convert("RGB").save(out_path, "PNG")
    return out_path


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
    teasers: list[ShortTeaser | None],
    out_paths: list[Path | None],
    srt_paths: list[Path | None],
    image_sets: list[list[Path]],
    title_card_paths: list[Path | None] | None = None,
    manifest_path: Path,
    duration_seconds: float,
    durations: list[float] | None = None,
    target_wpm: float,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx, path in enumerate(out_paths, start=1):
        teaser = teasers[idx - 1] if idx - 1 < len(teasers) else None
        item_duration = (
            float(durations[idx - 1])
            if durations and idx - 1 < len(durations) else duration_seconds
        )
        entries.append({
            "rank": idx,
            "mode": "teaser",
            "start_seconds": 0.0,
            "end_seconds": round(item_duration, 3),
            "title_hint": teaser.title_hint if teaser else "",
            "reasoning": teaser.hook_notes if teaser else "",
            "teaser_script": teaser.script if teaser else "",
            "teaser_script_path": f"teaser_script_{idx:02d}.txt" if teaser else None,
            "path": str(path) if path else None,
            "caption_path": str(srt_paths[idx - 1]) if idx - 1 < len(srt_paths) and srt_paths[idx - 1] else None,
            "title_card_path": (
                str(title_card_paths[idx - 1])
                if title_card_paths
                and idx - 1 < len(title_card_paths)
                and title_card_paths[idx - 1]
                else None
            ),
            "target_wpm": target_wpm,
            "script_word_count": teaser.word_count if teaser else 0,
            "images": [p.name for p in image_sets[idx - 1]] if idx - 1 < len(image_sets) else [],
        })
    first_teaser = next((t for t in teasers if t), None)
    manifest_path.write_text(json.dumps({
        "mode": "teaser",
        "teaser_script": first_teaser.script if first_teaser else "",
        "teaser_scripts": [
            {
                "rank": idx,
                "path": f"teaser_script_{idx:02d}.txt",
                "script": teaser.script,
                "title_hint": teaser.title_hint,
                "hook_notes": teaser.hook_notes,
                "word_count": teaser.word_count,
            }
            for idx, teaser in enumerate(teasers, start=1)
            if teaser
        ],
        "shorts": entries,
    }, indent=2))


def _clean_teaser_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("\"' ")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_caption_units(
    text: str, *, max_words: int, max_chars: int
) -> list[str]:
    units: list[str] = []
    for sentence in _split_sentences(text):
        words = sentence.split()
        buf: list[str] = []
        for word in words:
            candidate = " ".join(buf + [word])
            if (
                buf
                and (
                    len(buf) >= max(1, max_words)
                    or len(candidate) > max(12, max_chars)
                )
            ):
                units.append(" ".join(buf))
                buf = [word]
            else:
                buf.append(word)
        if buf:
            units.append(" ".join(buf))
    return units or [text.strip()]


def _shorts_motion_filter(
    *, index: int, duration_seconds: float, motion_strength: float
) -> str:
    frames = max(1, int(round(duration_seconds * 30)))
    strength = max(0.0, min(0.35, float(motion_strength)))
    progress = f"(on/{max(1, frames - 1)})"
    ease = f"{progress}*{progress}*(3-2*{progress})"
    x_center = "(iw-iw/zoom)/2"
    y_center = "(ih-ih/zoom)/2"
    if index % 3 == 1:
        zoom = f"1+{strength:.4f}*{ease}"
        x_expr = x_center
        y_expr = y_center
    elif index % 3 == 2:
        zoom = f"{1.0 + strength:.4f}"
        # Crop origin moves right, so the visible image pans left.
        x_expr = f"(iw-iw/zoom)*{ease}"
        y_expr = y_center
    else:
        zoom = f"{1.0 + strength:.4f}"
        # Crop origin moves left, so the visible image pans right.
        x_expr = f"(iw-iw/zoom)*(1-{ease})"
        y_expr = y_center
    return (
        "scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=2160:3840,"
        f"zoompan=z='{zoom}':x='{x_expr}':y='{y_expr}'"
        f":d={frames}:s=1080x1920:fps=30,"
        "setsar=1"
    )


def _join_short_clips(
    *,
    clip_paths: list[Path],
    out_path: Path,
    clip_seconds: float,
    transition_seconds: float,
) -> bool:
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], out_path)
        return True
    transition = max(0.0, min(float(transition_seconds), clip_seconds / 2))
    if transition > 0:
        cmd: list[str] = [require_ffmpeg(), "-y"]
        for p in clip_paths:
            cmd.extend(["-i", str(p)])
        parts: list[str] = []
        prev = "[0:v]"
        for idx in range(1, len(clip_paths)):
            out_label = f"[v{idx}]"
            offset = max(0.0, idx * (clip_seconds - transition))
            parts.append(
                f"{prev}[{idx}:v]xfade=transition=slideleft"
                f":duration={transition:.3f}:offset={offset:.3f}"
                f"{out_label}"
            )
            prev = out_label
        cmd.extend([
            "-filter_complex", ";".join(parts),
            "-map", f"[v{len(clip_paths) - 1}]",
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ])
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           stdin=subprocess.DEVNULL, timeout=300)
            return out_path.exists() and out_path.stat().st_size > 1000
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")
            logger.warning(
                "Shorts xfade transition failed; falling back to hard cuts: %s",
                stderr[-800:],
            )
        except Exception as e:
            logger.warning(
                "Shorts xfade transition failed; falling back to hard cuts: %s",
                e,
            )

    concat_file = out_path.parent / f"{out_path.stem}.concat.txt"
    concat_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in clip_paths)
    )
    cmd = [
        require_ffmpeg(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-an",
        "-c", "copy",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=180)
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception as e:
        logger.warning("Shorts hard-cut concat failed: %s", e)
        return False


def _render_still_clip(
    *,
    image_path: Path,
    out_path: Path,
    duration_seconds: float,
) -> bool:
    frames = max(1, int(round(float(duration_seconds) * 30)))
    cmd = [
        require_ffmpeg(), "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf", "scale=1080:1920:flags=lanczos,setsar=1",
        "-r", "30",
        "-frames:v", str(frames),
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=120)
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception as e:
        logger.warning("Shorts still clip failed for %s: %s", image_path.name, e)
        return False


def _concat_hard_cut(clip_paths: list[Path], out_path: Path) -> bool:
    concat_file = out_path.parent / f"{out_path.stem}.concat.txt"
    concat_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in clip_paths)
    )
    cmd = [
        require_ffmpeg(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-an",
        "-c", "copy",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=120)
        return out_path.exists() and out_path.stat().st_size > 1000
    except Exception as e:
        logger.warning("Shorts title concat failed: %s", e)
        return False


WINNING_SHORT_TITLES = [
    ["How", "[LOGO]", "won it"],
    ["The", "[LOGO]", "path to victory"],
    ["How", "[LOGO]", "pulled it off"],
    ["The", "[LOGO]", "winning method"],
    ["The", "[LOGO]", "road to win"],
    ["How", "[LOGO]", "came out", "on top"],
    ["[LOGO]", "secret to", "success"],
    ["How", "[LOGO]", "made history"],
    ["The", "[LOGO]", "glorious ascent"],
    ["The", "[LOGO]", "victory blueprint"],
    ["The", "[LOGO]", "glide to glory"],
    ["How", "[LOGO]", "took the crown"],
    ["[LOGO]", "road to triumph"],
    ["How", "[LOGO]", "secured the win"],
    ["The", "[LOGO]", "masterstroke"],
    ["The", "[LOGO]", "path to dominance"],
    ["How", "[LOGO]", "emerged victorious"],
    ["The", "[LOGO]", "formula for success"],
    ["How", "[LOGO]", "defied the odds"],
    ["The", "[LOGO]", "route to victory"],
    ["The", "[LOGO]", "tactical win plan"],
    ["How", "[LOGO]", "made the win possible"],
    ["The", "[LOGO]", "step-by-step to success"]
]

LOSING_SHORT_TITLES = [
    ["How", "[LOGO]", "fell short"],
    ["How", "[LOGO]", "lost its way"],
    ["[LOGO]", "road to", "defeat"],
    ["How", "[LOGO]", "came up", "short"],
    ["[LOGO]", "downfall"],
    ["[LOGO]", "path to", "defeat"],
    ["How", "[LOGO]", "let it go"],
    ["How", "[LOGO]", "missed the mark"],
    ["[LOGO]", "route to defeat"],
    ["How", "[LOGO]", "slipped up"],
    ["[LOGO]", "path to failure"],
    ["How", "[LOGO]", "lost control"],
    ["How", "[LOGO]", "failed to deliver"],
    ["[LOGO]", "fall from grace"],
    ["How", "[LOGO]", "came undone"],
    ["How", "[LOGO]", "lost the edge"],
    ["[LOGO]", "unraveling"],
    ["[LOGO]", "route to failure"],
    ["How", "[LOGO]", "let victory slip"],
    ["The", "[LOGO]", "losing formula"],
    ["How", "[LOGO]", "stumbled late"],
    ["[LOGO]", "slide into defeat"],
]


def _story_outcome_category(incident: dict[str, Any]) -> str:
    story_kind = str(incident.get("story_kind") or "").lower()
    text = " ".join([
        story_kind,
        str(incident.get("one_line_pitch") or ""),
        str(incident.get("conflict") or ""),
    ]).lower()
    losing_markers = (
        "fall", "fell", "failure", "failed", "lost", "defeat", "collapse",
        "bankrupt", "scandal", "postmortem", "decline", "shutdown",
        "missed", "cautionary",
    )
    if any(marker in text for marker in losing_markers):
        return "losing"
    return "winning"


def _short_title_template(
    *, company: str, category: str, rank: int
) -> list[str]:
    choices = LOSING_SHORT_TITLES if category == "losing" else WINNING_SHORT_TITLES
    import hashlib
    seed = hashlib.md5(f"{company}|{category}|{rank}".encode()).hexdigest()
    return choices[int(seed[:8], 16) % len(choices)]


def _cover_vertical(img: Image.Image) -> Image.Image:
    target_w, target_h = 1080, 1920
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_size = (max(target_w, int(src_w * scale)), max(target_h, int(src_h * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    x0 = (img.width - target_w) // 2
    y0 = (img.height - target_h) // 2
    return img.crop((x0, y0, x0 + target_w, y0 + target_h))


def _load_short_logo(path: Path | None) -> Image.Image | None:
    if path is None or not path.exists():
        return None
    try:
        with Image.open(path) as raw:
            logo = raw.convert("RGBA")
    except Exception as e:
        logger.warning("Shorts logo load failed (%s): %s", path, e)
        return None
    bg = Image.new("RGBA", logo.size, logo.getpixel((0, 0)))
    bbox = ImageChops.difference(logo, bg).getbbox()
    if bbox:
        logo = logo.crop(bbox)
    has_alpha = _has_transparent_pixels(logo)
    px = logo.load()
    if px is not None and not has_alpha:
        for y in range(logo.height):
            for x in range(logo.width):
                r, g, b, a = px[x, y]
                if a and r > 245 and g > 245 and b > 245:
                    px[x, y] = (r, g, b, 0)
    if logo.getchannel("A").getbbox() is None:
        return None
    edge_pad = int(1080 * 0.06)
    max_w = 1080 - (edge_pad * 2)
    max_h = int(1920 * 0.32)
    scale = min(max_w / logo.width, max_h / logo.height)
    return logo.resize(
        (max(1, int(logo.width * scale)), max(1, int(logo.height * scale))),
        Image.LANCZOS,
    )


def _has_transparent_pixels(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    alpha = img.getchannel("A")
    return alpha.getextrema()[0] < 255


def _draw_short_title_lines(
    img: Image.Image,
    lines: list[str],
    *,
    logo: Image.Image | None,
    company: str,
) -> None:
    draw = ImageDraw.Draw(img)
    stroke_w = 8
    line_gap = 24
    font = _fit_short_title_font(
        draw=draw,
        lines=lines,
        logo=logo,
        company=company,
        stroke_w=stroke_w,
        line_gap=line_gap,
    )
    metrics: list[tuple[str, int, int]] = []
    total_h = 0
    for line in lines:
        if line == "[LOGO]" and logo is not None:
            w, h = logo.size
        else:
            text = company if line == "[LOGO]" else line
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        metrics.append((line, w, h))
        total_h += h
    total_h += line_gap * max(0, len(lines) - 1)
    y = (1920 - total_h) // 2
    for line, w, h in metrics:
        if line == "[LOGO]" and logo is not None:
            x = (1080 - logo.width) // 2
            shadow = Image.new("RGBA", logo.size, (0, 0, 0, 0))
            shadow.putalpha(
                logo.getchannel("A").filter(ImageFilter.GaussianBlur(10))
            )
            img.alpha_composite(shadow, (x + 5, y + 7))
            img.alpha_composite(logo, (x, y))
        else:
            text = company if line == "[LOGO]" else line
            x = (1080 - w) // 2
            draw.text(
                (x, y),
                text,
                font=font,
                fill=(255, 230, 0, 255),
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 255),
            )
        y += h + line_gap


def _short_title_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit_short_title_font(
    *,
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    logo: Image.Image | None,
    company: str,
    stroke_w: int,
    line_gap: int,
) -> ImageFont.ImageFont:
    max_w = 980
    max_h = 1500
    for size in range(264, 95, -6):
        font = _short_title_font(size)
        total_h = 0
        ok = True
        for line in lines:
            if line == "[LOGO]" and logo is not None:
                w, h = logo.size
            else:
                text = company if line == "[LOGO]" else line
                bbox = draw.textbbox((0, 0), text, font=font,
                                     stroke_width=stroke_w)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w > max_w:
                ok = False
                break
            total_h += h
        total_h += line_gap * max(0, len(lines) - 1)
        if ok and total_h <= max_h:
            return font
    return _short_title_font(96)


def _caption_overlays(
    subtitles: list[Segment],
    duration_seconds: float,
    *,
    work_dir: Path,
) -> list[tuple[Path, float, float]]:
    overlays: list[tuple[Path, float, float]] = []
    font = _caption_font(62)
    for idx, seg in enumerate(subtitles, start=1):
        start = max(0.0, float(seg.start_seconds))
        end = min(float(duration_seconds), float(seg.end_seconds))
        if end <= start:
            continue
        text = seg.text.strip()
        if not text:
            continue
        path = work_dir / f"caption_{idx:02d}.png"
        _write_caption_overlay(text=text, font=font, out_path=path)
        overlays.append((path, start, end))
    return overlays


def _write_caption_overlay(
    *, text: str, font: ImageFont.ImageFont, out_path: Path
) -> None:
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    lines = _wrap_caption(text, font=font, max_width=900)
    line_heights = [
        draw.textbbox((0, 0), line, font=font, stroke_width=4)[3]
        for line in lines
    ]
    line_gap = 10
    text_h = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    box_pad_x = 42
    box_pad_y = 30
    box_w = 980
    box_h = text_h + box_pad_y * 2
    box_x = (1080 - box_w) // 2
    box_y = 1920 - box_h - 190
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=28,
        fill=(0, 0, 0, 178),
    )
    y = box_y + box_pad_y
    for line, line_h in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=4)
        x = (1080 - (bbox[2] - bbox[0])) // 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=4,
            stroke_fill=(0, 0, 0, 255),
        )
        y += line_h + line_gap
    img.save(out_path)


def _overlay_filter(overlays: list[tuple[Path, float, float]]) -> str:
    parts = ["[0:v]setsar=1[v0]"]
    for idx, (_path, start, end) in enumerate(overlays, start=1):
        parts.append(
            f"[v{idx - 1}][{idx}:v]overlay=0:0"
            f":enable='between(t,{start:.2f},{end:.2f})'"
            f"[v{idx}]"
        )
    return ";".join(parts)


def _wrap_caption(
    text: str, *, font: ImageFont.ImageFont, max_width: int
) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=4)
        if current and (bbox[2] - bbox[0]) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text[:60]]


def _caption_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _ts_srt(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
