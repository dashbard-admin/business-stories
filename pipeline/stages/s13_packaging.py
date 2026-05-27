"""S13 — Packaging (Batch D 2026-05-27).

Post-S12 stage that runs three discovery/CTR features:

  Phase 1: Title variants
    Calls pipeline.titles.generate_variants() → 06_metadata/titles.json
    with up to 10 candidate titles tagged by style hypothesis +
    predicted CTR band.

  Phase 2: Thumbnail variants
    Calls pipeline.thumbnails.generate_variants() → 5 thumbnail
    JPGs at 05_video/thumbnails/thumb_<layout>.jpg. Layouts:
    founder_closeup, split_frame, big_number, shocked_face, noir.

  Phase 3: Shorts cutter
    Calls pipeline.shorts.pick_windows() to identify 3 dramatic
    30s windows, then pipeline.shorts.cut_short() to cut each as
    1080x1920 vertical with hard-burned subtitles (whisper.cpp
    via pipeline.asr). Outputs at 05_video/shorts/short_N.mp4 +
    05_video/shorts/manifest.json.

S13 is in STAGE_DISPATCH so it auto-runs after S12. In preview
mode (Act 0 + Act 5 only renders), S13 still runs but title and
thumbnail variants are based on the partial script — useful for
tone-check on the packaging itself.
"""

from __future__ import annotations

import json
import logging

from ..config import load_config
from ..shorts import (
    ShortWindow, cut_short, extract_audio_for_window,
    pick_windows, write_manifest,
)
from ..state import find_episode_workspace
from ..thumbnails import generate_variants as generate_thumbnail_variants
from ..titles import generate_variants as generate_title_variants
from ..titles import write_variants as write_title_variants
from ..asr import transcribe

logger = logging.getLogger("hermes.stage.s13")


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    incident = episode.get("incident") or {}
    pack_cfg = cfg.packaging

    # ---------- Phase 1: title variants ----------
    titles_count = int(pack_cfg.get("titles_count", 10))
    beat_sheet_path = ws / "02_script" / "beat_sheet.json"
    beat_sheet = {}
    if beat_sheet_path.exists():
        try:
            beat_sheet = json.loads(beat_sheet_path.read_text())
        except Exception as e:
            logger.warning("S13: beat_sheet.json unreadable: %s", e)

    variants = generate_title_variants(
        incident=incident, beat_sheet=beat_sheet, n=titles_count,
    )
    titles_json = ws / "06_metadata" / "titles.json"
    write_title_variants(variants, titles_json)
    logger.info("S13 titles: %d variants → %s",
                len(variants), titles_json.name)

    # Pick best title for thumbnail compositing (rank 1).
    best_title = variants[0].text if variants else (
        incident.get("company_name", "Episode")
    )

    # ---------- Phase 2: thumbnail variants ----------
    if pack_cfg.get("thumbnails_enabled", True):
        thumb_dir = ws / "05_video" / "thumbnails"
        flux_dir = ws / "03_assets" / "flux"
        try:
            thumb_variants = generate_thumbnail_variants(
                title=best_title,
                incident=incident,
                beat_sheet=beat_sheet,
                flux_dir=flux_dir,
                out_dir=thumb_dir,
                visual_style=episode.get("visual_style") or "V1",
            )
            logger.info("S13 thumbnails: %d variants → %s",
                        len(thumb_variants), thumb_dir.name)
        except Exception as e:
            logger.warning("S13 thumbnail generation failed: %s", e)

    # ---------- Phase 3: Shorts cutter ----------
    if not pack_cfg.get("shorts_enabled", True):
        logger.info("S13: shorts_enabled=false; skipping shorts phase")
        return None

    final_mp4 = ws / "05_video" / "final.mp4"
    if not final_mp4.exists():
        # Preview-mode final_preview.mp4 fallback (shorts still useful)
        prev = ws / "05_video" / "final_preview.mp4"
        if prev.exists():
            final_mp4 = prev
            logger.info("S13 shorts: using final_preview.mp4 "
                        "(preview-mode episode)")
        else:
            logger.warning("S13: no final.mp4 — skipping shorts phase")
            return None

    shorts_count = int(pack_cfg.get("shorts_count", 3))
    shorts_seconds = float(pack_cfg.get("shorts_target_seconds", 30.0))

    script_path = ws / "02_script" / "script.txt"
    script = script_path.read_text() if script_path.exists() else ""

    voice_timing_path = ws / "04_audio" / "voice_timing.json"
    voice_timing = None
    if voice_timing_path.exists():
        try:
            voice_timing = json.loads(voice_timing_path.read_text())
        except Exception:
            voice_timing = None

    windows = pick_windows(
        incident=incident,
        script=script,
        beat_sheet=beat_sheet,
        voice_timing=voice_timing,
        n=shorts_count,
        target_seconds=shorts_seconds,
    )
    if not windows:
        logger.info("S13 shorts: no windows picked")
        return None

    shorts_dir = ws / "05_video" / "shorts"
    shorts_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list = []
    burn_subs = bool(pack_cfg.get("shorts_burn_subtitles", True))

    for w in windows:
        out_path = shorts_dir / f"short_{w.rank:02d}.mp4"

        # Audio extract + ASR for subtitles.
        segments = None
        if burn_subs:
            audio_wav = shorts_dir / f"short_{w.rank:02d}_audio.wav"
            ok = extract_audio_for_window(
                final_mp4, audio_wav,
                start_seconds=w.start_seconds,
                duration_seconds=shorts_seconds,
            )
            if ok:
                segments = transcribe(audio_wav)
                # whisper segments are relative to the audio file
                # we just extracted. Adjust to absolute time so the
                # cut_short's enable= filter is correct.
                if segments:
                    for s in segments:
                        s.start_seconds += w.start_seconds
                        s.end_seconds += w.start_seconds
                # Don't leave loose WAVs around.
                try:
                    audio_wav.unlink()
                except OSError:
                    pass

        ok = cut_short(
            src_mp4=final_mp4,
            out_mp4=out_path,
            start_seconds=w.start_seconds,
            duration_seconds=shorts_seconds,
            burn_subtitles=burn_subs,
            subtitles=segments,
        )
        if ok:
            out_paths.append(out_path)
            logger.info("S13 short %d: %s (%.1fs - %.1fs)",
                        w.rank, out_path.name,
                        w.start_seconds, w.end_seconds)
        else:
            out_paths.append(None)
            logger.warning("S13 short %d FAILED for window %.1fs",
                           w.rank, w.start_seconds)

    write_manifest(windows, out_paths, shorts_dir / "manifest.json")
    logger.info("S13 complete: %d titles, %d shorts",
                len(variants), sum(1 for p in out_paths if p))
    return None
