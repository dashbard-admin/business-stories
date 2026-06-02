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

  Phase 3: Shorts teaser builder
    Calls pipeline.shorts.generate_teaser_script() to compress the
    whole episode into a viral 30-second teaser, renders fast standalone
    TTS, then cuts quickly across reused beat images as 1080x1920
    vertical Shorts with no music/SFX. Outputs at
    05_video/shorts/short_N.mp4 + manifest.json.

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
    build_teaser_short,
    build_teaser_subtitles,
    collect_story_images,
    generate_teaser_script,
    render_short_title_card,
    render_teaser_audio,
    shift_subtitles,
    write_srt,
    write_teaser_manifest,
)
from ..state import find_episode_workspace
from ..thumbnails import generate_variants as generate_thumbnail_variants
from ..titles import generate_variants as generate_title_variants
from ..titles import write_variants as write_title_variants
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

    # ---------- Phase 3: Shorts teaser builder ----------
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

    target_wpm = float(pack_cfg.get("shorts_tts_wpm", 230.0))
    teaser = generate_teaser_script(
        incident=incident,
        script=script,
        beat_sheet=beat_sheet,
        target_seconds=shorts_seconds,
        target_wpm=target_wpm,
    )
    if not teaser:
        logger.info("S13 shorts: no teaser generated")
        return None

    shorts_dir = ws / "05_video" / "shorts"
    shorts_dir.mkdir(parents=True, exist_ok=True)
    (shorts_dir / "teaser_script.txt").write_text(teaser.script)

    try:
        teaser_audio, audio_seconds = render_teaser_audio(
            teaser=teaser,
            narrator_id=episode["narrator"],
            out_dir=shorts_dir,
            target_seconds=shorts_seconds,
            target_wpm=target_wpm,
            tts_speed=float(pack_cfg.get("shorts_tts_speed", 1.85)),
            enforce_wpm=bool(pack_cfg.get("shorts_enforce_tts_wpm", False)),
        )
    except Exception as e:
        logger.warning("S13 shorts teaser audio failed: %s", e)
        return None

    subtitles = build_teaser_subtitles(
        teaser.script,
        audio_seconds,
        max_words=int(pack_cfg.get("shorts_caption_max_words", 6)),
        max_chars=int(pack_cfg.get("shorts_caption_max_chars", 44)),
    )
    base_srt = shorts_dir / "teaser_captions.srt"
    write_srt(subtitles, base_srt)

    out_paths: list = []
    srt_paths: list = []
    image_sets: list = []
    title_card_paths: list = []
    burn_subs = bool(pack_cfg.get("shorts_burn_subtitles", True))
    seconds_per_image = float(pack_cfg.get("shorts_seconds_per_image", 3.75))
    title_card_enabled = bool(pack_cfg.get("shorts_title_card_enabled", True))
    title_card_seconds = (
        max(0.0, float(pack_cfg.get("shorts_title_card_seconds", 1.0)))
        if title_card_enabled else 0.0
    )
    subtitles_for_video = shift_subtitles(subtitles, title_card_seconds)
    images_per_short = max(
        1, int(round(audio_seconds / max(1.0, seconds_per_image)))
    )
    logo_path = _find_company_logo(ws)

    for rank in range(1, shorts_count + 1):
        out_path = shorts_dir / f"short_{rank:02d}.mp4"
        short_srt = shorts_dir / f"short_{rank:02d}.srt"
        image_paths = collect_story_images(
            ws=ws,
            beat_sheet=beat_sheet,
            count=images_per_short,
            offset=(rank - 1) * max(1, images_per_short // 2),
        )
        image_sets.append(image_paths)
        short_title = None
        if image_paths and title_card_seconds > 0:
            short_title = render_short_title_card(
                image_path=image_paths[0],
                logo_path=logo_path,
                out_path=shorts_dir / f"short_{rank:02d}_title_card.png",
                incident=incident,
                rank=rank,
                enabled=title_card_enabled,
            )
        title_card_paths.append(short_title)
        write_srt(subtitles_for_video, short_srt)

        ok = build_teaser_short(
            image_paths=image_paths,
            audio_path=teaser_audio,
            out_mp4=out_path,
            duration_seconds=audio_seconds,
            subtitles=subtitles_for_video,
            burn_subtitles=burn_subs,
            seconds_per_image=seconds_per_image,
            transition_seconds=float(
                pack_cfg.get("shorts_transition_seconds", 0.22)
            ),
            motion_strength=float(pack_cfg.get("shorts_motion_strength", 0.10)),
            title_card_path=short_title,
            title_card_seconds=title_card_seconds,
        )
        if ok:
            out_paths.append(out_path)
            srt_paths.append(short_srt)
            logger.info("S13 short %d: %s (%.1fs teaser)",
                        rank, out_path.name, audio_seconds)
        else:
            out_paths.append(None)
            srt_paths.append(None)
            logger.warning("S13 short %d FAILED for teaser build", rank)

    write_teaser_manifest(
        teaser=teaser,
        out_paths=out_paths,
        srt_paths=srt_paths,
        image_sets=image_sets,
        title_card_paths=title_card_paths,
        manifest_path=shorts_dir / "manifest.json",
        duration_seconds=audio_seconds + title_card_seconds,
        target_wpm=target_wpm,
    )
    logger.info("S13 complete: %d titles, %d shorts",
                len(variants), sum(1 for p in out_paths if p))
    return None


def _find_company_logo(ws):
    title_meta = ws / "03_assets" / "title_logo.json"
    if title_meta.exists():
        try:
            rel = json.loads(title_meta.read_text()).get("local_path")
            if rel:
                p = ws / rel
                if p.exists():
                    return p
        except Exception:
            pass
    pd_dir = ws / "03_assets" / "pd"
    for name in ("company_logo.png", "company_logo.jpg", "logo.png", "logo.jpg"):
        p = pd_dir / name
        if p.exists():
            return p
    for p in sorted(pd_dir.glob("*logo*")):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return p
    return None
