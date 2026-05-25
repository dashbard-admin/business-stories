"""S12 — Video Assembly.

For each beat:
  1. Look up the image (PD asset path or FLUX asset path on the beat).
  2. Render a Ken Burns clip of the beat's duration with the
     specified motion.
Then prepend a title-card clip + append a closing source-attribution
clip, concat with final_mix.wav, and generate SRT + VTT captions
aligned to voice_timing.json.

Title and closing cards use the FLUX-rendered backdrops from S09
(03_assets/flux/title.png and credits.png) when available, with text
composited over them via Pillow. If the FLUX renders aren't on disk
the cards fall back to a programmatic Pillow design.

Inputs:  02_script/beat_sheet.json
         04_audio/voice_timing.json
         04_audio/final_mix.wav
         03_assets/flux/title.png, credits.png (optional)
Outputs: 05_video/final.mp4
         05_video/captions.srt + captions.vtt
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ..config import load_config
from ..ffmpeg_builder import (
    concat_clips,
    get_duration_seconds,
    ken_burns_clip,
)
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s12")

OUT_W, OUT_H = 1920, 1080


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    beat_sheet = json.loads((ws / "02_script" / "beat_sheet.json").read_text())
    beats = beat_sheet["beats"]
    timing_path = ws / "04_audio" / "voice_timing.json"
    if not timing_path.exists():
        return "no voice_timing.json"
    timing = json.loads(timing_path.read_text())
    timing_by_beat = {b["beat_id"]: b for b in timing["beats"]}

    final_mix = ws / "04_audio" / "final_mix.wav"
    if not final_mix.exists():
        return "no final_mix.wav"

    clips_dir = ws / "05_video" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Per-clip fade in / fade out. Applied to title card, every beat,
    # and the closing card. Configurable via production.fade_in_seconds
    # and production.fade_out_seconds. Set to 0 to disable either side.
    fade_in = float(cfg.production.get("fade_in_seconds", 0.0))
    fade_out = float(cfg.production.get("fade_out_seconds", 0.0))

    # The new 6-act writer prompt is explicit: NO INTRO, NO LOGO,
    # NO "WELCOME BACK". The first frame should be Act 0's cold open.
    # `production.opening_title_card_seconds` defaults to 0 (disabled).
    # Set to 2-8 in config.yaml to re-enable a brand stamp at the cost
    # of the cold-open 15s retention window.
    title_card_seconds = float(cfg.production.get("opening_title_card_seconds", 0))
    closing_card_seconds = float(cfg.production.get("closing_card_seconds", 5))

    clip_paths: list[Path] = []

    # ----- optional opening title card -----
    if title_card_seconds > 0:
        title_clip = clips_dir / "00_title.mp4"
        if not _clip_is_valid(title_clip):
            if title_clip.exists():
                try:
                    title_clip.unlink()
                except Exception:
                    pass
            title_card_png = ws / "05_video" / "title_card.png"
            narrator_id = episode.get("narrator")
            narrator_name = None
            if narrator_id:
                try:
                    narr = cfg.narrator_by_id(narrator_id)
                    narrator_name = (narr or {}).get("name")
                except Exception:
                    pass

            flux_title_png = ws / "03_assets" / "flux" / "title.png"
            _render_title_card(
                out_path=title_card_png,
                backdrop=flux_title_png if flux_title_png.exists() else None,
                channel=cfg.channel["name"],
                episode_title=episode["incident"]["company_name"],
                year=episode["incident"].get("year_anchor"),
                brand_color=cfg.channel.get("brand_color", "#1a2b3c"),
                narrator_name=narrator_name,
            )
            try:
                ken_burns_clip(
                    title_card_png, title_card_seconds, "slow_zoom_in",
                    title_clip,
                    fade_in_seconds=fade_in,
                    fade_out_seconds=fade_out,
                )
            except Exception as e:
                return f"title card render failed: {e}"
        clip_paths.append(title_clip)
        logger.info("S12 opening title card: %.1fs", title_card_seconds)
    else:
        logger.info("S12 opening title card: disabled "
                    "(production.opening_title_card_seconds=0)")

    # ----- per-beat clips -----
    for b in beats:
        beat_id = b["beat_id"]
        clip_path = clips_dir / f"{beat_id}.mp4"

        if _clip_is_valid(clip_path):
            clip_paths.append(clip_path)
            continue
        if clip_path.exists():
            logger.warning("beat %s: cached clip %s invalid; re-rendering",
                           beat_id, clip_path.name)
            try:
                clip_path.unlink()
            except Exception:
                pass

        image_path: Path | None = None
        if "pd_asset_path" in b:
            image_path = ws / b["pd_asset_path"]
        elif "flux_asset_path" in b:
            image_path = ws / b["flux_asset_path"]
        else:
            disk_path = ws / "03_assets" / "flux" / f"{beat_id}.png"
            if disk_path.exists() and disk_path.stat().st_size > 1000:
                image_path = disk_path
                b["flux_asset_path"] = str(disk_path.relative_to(ws))
                logger.warning("beat %s missing image path; recovered from disk",
                               beat_id)
        if image_path is None:
            return f"beat {beat_id} has no image"
        if not image_path.exists():
            return f"image missing for beat {beat_id}: {image_path}"

        t = timing_by_beat.get(beat_id)
        duration = (t["duration_seconds"]
                    if t and t["duration_seconds"] > 0.5
                    else b.get("estimated_seconds", 5.0))

        motion = b.get("ken_burns_motion", "slow_zoom_in")
        try:
            ken_burns_clip(
                image_path, float(duration), motion, clip_path,
                fade_in_seconds=fade_in,
                fade_out_seconds=fade_out,
            )
        except Exception as e:
            return f"ken burns render failed for {beat_id}: {e}"
        clip_paths.append(clip_path)

    # ----- optional closing source-attribution card -----
    if closing_card_seconds > 0:
        closing_clip = clips_dir / "zz_closing.mp4"
        if not _clip_is_valid(closing_clip):
            if closing_clip.exists():
                try:
                    closing_clip.unlink()
                except Exception:
                    pass
            closing_png = ws / "05_video" / "closing_card.png"
            flux_credits_png = ws / "03_assets" / "flux" / "credits.png"
            _render_closing_card(
                out_path=closing_png,
                backdrop=flux_credits_png if flux_credits_png.exists() else None,
                workspace=ws,
                brand_color=cfg.channel.get("brand_color", "#1a2b3c"),
            )
            try:
                ken_burns_clip(
                    closing_png, closing_card_seconds, "hold_still",
                    closing_clip,
                    fade_in_seconds=fade_in,
                    fade_out_seconds=fade_out,
                )
            except Exception as e:
                return f"closing card render failed: {e}"
        clip_paths.append(closing_clip)
        logger.info("S12 closing card: %.1fs", closing_card_seconds)

    # ----- concat -----
    final_mp4 = ws / "05_video" / "final.mp4"
    try:
        concat_clips(clip_paths, final_mix, final_mp4)
    except Exception as e:
        return f"final concat failed: {e}"

    # ----- captions -----
    # Caption timestamps are offset by however many seconds of
    # non-beat content (title card) play before the first beat.
    # When the title card is disabled (default per new prompt), the
    # offset is 0 and the first caption fires at t=0.
    try:
        srt, vtt = _build_captions(beats, timing, caption_offset=title_card_seconds)
        (ws / "05_video" / "captions.srt").write_text(srt)
        (ws / "05_video" / "captions.vtt").write_text(vtt)
    except Exception as e:
        logger.warning("caption build failed (non-fatal): %s", e)

    try:
        final_dur = get_duration_seconds(final_mp4)
        logger.info("S12 complete: %s (%d clips, %.1fs)",
                    final_mp4, len(clip_paths), final_dur)
    except Exception:
        logger.info("S12 complete: %s (%d clips)", final_mp4, len(clip_paths))
    return None


# ------------------ helpers ------------------

def _clip_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        return get_duration_seconds(path) > 0.0
    except Exception:
        return False


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore


def _render_title_card(*, out_path: Path, backdrop: Path | None,
                       channel: str, episode_title: str,
                       year, brand_color: str,
                       narrator_name: str | None) -> None:
    """Render the opening title card. When `backdrop` is provided
    (FLUX-rendered comic panel from S09), composite the title text
    over the bottom third of it with a semi-transparent dark band
    behind the text for legibility. Otherwise fall back to a
    programmatic dark card with brand bar."""
    rgb = _hex_to_rgb(brand_color)

    if backdrop is not None and backdrop.exists():
        with Image.open(backdrop) as bg:
            img = bg.convert("RGB").resize((OUT_W, OUT_H))
        # Translucent dark band across the bottom third for text legibility
        overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([0, OUT_H * 2 // 3, OUT_W, OUT_H], fill=(0, 0, 0, 160))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    else:
        img = Image.new("RGB", (OUT_W, OUT_H), color=(10, 12, 16))

    d = ImageDraw.Draw(img)
    d.rectangle([0, 940, OUT_W, 948], fill=rgb)
    d.text((100, 750), channel.upper(), fill=(220, 220, 220), font=_font(36))
    title_font = _font(64)
    wrapped = "\n".join(textwrap.wrap(str(episode_title), width=32))
    d.text((100, 800), wrapped, fill=(245, 245, 245), font=title_font)
    if year:
        d.text((100, 1010), str(year), fill=(190, 190, 190), font=_font(28))
    if narrator_name:
        d.text((OUT_W - 700, 1010), f"Narrator: {narrator_name}",
               fill=(170, 170, 170), font=_font(22))
    img.save(out_path, "PNG")


def _render_closing_card(*, out_path: Path, backdrop: Path | None,
                         workspace: Path, brand_color: str) -> None:
    rgb = _hex_to_rgb(brand_color)

    if backdrop is not None and backdrop.exists():
        with Image.open(backdrop) as bg:
            base = bg.convert("RGB").resize((OUT_W, OUT_H))
        # Apply a slight blur to push the backdrop further behind the
        # attribution text — the credits panel was rendered with a
        # composition that leaves the lower third uncluttered, but a
        # gentle blur improves legibility further.
        base = base.filter(ImageFilter.GaussianBlur(radius=2))
        # Translucent dark band across the lower 70% for text
        overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([0, OUT_H * 1 // 3, OUT_W, OUT_H], fill=(0, 0, 0, 180))
        img = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    else:
        img = Image.new("RGB", (OUT_W, OUT_H), color=(8, 10, 14))

    d = ImageDraw.Draw(img)
    d.rectangle([0, 380, OUT_W, 388], fill=rgb)
    d.text((100, 410), "SOURCES & ATTRIBUTION",
           fill=(220, 220, 220), font=_font(36))

    lines: list[str] = []
    inv_path = workspace / "00_research" / "source_inventory.json"
    if inv_path.exists():
        try:
            for s in json.loads(inv_path.read_text())["sources"][:5]:
                pub = s.get("publisher", "?")
                tit = (s.get("title") or "")[:60]
                lines.append(f"• {pub} — {tit}")
        except Exception:
            pass

    manifest_path = workspace / "03_assets" / "asset_manifest.json"
    if manifest_path.exists():
        try:
            pd = json.loads(manifest_path.read_text()).get("pd_assets", [])
            if pd:
                lines.append("")
                lines.append("Visual sources include public-domain works from:")
                seen = set()
                for a in pd[:6]:
                    src = a.get("source_page") or ""
                    dom = src.split("/")[2] if "//" in src else src
                    if dom and dom not in seen:
                        seen.add(dom)
                        lines.append(f"  · {dom}")
        except Exception:
            pass

    lines.append("")
    lines.append("AI-assisted narration and AI-assisted illustrations where noted.")
    lines.append("All facts traced to references above.")

    d.multiline_text((100, 490), "\n".join(lines),
                     fill=(230, 230, 230), font=_font(24), spacing=8)
    img.save(out_path, "PNG")


def _build_captions(
    beats: list[dict], timing: dict, *, caption_offset: float = 0.0,
) -> tuple[str, str]:
    """Build SRT + VTT captions.

    `caption_offset` is the number of seconds of non-beat content
    (title card) that play BEFORE the first beat in the assembled
    video. Captions are shifted by this amount so they line up with
    the voice track in the muxed final.mp4.
    """
    timing_by_beat = {b["beat_id"]: b for b in timing["beats"]}
    offset = float(caption_offset)

    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]
    cue_idx = 0

    for b in beats:
        t = timing_by_beat.get(b["beat_id"])
        if not t:
            continue
        text = (b.get("script_text") or "").strip()
        if not text:
            continue
        sentences = _split_sentences(text)
        if not sentences:
            continue
        total_words = sum(len(s.split()) for s in sentences) or 1
        start = t["start_seconds"] + offset
        dur = t["duration_seconds"]
        cursor = start
        for s in sentences:
            sw = max(1, len(s.split()))
            seg_dur = dur * (sw / total_words)
            s_start = cursor
            s_end = min(start + dur, cursor + seg_dur)
            cursor = s_end
            cue_idx += 1
            srt_lines.append(str(cue_idx))
            srt_lines.append(f"{_ts_srt(s_start)} --> {_ts_srt(s_end)}")
            srt_lines.append(s.strip())
            srt_lines.append("")
            vtt_lines.append(f"{_ts_vtt(s_start)} --> {_ts_vtt(s_end)}")
            vtt_lines.append(s.strip())
            vtt_lines.append("")

    return "\n".join(srt_lines), "\n".join(vtt_lines)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")
