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
            flux_title_png = ws / "03_assets" / "flux" / "title.png"
            prod = cfg.production
            _render_title_card(
                out_path=title_card_png,
                backdrop=flux_title_png if flux_title_png.exists() else None,
                episode_title=episode["incident"]["company_name"],
                year=episode["incident"].get("year_anchor"),
                episode_id=episode.get("id", "EP_unknown"),
                text_color=prod.get("title_card_text_color", "#FFE600"),
                stroke_color=prod.get("title_card_stroke_color", "#D02020"),
                stroke_width=int(prod.get("title_card_stroke_width", 8)),
                font_size_pct=float(prod.get("title_card_font_size_pct", 0.13)),
                padding_pct=float(prod.get("title_card_padding_pct", 0.06)),
                corner=str(prod.get("title_card_corner", "random")),
                uppercase=bool(prod.get("title_card_uppercase", True)),
                show_year=bool(prod.get("title_card_show_year", True)),
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
                text_color=cfg.production.get(
                    "title_card_text_color", "#FFE600"
                ),
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
    # Preview mode (Batch B 2026-05-26): produces final_preview.mp4
    # to signal it's a tone-check render, not for upload.
    preview_mode = bool(episode.get("preview_mode"))
    final_name = "final_preview.mp4" if preview_mode else "final.mp4"
    final_mp4 = ws / "05_video" / final_name
    try:
        concat_clips(clip_paths, final_mix, final_mp4)
    except Exception as e:
        return f"final concat failed: {e}"
    if preview_mode:
        logger.info("S12 PREVIEW MODE: muxed %s (Acts 0+5 only, "
                    "tone-check render)", final_mp4)

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


def _render_title_card(
    *,
    out_path: Path,
    backdrop: Path | None,
    episode_title: str,
    year,
    episode_id: str,
    text_color: str = "#FFE600",
    stroke_color: str = "#000000",
    stroke_width: int = 8,
    font_size_pct: float = 0.13,
    padding_pct: float = 0.06,
    corner: str = "random",
    uppercase: bool = True,
    show_year: bool = True,
) -> None:
    """Render the opening title card in YouTube-thumbnail style.

    Layout: large bold title text in a high-contrast colour (yellow
    body + black stroke by default), slammed into one of the four
    corners with edge padding. No background band or rectangle — the
    stroke alone carries legibility against any backdrop, including
    the FLUX-rendered title.png panel.

    Style choices ported from retail YouTube history channels
    (Tim Reborn History et al). Every visible value is configurable
    via production.title_card_* in config.yaml; only the layout
    geometry (corner / padding logic) is fixed.

    `corner` accepts "random" (deterministic per episode_id),
    "top-left", "top-right", "bottom-left", "bottom-right".

    The small year tag, when shown, lands in the OPPOSITE corner from
    the title — provides date context without crowding the title.
    """
    # ---- backdrop ----
    if backdrop is not None and backdrop.exists():
        with Image.open(backdrop) as bg:
            img = bg.convert("RGB").resize((OUT_W, OUT_H))
    else:
        # No FLUX backdrop — render onto a neutral dark canvas so the
        # yellow + red title still has something to land on.
        img = Image.new("RGB", (OUT_W, OUT_H), color=(15, 18, 22))

    body_rgb = _hex_to_rgb(text_color)
    stroke_rgb = _hex_to_rgb(stroke_color)

    # ---- corner selection ----
    corner_id = _resolve_title_corner(corner, episode_id)
    pad = int(OUT_H * max(0.0, padding_pct))

    # ---- title text fit ----
    title = str(episode_title or "").strip()
    if uppercase:
        title = title.upper()

    # Target font size, then shrink if we can't fit even a 1-word line.
    # Allow up to 65% of the frame width for the title block (leaves
    # breathing room on the side opposite the chosen corner).
    target_size = max(48, int(OUT_H * font_size_pct))
    max_block_w = int(OUT_W * 0.65)
    # Allow the title to occupy up to ~55% of frame height before we
    # call it too tall and reduce the font further.
    max_block_h = int(OUT_H * 0.55)
    title_font, wrapped_lines = _fit_bold_title(
        text=title,
        target_size=target_size,
        max_block_w=max_block_w,
        max_block_h=max_block_h,
        stroke_width=stroke_width,
    )

    # ---- compute title block size ----
    block_w, block_h, line_metrics = _measure_title_block(
        wrapped_lines, title_font, stroke_width=stroke_width,
    )

    # ---- anchor + per-line alignment based on corner ----
    if corner_id.startswith("top"):
        block_y = pad
    else:
        block_y = OUT_H - pad - block_h

    if corner_id.endswith("right"):
        # Right-anchored block: per-line x is computed so the LINE
        # right-edge lands at OUT_W - pad. Per-line, not block, so
        # each line's right edge aligns cleanly.
        align = "right"
        block_x_right = OUT_W - pad
    else:
        align = "left"
        block_x_left = pad

    d = ImageDraw.Draw(img)

    cursor_y = block_y
    for line_text, (line_w, line_h) in zip(wrapped_lines, line_metrics):
        if align == "right":
            x = block_x_right - line_w
        else:
            x = block_x_left
        d.text(
            (x, cursor_y),
            line_text,
            fill=body_rgb,
            font=title_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_rgb,
        )
        cursor_y += line_h

    # ---- year tag in the opposite corner (small, plain) ----
    if show_year and year:
        year_font_size = max(28, int(OUT_H * font_size_pct * 0.25))
        year_font = _bold_font(year_font_size)
        year_text = str(year)
        # Small stroke so the year stays legible without the loud
        # red outline of the title.
        year_stroke_w = max(2, stroke_width // 4)
        ytext_w, ytext_h = _text_size(d, year_text, year_font,
                                       stroke_width=year_stroke_w)

        # Opposite corner: invert both axes from the title corner.
        opp_top = not corner_id.startswith("top")
        opp_right = not corner_id.endswith("right")
        yx = (OUT_W - pad - ytext_w) if opp_right else pad
        yy = pad if opp_top else (OUT_H - pad - ytext_h)
        d.text(
            (yx, yy),
            year_text,
            fill=body_rgb,
            font=year_font,
            stroke_width=year_stroke_w,
            stroke_fill=stroke_rgb,
        )

    img.save(out_path, "PNG")


def _resolve_title_corner(corner: str, episode_id: str) -> str:
    """Map config 'corner' value to one of four concrete corners.
    'random' is seeded by episode_id so re-runs produce identical
    output and a multi-episode channel cycles across all four."""
    corner = (corner or "random").strip().lower().replace("_", "-")
    valid = {"top-left", "top-right", "bottom-left", "bottom-right"}
    if corner in valid:
        return corner
    # Deterministic random.
    import hashlib
    h = hashlib.md5((episode_id or "").encode()).hexdigest()
    seed = int(h[:8], 16)
    return ("top-left", "top-right", "bottom-left", "bottom-right")[seed % 4]


def _bold_font(size: int) -> ImageFont.FreeTypeFont:
    """Locate a bold condensed display font for the title.

    Tries Impact (the YouTube-thumbnail default) first, then Anton
    or Bebas Neue if the operator bundled them, then HelveticaNeue
    Black/Bold as a safe fallback. As a last resort falls back to
    PIL's default bitmap font — workable but visually flat."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/Library/Fonts/Arial Black.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(d: ImageDraw.ImageDraw, text: str,
               font: ImageFont.FreeTypeFont, *, stroke_width: int = 0
               ) -> tuple[int, int]:
    """Measure a single line's bounding box, stroke-aware."""
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_bold_title(*, text: str, target_size: int, max_block_w: int,
                    max_block_h: int, stroke_width: int
                    ) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Find the largest font size at which `text` wraps to fit inside
    (max_block_w, max_block_h). Tries the target size first, shrinks
    by 8 px increments down to a floor of 56 px. Word-wraps greedily
    line-by-line."""
    size = target_size
    floor = 56
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    while size >= floor:
        font = _bold_font(size)
        lines = _wrap_to_width(text, font, max_block_w, draw, stroke_width)
        # Estimate block height as sum of per-line heights.
        total_h = 0
        ok = True
        for ln in lines:
            _, h = _text_size(draw, ln, font, stroke_width=stroke_width)
            total_h += h
        if total_h <= max_block_h:
            return font, lines
        size -= 8
    # Floor: accept overflow if we ran out of headroom.
    font = _bold_font(floor)
    lines = _wrap_to_width(text, font, max_block_w, draw, stroke_width)
    return font, lines


def _wrap_to_width(text: str, font: ImageFont.FreeTypeFont,
                   max_w: int, draw: ImageDraw.ImageDraw,
                   stroke_width: int) -> list[str]:
    """Greedy word-wrap for the title. Falls back to character-break
    if a single word is wider than max_w."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        tw, _ = _text_size(draw, trial, font, stroke_width=stroke_width)
        if tw <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _measure_title_block(lines: list[str], font: ImageFont.FreeTypeFont,
                         *, stroke_width: int
                         ) -> tuple[int, int, list[tuple[int, int]]]:
    """Return (block_w, block_h, per_line_metrics).

    Per-line metrics carry each rendered line's bounding-box size, so
    the caller can right-align by line if needed.
    """
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    metrics: list[tuple[int, int]] = []
    block_w = 0
    block_h = 0
    for ln in lines:
        w, h = _text_size(draw, ln, font, stroke_width=stroke_width)
        metrics.append((w, h))
        block_w = max(block_w, w)
        block_h += h
    return block_w, block_h, metrics


def _render_closing_card(*, out_path: Path, backdrop: Path | None,
                         workspace: Path, brand_color: str,
                         text_color: str = "#FFE600") -> None:
    """Render the closing source-attribution card.

    Style intentionally matches the title card's colour palette
    (all text in yellow, no stroke) but uses a thinner, normal-width
    bold sans-serif so smaller text stays legible. No brand bar — a
    clean text-only layout over the optionally-blurred FLUX credits
    backdrop. Lines containing 'Unknown' or 'Untitled' are dropped
    so missing source metadata doesn't clutter the credits.
    """
    fill_rgb = _hex_to_rgb(text_color)

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
    d.text((100, 410), "SOURCES & ATTRIBUTION",
           fill=fill_rgb, font=_closing_font(48))

    lines: list[str] = []
    inv_path = workspace / "00_research" / "source_inventory.json"
    if inv_path.exists():
        try:
            for s in json.loads(inv_path.read_text())["sources"][:5]:
                pub = s.get("publisher", "")
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

    # Drop lines that contain placeholder tokens — they show up when
    # a source has no title or a PD asset has no publisher recorded,
    # and they make the credits look broken. Empty lines (blank
    # spacers between sections) are preserved.
    lines = [ln for ln in lines if not _is_placeholder_line(ln)]

    d.multiline_text((100, 490), "\n".join(lines),
                     fill=fill_rgb, font=_closing_font(30), spacing=8)
    img.save(out_path, "PNG")


_PLACEHOLDER_TOKENS = ("unknown", "untitled")


def _is_placeholder_line(line: str) -> bool:
    """True if a line has visible content AND that content contains
    a placeholder token (e.g. 'Unknown publisher' or 'Untitled')
    that would clutter the credits. Empty lines (used as spacers
    between sections) are deliberately kept."""
    s = (line or "").strip()
    if not s:
        return False
    low = s.lower()
    return any(t in low for t in _PLACEHOLDER_TOKENS)


def _closing_font(size: int) -> ImageFont.FreeTypeFont:
    """Bold-but-thinner font for the closing credits.

    Resembles the title's Impact-style boldness but with normal-width
    letterforms so text stays legible at smaller sizes (24-48 px).
    Prefers Helvetica Neue / Helvetica Bold via TTC index (typical
    on macOS), falls back to standalone Arial Bold, then regular
    Helvetica, and finally PIL's default.
    """
    # (path, index) candidates. Index targets the Bold face inside
    # a macOS .ttc collection; standalone .ttf files ignore the
    # index argument harmlessly.
    candidates: list[tuple[str, int]] = [
        # HelveticaNeue.ttc — Bold face is typically at index 1 on
        # current macOS; older versions sometimes index=2.
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 2),
        # Helvetica.ttc — same pattern.
        ("/System/Library/Fonts/Helvetica.ttc", 1),
        ("/System/Library/Fonts/Helvetica.ttc", 2),
        # Standalone Bold TTFs.
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
        ("/Library/Fonts/Arial Bold.ttf", 0),
        # Regular weight as the final usable fallback (still thinner
        # than Impact, just not bold).
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ]
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size, index=index)
        except Exception:
            continue
    return ImageFont.load_default()


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
