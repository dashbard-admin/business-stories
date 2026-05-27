"""Thumbnail variant generator (Batch D 2026-05-27).

Generates 5 YouTube-thumbnail variants per episode by compositing
Pillow text + overlays onto FLUX-rendered beat images. The five
fixed layouts target different click psychology:

  1. founder_closeup    — Founder big-face thumbnail with a huge
                          shocked-or-determined expression cue.
                          Universal default; works for any topic.
  2. split_frame        — Left half: a "before" image (early founder
                          / first product). Right half: an "after"
                          image (collapsed building, courtroom).
                          Arrow between. Works for rise-and-fall.
  3. big_number         — Centred founder image, MASSIVE yellow
                          number overlay (e.g. "$9 BILLION", "9,000
                          STORES"). Best for shock/decline stories.
  4. shocked_face       — Founder face cropped tight, with a giant
                          red "FAILED" / "BANKRUPT" stamp diagonal.
                          For scandal_postmortem only.
  5. noir               — V2 visual style: heavy shadow, single
                          accent color, founder silhouetted. For
                          dark / decline stories on the V2 style.

YouTube's native title-test A/B since Dec 2023 supports 3 thumbnail
variants per video — the operator picks the strongest 3 from this
batch and uploads.

All thumbnails are 1280x720 JPG at 90% quality (YouTube spec).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import load_config

logger = logging.getLogger("hermes.thumbnails")

THUMB_W = 1280
THUMB_H = 720

DEFAULT_LAYOUTS = (
    "founder_closeup",
    "split_frame",
    "big_number",
    "shocked_face",
    "noir",
)


@dataclass
class ThumbnailVariant:
    layout: str
    path: Path
    text: str
    backdrop_beat_id: str


def _load_font(size_px: int, *, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Best-effort font load. Impact when available (default for
    YouTube thumbnails for 20 years); fallback chain otherwise."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size_px)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _fit_to_thumb(img: Image.Image) -> Image.Image:
    """Resize + center-crop to 1280x720."""
    target_ratio = THUMB_W / THUMB_H
    w, h = img.size
    src_ratio = w / h
    if src_ratio > target_ratio:
        # Source is wider — crop horizontally.
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        img = img.crop((0, y0, w, y0 + new_h))
    return img.resize((THUMB_W, THUMB_H), Image.LANCZOS)


def _draw_text_with_stroke(
    img: Image.Image,
    text: str,
    *,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: str = "#FFE600",
    stroke: str = "#000000",
    stroke_w: int = 8,
    anchor: str = "lt",
) -> None:
    d = ImageDraw.Draw(img)
    d.text((x, y), text, font=font, fill=fill,
           stroke_width=stroke_w, stroke_fill=stroke, anchor=anchor)


def _pick_backdrop(beat_sheet: dict, flux_dir: Path,
                   prefer_intent: str | None = None) -> tuple[Path | None, str]:
    """Pick the strongest beat image as backdrop. If `prefer_intent`
    is set, look for that visual_intent first. Returns (path, beat_id).
    """
    beats = beat_sheet.get("beats", [])
    if prefer_intent:
        for b in beats:
            if (b.get("visual_intent") or "") == prefer_intent:
                p = flux_dir / f"{b.get('beat_id', '')}.png"
                if p.exists():
                    return p, b.get("beat_id", "")

    # Default: scan for the FIRST beat with an existing flux render
    # that's in Act 0 or Act 4 (most dramatic visual moments).
    for b in beats:
        bid = b.get("beat_id", "")
        if not bid:
            continue
        p = flux_dir / f"{bid}.png"
        if p.exists():
            return p, bid
    return None, ""


def _render_founder_closeup(
    backdrop: Image.Image, title: str, *, episode_meta: dict,
) -> Image.Image:
    """Layout 1: big face, title in lower third. Yellow body + black
    stroke matching the title-card styling."""
    img = backdrop.copy()
    # Slight darken overlay on lower half for legibility.
    dark = Image.new("RGBA", (THUMB_W, THUMB_H // 2), (0, 0, 0, 110))
    img.paste(dark, (0, THUMB_H // 2), dark)

    font_main = _load_font(int(THUMB_H * 0.13))
    _draw_text_with_stroke(
        img, title.upper()[:60],
        x=THUMB_W // 2, y=int(THUMB_H * 0.72),
        font=font_main, anchor="mm",
    )
    return img


def _render_split_frame(
    backdrop: Image.Image, title: str, *, episode_meta: dict,
) -> Image.Image:
    """Layout 2: vertical split. Left = backdrop, right = darkened
    duplicate with a red tint. Centered arrow between."""
    half_w = THUMB_W // 2
    left = backdrop.copy().crop((0, 0, half_w, THUMB_H))
    right = backdrop.copy().crop((half_w, 0, THUMB_W, THUMB_H))

    # Tint the right half red (the "after" / failure side).
    tint = Image.new("RGBA", (half_w, THUMB_H), (160, 20, 20, 110))
    right_rgba = right.convert("RGBA")
    right_rgba.paste(tint, (0, 0), tint)
    right = right_rgba.convert("RGB")

    img = Image.new("RGB", (THUMB_W, THUMB_H), "black")
    img.paste(left, (0, 0))
    img.paste(right, (half_w, 0))

    # Center seam: yellow arrow.
    font_arrow = _load_font(int(THUMB_H * 0.30))
    _draw_text_with_stroke(
        img, "→",
        x=THUMB_W // 2, y=THUMB_H // 2,
        font=font_arrow, anchor="mm",
    )

    # Title across the bottom.
    font_main = _load_font(int(THUMB_H * 0.10))
    _draw_text_with_stroke(
        img, title.upper()[:50],
        x=THUMB_W // 2, y=int(THUMB_H * 0.88),
        font=font_main, anchor="mm",
    )
    return img


def _render_big_number(
    backdrop: Image.Image, title: str, *, episode_meta: dict,
    big_number: str = "",
) -> Image.Image:
    """Layout 3: massive yellow number centered, backdrop darkened."""
    img = backdrop.copy()
    # Darken whole image.
    dark = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 130))
    img.paste(dark, (0, 0), dark)

    number = big_number or _extract_first_number(title) or "$1B"
    font_big = _load_font(int(THUMB_H * 0.32))
    _draw_text_with_stroke(
        img, number, x=THUMB_W // 2, y=int(THUMB_H * 0.42),
        font=font_big, anchor="mm", stroke_w=12,
    )

    # Small title underneath.
    font_small = _load_font(int(THUMB_H * 0.07))
    _draw_text_with_stroke(
        img, title.upper()[:65],
        x=THUMB_W // 2, y=int(THUMB_H * 0.82),
        font=font_small, anchor="mm", stroke_w=5,
    )
    return img


def _render_shocked_face(
    backdrop: Image.Image, title: str, *, episode_meta: dict,
) -> Image.Image:
    """Layout 4: face tight + diagonal "FAILED" / "BANKRUPT" stamp.
    For scandal_postmortem stories."""
    img = backdrop.copy()
    stamp_word = "BANKRUPT"
    if episode_meta.get("story_kind") == "scandal_postmortem":
        stamp_word = "FAILED"

    font_stamp = _load_font(int(THUMB_H * 0.18))
    # Render stamp on a transparent overlay, rotate, then composite.
    txt_w, txt_h = font_stamp.getbbox(stamp_word)[2:]
    stamp_img = Image.new("RGBA", (txt_w + 80, txt_h + 60),
                          (0, 0, 0, 0))
    d = ImageDraw.Draw(stamp_img)
    d.text((40, 30), stamp_word, font=font_stamp,
           fill="#FF1A1A", stroke_width=8, stroke_fill="#000000")
    stamp_img = stamp_img.rotate(-18, expand=True, resample=Image.BICUBIC)
    # Paste in upper-right area.
    sw, sh = stamp_img.size
    img.paste(stamp_img,
              (int(THUMB_W * 0.55), int(THUMB_H * 0.18)),
              stamp_img)

    # Title across the bottom.
    font_main = _load_font(int(THUMB_H * 0.09))
    _draw_text_with_stroke(
        img, title.upper()[:60],
        x=THUMB_W // 2, y=int(THUMB_H * 0.87),
        font=font_main, anchor="mm",
    )
    return img


def _render_noir(
    backdrop: Image.Image, title: str, *, episode_meta: dict,
) -> Image.Image:
    """Layout 5: heavy shadow, desaturated, single accent. V2 vibe."""
    img = backdrop.copy()
    # Desaturate.
    img = img.convert("L").convert("RGB")
    # Slight blur on bottom half for tension.
    bottom = img.crop((0, THUMB_H // 2, THUMB_W, THUMB_H))
    bottom = bottom.filter(ImageFilter.GaussianBlur(radius=3))
    img.paste(bottom, (0, THUMB_H // 2))
    # Dark overlay.
    dark = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 90))
    img = img.convert("RGBA")
    img.paste(dark, (0, 0), dark)
    img = img.convert("RGB")

    # Accent-color (amber) title.
    font_main = _load_font(int(THUMB_H * 0.11))
    _draw_text_with_stroke(
        img, title.upper()[:55],
        x=THUMB_W // 2, y=int(THUMB_H * 0.78),
        font=font_main, anchor="mm",
        fill="#FFAA33", stroke="#000000", stroke_w=8,
    )
    return img


def _extract_first_number(text: str) -> str | None:
    """Pull the first $-amount or numeric-with-unit out of a title."""
    import re
    m = re.search(
        r"\$\d+(?:[.,]\d+)*\s*(?:[KMBT]|billion|million|thousand)?",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(0).strip().upper()
    m = re.search(r"\b\d{4,}\b", text)
    if m:
        return m.group(0)
    return None


def generate_variants(
    *,
    title: str,
    incident: dict,
    beat_sheet: dict,
    flux_dir: Path,
    out_dir: Path,
    layouts: tuple[str, ...] = DEFAULT_LAYOUTS,
    visual_style: str = "V1",
) -> list[ThumbnailVariant]:
    """Generate one thumbnail per layout and write 1280x720 JPGs to
    `out_dir`. Skips layouts that don't suit the story kind (e.g. the
    noir layout requires visual_style=V2 for stylistic coherence)."""
    cfg = load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants: list[ThumbnailVariant] = []

    # Pick the strongest backdrop (default: any rendered beat).
    backdrop_path, backdrop_beat_id = _pick_backdrop(
        beat_sheet, flux_dir,
        prefer_intent="founder_portrait",
    )
    if not backdrop_path:
        logger.warning("thumbnails: no FLUX backdrop available; skipping")
        return []

    try:
        raw_backdrop = Image.open(backdrop_path).convert("RGB")
        backdrop = _fit_to_thumb(raw_backdrop)
    except Exception as e:
        logger.warning("thumbnails: backdrop load failed (%s); skipping", e)
        return []

    # Channel logo overlay (Q-D2 confirmed: yes, with knob).
    show_logo = bool(cfg.packaging.get("show_channel_logo", True))
    logo_path = (cfg.assets_dir / "branding" / "channel_mark.png")
    logo: Image.Image | None = None
    if show_logo and logo_path.exists():
        try:
            logo = Image.open(logo_path).convert("RGBA")
            # Scale to ~10% of frame width.
            new_w = int(THUMB_W * 0.10)
            ratio = new_w / logo.width
            logo = logo.resize(
                (new_w, int(logo.height * ratio)),
                Image.LANCZOS,
            )
        except Exception as e:
            logger.warning("thumbnails: logo load failed (%s)", e)
            logo = None

    episode_meta = {
        "story_kind": incident.get("story_kind", ""),
        "founder": incident.get("founder_or_protagonist", ""),
        "company": incident.get("company_name", ""),
    }

    big_number = _extract_first_number(title) or _extract_first_number(
        incident.get("one_line_pitch", "")
    ) or ""

    for layout in layouts:
        try:
            if layout == "founder_closeup":
                img = _render_founder_closeup(backdrop, title,
                                              episode_meta=episode_meta)
            elif layout == "split_frame":
                img = _render_split_frame(backdrop, title,
                                          episode_meta=episode_meta)
            elif layout == "big_number":
                img = _render_big_number(backdrop, title,
                                         episode_meta=episode_meta,
                                         big_number=big_number)
            elif layout == "shocked_face":
                img = _render_shocked_face(backdrop, title,
                                           episode_meta=episode_meta)
            elif layout == "noir":
                # Only emit for V2 style episodes.
                if visual_style != "V2":
                    logger.info("thumbnails: skipping noir layout "
                                "(visual_style=%s, not V2)", visual_style)
                    continue
                img = _render_noir(backdrop, title,
                                   episode_meta=episode_meta)
            else:
                logger.warning("unknown layout %r; skipping", layout)
                continue

            # Composite channel logo (corner, low-right) if available.
            if logo is not None:
                img = img.convert("RGBA")
                pad = int(THUMB_H * 0.03)
                img.alpha_composite(
                    logo,
                    (THUMB_W - logo.width - pad, THUMB_H - logo.height - pad),
                )
                img = img.convert("RGB")

            path = out_dir / f"thumb_{layout}.jpg"
            img.save(path, "JPEG", quality=90)
            variants.append(ThumbnailVariant(
                layout=layout, path=path, text=title,
                backdrop_beat_id=backdrop_beat_id,
            ))
        except Exception as e:
            logger.warning("thumbnail layout %s failed: %s", layout, e)
            continue

    logger.info("thumbnails: generated %d variants in %s",
                len(variants), out_dir)
    return variants
