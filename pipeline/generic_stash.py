"""Generic stock photo stash manager.

Operator-curated library of generic atmospheric photos at
assets/generic_stock/. S8 uses these as a fallback layer between
incident-specific PD assets and FLUX generation:

  Tier 1: incident-specific PD assets (S5 Phases 0-2)
  Tier 2: generic stash (this module)
  Tier 3: FLUX synthetic generation

The stash never competes with incident-specific PD assets in S8's
Pass 1 / Pass 2 — even if a stash entry has higher cosine sim to a
beat. The Pass 2.5 stage in S8 evaluates stash entries only for beats
that still routed to FLUX after Pass 1 and Pass 2.

Lifecycle:
  1. Operator drops images into assets/generic_stock/.
  2. S5 Phase 5 calls ensure_stash_manifest() which walks the dir,
     captions new/changed files via VLM, and persists manifest.json.
  3. Stash entries are appended to per-episode asset_manifest.json
     with is_generic_stash=True.

For business stories the operator stash should hold things like:
generic office interiors, boardroom shots, factory floors, founder
silhouettes, market crowds, financial newsprint, abstract growth
charts — the kind of beats every story has but specific PD assets
rarely cover.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from .config import load_config

logger = logging.getLogger("hermes.generic_stash")

STASH_SUBDIR = "generic_stock"
MANIFEST_FILENAME = "manifest.json"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

MIN_INPUT_PX = 200


def stash_dir() -> Path | None:
    try:
        return load_config().assets_dir / STASH_SUBDIR
    except Exception:
        return None


def manifest_path() -> Path | None:
    d = stash_dir()
    return None if d is None else d / MANIFEST_FILENAME


def ensure_stash_manifest(vlm=None) -> list[dict]:
    """Walk the stash dir, caption new/changed files, return manifest
    entries suitable for appending to a per-episode asset_manifest.

    Each entry has the same shape as a PD asset manifest entry plus:
      - is_generic_stash: True
      - source: "generic_stash"
      - local_path: ABSOLUTE filesystem path

    VLM is optional. If None or per-file fails, the file's stem (with
    underscores/hyphens → spaces) is used as the caption.

    Returns [] if the stash dir doesn't exist.
    """
    d = stash_dir()
    if d is None or not d.exists():
        logger.info("generic_stash: directory %s not present; skipping", d)
        return []

    mp = manifest_path()
    cache: dict[str, dict] = {}
    if mp and mp.exists():
        try:
            cache = (json.loads(mp.read_text()) or {}).get("files", {})
        except Exception as e:
            logger.warning(
                "generic_stash: manifest %s unreadable (%s); rebuilding", mp, e,
            )
            cache = {}

    entries: list[dict] = []
    changed = False
    skipped_small = 0
    captioned = 0

    candidates: list[Path] = []
    for f in sorted(d.iterdir()):
        if not f.is_file() or f.name == MANIFEST_FILENAME:
            continue
        if f.suffix.lower() not in SUPPORTED_EXTS:
            continue
        candidates.append(f)

    to_caption_est = 0
    for f in candidates:
        cached = cache.get(f.name)
        mtime = f.stat().st_mtime
        if (
            cached
            and abs(cached.get("mtime", 0) - mtime) < 1.0
            and (cached.get("caption") or "")
        ):
            continue
        to_caption_est += 1
    if to_caption_est > 0:
        logger.info(
            "generic_stash: %d/%d file(s) need (re-)captioning (~%d s/file)",
            to_caption_est, len(candidates), 4,
        )

    INCREMENTAL_SAVE_EVERY = 5

    for f in candidates:
        try:
            with Image.open(f) as img:
                w, h = img.size
        except Exception as e:
            logger.warning("generic_stash: can't open %s: %s", f.name, e)
            continue
        if min(w, h) < MIN_INPUT_PX:
            skipped_small += 1
            continue

        cached = cache.get(f.name)
        mtime = f.stat().st_mtime
        valid_cache = (
            cached
            and abs(cached.get("mtime", 0) - mtime) < 1.0
            and cached.get("width") == w
            and cached.get("height") == h
            and (cached.get("caption") or "")
        )
        if valid_cache:
            caption = cached["caption"]
            description = cached.get("description") or caption
        else:
            caption, description = _caption_file(f, vlm)
            cache[f.name] = {
                "caption": caption,
                "description": description,
                "width": w,
                "height": h,
                "mtime": mtime,
            }
            changed = True
            captioned += 1
            logger.info(
                "generic_stash: captioned %s (%d/%d) → %s",
                f.name, captioned, to_caption_est, caption[:100],
            )
            if mp and captioned % INCREMENTAL_SAVE_EVERY == 0:
                try:
                    mp.write_text(json.dumps({"files": cache}, indent=2))
                except Exception as e:
                    logger.warning(
                        "generic_stash: incremental cache save failed: %s", e,
                    )

        entries.append({
            "id": f"stash_{_slug(f.stem)}",
            "local_path": str(f.resolve()),
            "original_url": None,
            "source_page": None,
            "license": "operator-supplied",
            "attribution_required": False,
            "attribution_text": None,
            "width": w,
            "height": h,
            "direct_use_eligible": min(w, h) >= 800,
            "title": f.stem.replace("_", " ").replace("-", " "),
            "description": description,
            "caption": caption,
            "is_generic_stash": True,
            "source": "generic_stash",
        })

    if changed and mp:
        mp.write_text(json.dumps({"files": cache}, indent=2))

    logger.info(
        "generic_stash: %d entries (%d newly captioned; %d too small skipped)",
        len(entries), captioned, skipped_small,
    )
    return entries


def _caption_file(f: Path, vlm) -> tuple[str, str]:
    fallback = f.stem.replace("_", " ").replace("-", " ").strip()
    if vlm is None:
        return fallback, fallback
    try:
        cap = vlm.caption_image(f, incident_name="generic business imagery")
    except Exception as e:
        logger.warning("generic_stash: VLM failed on %s (%s); using filename",
                       f.name, e)
        return fallback, fallback
    cap = (cap or "").strip()
    if not cap:
        return fallback, fallback
    return cap, cap


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:60]
