"""S05 — Public-Domain Asset Hunt.

Drives the browser through Wikimedia Commons + Wikipedia + LoC +
archive.org + business-domain image archives, downloads candidates,
applies license + dimension gates, dedups via perceptual hash, and
produces 03_assets/asset_manifest.json.

Phases (port of the maritime pipeline with business-domain recipes):
  Phase 0 — catalog whatever's already on disk (operator drops, prior
            runs)
  Phase 1 — Wikimedia Commons + Wikipedia searches for {company},
            {founder}, key products / events
  Phase 2 — SearXNG `site:...` for business-domain image archives
  Phase 5 — generic-stash entries (VLM-captioned operator library)

Map renderer (maritime's Phase 4) is intentionally absent — business
stories don't need a single geographic incident location.

Phase 1b external APIs (Smithsonian, Europeana, Pixabay) and Phase 3
upscaling are scaffolded behind config gates but skipped if either
the keys aren't present or the helper modules aren't installed.

Inputs:  episode.incident, source_inventory.json
Outputs: 03_assets/pd/*.png  +  03_assets/asset_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..browser import Browser, safe_filename
from ..config import load_config
from ..generic_stash import ensure_stash_manifest
from ..state import find_episode_workspace
from ..vlm import VLM
from .. import wikimedia

logger = logging.getLogger("hermes.stage.s05")


ACCEPTABLE_LICENSES = {
    "pd", "pd-old-70", "pd-old", "pd-usgov", "pd-1923",
    "cc0", "cc-by", "cc-by-1.0", "cc-by-2.0", "cc-by-2.5",
    "cc-by-3.0", "cc-by-4.0",
    "no known copyright restrictions",
}
REJECTED_LICENSES = {"cc-by-nc", "cc-by-sa", "cc-by-nd", "all rights reserved"}


@dataclass
class _PD:
    license: str
    license_url: str | None
    attribution: str | None
    url: str
    source_page: str | None


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    incident = episode["incident"]
    company = incident["company_name"]
    founder = (incident.get("founder_or_protagonist") or "").strip()

    # ---- master switch ----
    # When asset_hunt.enabled is false (config default for this
    # pipeline), S05 is a near-instant no-op: it writes an empty
    # asset_manifest.json and returns. S08 then routes every beat to
    # FLUX, which is the right call for comic-book-styled episodes
    # where PD photo assets would clash with the locked visual style.
    if not bool((cfg.raw.get("asset_hunt") or {}).get("enabled", True)):
        logger.info("S05: asset_hunt.enabled=false — skipping PD hunt; "
                    "all beats will route to FLUX")
        empty = {
            "company_name": company,
            "founder_or_protagonist": founder,
            "pd_assets": [],
            "skipped_reason": "asset_hunt.enabled=false in config.yaml",
        }
        (ws / "03_assets").mkdir(parents=True, exist_ok=True)
        (ws / "03_assets" / "asset_manifest.json").write_text(
            json.dumps(empty, indent=2)
        )
        return None

    browser = Browser()

    # Try to enrich query terms from the fact ledger (key products,
    # competitors, locations).
    extra_terms = _extra_search_terms(ws)

    pd_dir = ws / "03_assets" / "pd"
    pd_dir.mkdir(parents=True, exist_ok=True)
    quarantine = ws / "03_assets" / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    seen_hashes: set[str] = set()
    idx = 0

    caption_enabled = bool(cfg.image_qa.get("caption_pd_assets", True))
    vlm: VLM | None = VLM() if caption_enabled else None

    def _caption(path: Path, fallback: str) -> tuple[str, str | None]:
        if vlm is None:
            return fallback, None
        cap = vlm.caption_image(path, incident_name=company)
        if cap:
            return cap, cap
        return fallback, None

    # ---- Phase 0: catalog whatever's already on disk ----
    for f in sorted(pd_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        meta = _probe_image(f)
        if meta is None:
            continue
        phash = _phash_or_md5(f)
        if phash in seen_hashes:
            continue
        seen_hashes.add(phash)
        idx += 1
        title = f.stem.replace("_", " ").replace("-", " ")
        description, caption = _caption(f, fallback=title)
        manifest.append({
            "id": f"pd_{idx:04d}",
            "local_path": str(f.relative_to(ws)),
            "original_url": None,
            "source_page": None,
            "license": "operator-supplied",
            "attribution_required": False,
            "attribution_text": None,
            "width": meta[0],
            "height": meta[1],
            "direct_use_eligible": min(meta) >= 800,
            "title": title,
            "description": description,
            "caption": caption,
            "source": "phase0_disk",
        })
    logger.info("S05 Phase 0: %d existing assets cataloged", len(manifest))

    # ---- Phase 1: Wikimedia Commons via direct API ----
    # The earlier SearXNG-with-site:commons approach returned HTML page
    # URLs that we could not download as images. The Commons MediaWiki
    # API gives us direct file URLs + structured license metadata in
    # one call. This is the highest-yield source for known companies.
    phase1_terms = [t for t in [company, founder, *extra_terms] if t]
    for term in phase1_terms:
        if idx >= 50:
            break
        try:
            hits = wikimedia.search(term, limit=20)
        except Exception as e:
            logger.warning("wikimedia search failed for %r: %s", term, e)
            continue
        logger.info("S05 Phase 1: wikimedia returned %d hits for %r", len(hits), term)
        for h in hits:
            if idx >= 50:
                break
            if not wikimedia.is_license_acceptable(h.license_short):
                logger.debug("S05 Phase 1: license rejected %s: %s",
                             h.title, h.license_short)
                continue
            ok = _ingest_wikimedia(
                hit=h, browser=browser, pd_dir=pd_dir, ws=ws,
                manifest=manifest, seen=seen_hashes, idx=idx + 1,
                caption=_caption, fallback=term,
            )
            if ok:
                idx += 1
    logger.info("S05 after Phase 1: %d assets", len(manifest))

    # ---- Phase 2: SearXNG image-search mode (categories=images) ----
    # Image-category SearXNG results carry the direct image URL in
    # `img_src`, which browser.search() promotes to `url` for us. No
    # `.png/.jpg` extension filter required.
    image_search_terms = [t for t in [company, founder, *extra_terms[:3]] if t]
    for term in image_search_terms:
        if idx >= 80:
            break
        try:
            results = browser.search(term, n_results=20, categories="images")
        except Exception as e:
            logger.warning("S05 Phase 2: image search failed for %r: %s", term, e)
            continue
        logger.info("S05 Phase 2: image search returned %d hits for %r",
                    len(results), term)
        for r in results:
            if idx >= 80:
                break
            ok, _ = _try_ingest(
                r=r, browser=browser, pd_dir=pd_dir, ws=ws,
                manifest=manifest, seen=seen_hashes, idx=idx + 1,
                source_tag="phase2_image_search",
                caption=_caption, fallback=term,
            )
            if ok:
                idx += 1
    logger.info("S05 after Phase 2: %d assets", len(manifest))

    # ---- Phase 5: generic stash ----
    if cfg.generic_stash.get("enabled", True):
        try:
            stash_entries = ensure_stash_manifest(vlm=vlm)
        except Exception as e:
            logger.warning("generic_stash failed: %s", e)
            stash_entries = []
        # Re-key stash entries into the per-episode manifest namespace
        for s in stash_entries:
            idx += 1
            s["id"] = f"pd_{idx:04d}"
            manifest.append(s)
        logger.info("S05 Phase 5: +%d generic-stash entries", len(stash_entries))

    # ---- write manifest ----
    out = {
        "company_name": company,
        "founder_or_protagonist": founder,
        "pd_assets": manifest,
    }
    (ws / "03_assets" / "asset_manifest.json").write_text(json.dumps(out, indent=2))

    min_pd = cfg.quality_gates.get("min_pd_assets", 3)
    real = [m for m in manifest if not m.get("is_generic_stash")]
    if len(real) < min_pd:
        logger.warning(
            "S05: only %d incident-specific PD asset(s) (target %d) — "
            "S08 will route more beats to FLUX.",
            len(real), min_pd,
        )
    logger.info("S05 complete: %d assets total (%d incident-specific, %d stash)",
                len(manifest), len(real), len(manifest) - len(real))
    return None


# ---------------- helpers ----------------

def _extra_search_terms(ws: Path) -> list[str]:
    """Pull a few extra terms from the fact ledger to broaden Phase 1."""
    out: list[str] = []
    ledger_path = ws / "01_factcheck" / "fact_ledger.json"
    if not ledger_path.exists():
        return out
    try:
        ledger = json.loads(ledger_path.read_text())
    except Exception:
        return out
    seen = set()
    for claim in (ledger.get("claims") or []):
        ft = claim.get("fact_type")
        st = (claim.get("canonical_statement") or "").strip()
        if ft not in ("product_launch", "competitor", "acquisition", "ipo_event"):
            continue
        # Pull out capitalized noun phrases as candidate search terms.
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})\b", st):
            term = m.group(1)
            low = term.lower()
            if low not in seen and len(term) >= 4:
                seen.add(low)
                out.append(term)
        if len(out) >= 8:
            break
    return out[:8]


def _try_ingest(*, r, browser, pd_dir: Path, ws: Path,
                manifest: list[dict], seen: set[str], idx: int,
                source_tag: str, caption, fallback: str) -> tuple[bool, dict | None]:
    """Best-effort download → size gate → dedup → caption → manifest
    append. Returns (ok, entry-or-None).

    Used by Phase 2 image-search results. We trust SearXNG's
    `img_src` to point at a real image; the size gate after download
    rejects anything that turns out not to be an image (HTML error
    pages, missing extensions, 0-byte responses).
    """
    url = r.url
    if not url:
        return False, None

    # Preserve the source extension for the saved file when possible,
    # so PIL doesn't have to guess at the format.
    ext = _ext_from_url(url) or "jpg"
    dest = pd_dir / safe_filename(f"{source_tag}_{idx:04d}.{ext}", max_len=80)
    ok = browser.download(url, dest, timeout=45)
    if not ok or not dest.exists() or dest.stat().st_size < 5000:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False, None

    meta = _probe_image(dest)
    if meta is None or min(meta) < 200:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False, None

    phash = _phash_or_md5(dest)
    if phash in seen:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False, None
    seen.add(phash)

    title = r.title or fallback
    description, vlm_caption = caption(dest, fallback=title)
    entry = {
        "id": f"pd_{idx:04d}",
        "local_path": str(dest.relative_to(ws)),
        "original_url": url,
        "source_page": url,
        "license": "unknown",
        "attribution_required": True,
        "attribution_text": None,
        "width": meta[0],
        "height": meta[1],
        "direct_use_eligible": min(meta) >= 800,
        "title": title,
        "description": description,
        "caption": vlm_caption,
        "source": source_tag,
    }
    manifest.append(entry)
    logger.info("ingested %s (%dx%d) <- %s", entry["id"],
                meta[0], meta[1], source_tag)
    return True, entry


def _ingest_wikimedia(*, hit, browser, pd_dir: Path, ws: Path,
                      manifest: list[dict], seen: set[str], idx: int,
                      caption, fallback: str) -> bool:
    """Download a Wikimedia Commons hit (CommonsImage), apply the
    size + dedup gates, and add it to the manifest with the rich
    license attribution already extracted from the Commons API."""
    url = hit.url
    if not url:
        return False

    # Pre-filter on the API-reported dimensions so we don't download
    # tiny thumbnails.
    if hit.width and hit.height and min(hit.width, hit.height) < 300:
        logger.debug("S05 Phase 1: skipping %s (too small: %dx%d)",
                     hit.title, hit.width, hit.height)
        return False
    # Skip SVG / non-raster files — Ken Burns + ffmpeg expect PNG/JPEG.
    if hit.mime and "svg" in hit.mime.lower():
        return False

    ext = _ext_from_url(url) or _ext_from_mime(hit.mime) or "jpg"
    safe_title = safe_filename(hit.title.replace("File:", ""), max_len=60)
    dest = pd_dir / safe_filename(
        f"phase1_wikimedia_{idx:04d}_{safe_title}.{ext}", max_len=120,
    )
    ok = browser.download(url, dest, timeout=60)
    if not ok or not dest.exists() or dest.stat().st_size < 5000:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    meta = _probe_image(dest)
    if meta is None or min(meta) < 200:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    phash = _phash_or_md5(dest)
    if phash in seen:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    seen.add(phash)

    title = hit.title.replace("File:", "").rsplit(".", 1)[0].replace("_", " ")
    description, vlm_caption = caption(dest, fallback=title)

    attribution_text = None
    if hit.attribution_required:
        # Operator must credit. Build a readable attribution line.
        bits = [hit.artist or "Unknown author",
                f"({hit.license_short})" if hit.license_short else "",
                f"via Wikimedia Commons"]
        attribution_text = " ".join(b for b in bits if b)

    entry = {
        "id": f"pd_{idx:04d}",
        "local_path": str(dest.relative_to(ws)),
        "original_url": url,
        "source_page": hit.description_url,
        "license": hit.license_short or "unknown",
        "license_url": hit.license_url or None,
        "attribution_required": hit.attribution_required,
        "attribution_text": attribution_text,
        "width": meta[0],
        "height": meta[1],
        "direct_use_eligible": min(meta) >= 800,
        "title": title,
        "description": description,
        "caption": vlm_caption,
        "source": "phase1_wikimedia",
    }
    manifest.append(entry)
    logger.info("ingested %s (%dx%d, %s) <- wikimedia",
                entry["id"], meta[0], meta[1], hit.license_short or "?")
    return True


_EXT_RE = re.compile(r"\.([A-Za-z0-9]{2,5})(?:$|[?#])")


def _ext_from_url(url: str) -> str | None:
    """Pull the file extension off a URL path, if present and sane."""
    m = _EXT_RE.search(url or "")
    if not m:
        return None
    ext = m.group(1).lower()
    if ext in {"png", "jpg", "jpeg", "webp", "tif", "tiff"}:
        return ext
    return None


def _ext_from_mime(mime: str) -> str | None:
    if not mime:
        return None
    mime = mime.lower()
    if "jpeg" in mime or "jpg" in mime:
        return "jpg"
    if "png" in mime:
        return "png"
    if "webp" in mime:
        return "webp"
    if "tiff" in mime:
        return "tif"
    return None


def _probe_image(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def _phash_or_md5(path: Path) -> str:
    """Perceptual hash if imagehash is installed (preferred — defeats
    near-duplicates); fallback to MD5 of bytes."""
    try:
        import imagehash
        with Image.open(path) as img:
            return f"phash:{imagehash.phash(img)}"
    except Exception:
        with path.open("rb") as f:
            return f"md5:{hashlib.md5(f.read()).hexdigest()}"
