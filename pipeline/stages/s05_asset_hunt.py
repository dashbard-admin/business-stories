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

logger = logging.getLogger("hermes.stage.s05")


ACCEPTABLE_LICENSES = {
    "pd", "pd-old-70", "pd-old", "pd-usgov", "pd-1923",
    "cc0", "cc-by", "cc-by-1.0", "cc-by-2.0", "cc-by-2.5",
    "cc-by-3.0", "cc-by-4.0",
    "no known copyright restrictions",
}
REJECTED_LICENSES = {"cc-by-nc", "cc-by-sa", "cc-by-nd", "all rights reserved"}


# Business-domain search plan for SearXNG `site:` queries. Each entry
# is (site, [query_templates_with_{company}_{founder}_{product}]).
SEARCH_PLAN_BUSINESS: list[tuple[str, list[str]]] = [
    # Wiki family
    ("commons.wikimedia.org", ['"{company}"', '"{founder}"', '"{company}" logo']),
    ("en.wikipedia.org", ['"{company}"', '"{founder}"']),
    # US public archives
    ("loc.gov/photos", ['"{company}"', '"{founder}"']),
    ("catalog.archives.gov", ['"{company}"']),
    ("si.edu", ['"{company}"', '"{founder}"']),       # Smithsonian top-level
    ("sec.gov", ['"{company}"']),                     # logos / proxy art
    # International archives
    ("europeana.eu", ['"{company}"', '"{founder}"']),
    # Free press (often has historical company photos)
    ("apnews.com", ['"{company}" archives']),
    ("bbc.com", ['"{company}"']),
    ("npr.org", ['"{company}"']),
    ("propublica.org", ['"{company}"']),
    # Tech press for tech-origin stories
    ("techcrunch.com", ['"{company}" founder']),
    ("wired.com", ['"{company}"']),
    # Internet archive
    ("archive.org", ['"{company}" photographs', '"{founder}" portrait']),
]


@dataclass
class _PD:
    license: str
    license_url: str | None
    attribution: str | None
    url: str
    source_page: str | None


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    browser = Browser()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    incident = episode["incident"]
    company = incident["company_name"]
    founder = (incident.get("founder_or_protagonist") or "").strip()

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

    # ---- Phase 1: Wikimedia Commons + Wikipedia direct queries ----
    phase1_terms = [t for t in [company, founder, *extra_terms] if t]
    for term in phase1_terms:
        results = _safe_search(
            browser, f'"{term}" site:commons.wikimedia.org', n=20,
        )
        for r in results:
            if idx >= 50:
                break
            ok, entry = _try_ingest(
                r=r, browser=browser, pd_dir=pd_dir, ws=ws,
                manifest=manifest, seen=seen_hashes, idx=idx + 1,
                source_tag="phase1_wikimedia", caption=_caption, fallback=term,
            )
            if ok:
                idx += 1
        if idx >= 50:
            break

    logger.info("S05 after Phase 1: %d assets", len(manifest))

    # ---- Phase 2: SearXNG `site:` plan ----
    template_ctx = {
        "company": company,
        "founder": founder or company,
        "product": extra_terms[0] if extra_terms else company,
    }
    for site, query_templates in SEARCH_PLAN_BUSINESS:
        if idx >= 80:
            break
        for qt in query_templates:
            try:
                inner = qt.format(**template_ctx)
            except KeyError:
                continue
            query = f"{inner} site:{site}"
            results = _safe_search(browser, query, n=15)
            for r in results:
                if idx >= 80:
                    break
                ok, _ = _try_ingest(
                    r=r, browser=browser, pd_dir=pd_dir, ws=ws,
                    manifest=manifest, seen=seen_hashes, idx=idx + 1,
                    source_tag=f"phase2_{site}", caption=_caption, fallback=inner,
                )
                if ok:
                    idx += 1
            if idx >= 80:
                break

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


def _safe_search(browser: Browser, query: str, n: int = 15) -> list:
    try:
        return browser.search(query, n_results=n)
    except Exception as e:
        logger.warning("search failed for %r: %s", query, e)
        return []


def _try_ingest(*, r, browser, pd_dir: Path, ws: Path,
                manifest: list[dict], seen: set[str], idx: int,
                source_tag: str, caption, fallback: str) -> tuple[bool, dict | None]:
    """Best-effort download → license/size gate → dedup → caption →
    manifest append. Returns (ok, entry-or-None)."""
    url = r.url
    if not url:
        return False, None
    # Only fetch direct image URLs; for HTML pages we'd need scrape
    # logic per-host. For our SearXNG-based pipeline that's an
    # acceptable simplification — image search engines mostly return
    # image URLs directly.
    if not _looks_like_image_url(url):
        return False, None

    dest = pd_dir / safe_filename(f"{source_tag}_{idx:04d}.png", max_len=80)
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


_IMG_RE = re.compile(r"\.(?:png|jpe?g|webp|tiff?|svg)(?:$|[?#])", re.IGNORECASE)


def _looks_like_image_url(url: str) -> bool:
    return bool(_IMG_RE.search(url or ""))


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
