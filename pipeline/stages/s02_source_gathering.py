"""S02 — Source Gathering.

Issues a fixed set of SearXNG recipes scoped to OPEN business-press
domains, plus government / court / archive primary sources. Hits to
known paywall domains are NOT fetched at the canonical URL; instead
they're re-routed through the Wayback Machine, and if Wayback returns
no usable body the source is persisted as a title-only stub for
downstream awareness.

Inputs:  episode.incident (with company_name, year_anchor, founder_or_protagonist)
Outputs: 00_research/source_inventory.json  +  per-source extracted text
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from urllib.parse import urlparse

from ..browser import Browser, extract_publisher, wayback_url
from ..config import load_config
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s02")


# ---------------- domain buckets ----------------

# Authoritative + fully open. Always fetched at canonical URL.
OPEN_TIER1_DOMAINS = {
    # US securities / govt
    "sec.gov", "efts.sec.gov", "investor.gov",
    "govinfo.gov", "federalregister.gov", "congress.gov",
    "ftc.gov", "cftc.gov", "consumerfinance.gov",
    "justice.gov", "usao.justice.gov",
    "bls.gov", "census.gov", "bea.gov", "data.gov",
    # Court records
    "courtlistener.com", "pacer.uscourts.gov", "justia.com",
    "law.cornell.edu",
    # International registries
    "companieshouse.gov.uk",
    # Wiki family (anchor reference, dedup hub)
    "en.wikipedia.org", "simple.wikipedia.org",
    "en.wikiquote.org", "en.wikisource.org",
    "commons.wikimedia.org",
    # Archive
    "archive.org", "web.archive.org",
    # Academia / case studies
    "scholar.google.com", "hbs.edu", "library.harvard.edu",
    "mit.edu", "stanford.edu",
}

# Substantially-open press + investigative + free aggregators.
OPEN_TIER2_DOMAINS = {
    # Public broadcasters / wire (mostly free body)
    "apnews.com", "npr.org", "pbs.org", "bbc.com", "bbc.co.uk",
    "aljazeera.com", "dw.com", "france24.com", "cbc.ca", "abc.net.au",
    "reuters.com",
    # Free public business / general
    "theguardian.com", "cnbc.com", "marketwatch.com",
    "yahoo.com", "finance.yahoo.com", "seekingalpha.com",
    # Tech press (mostly free)
    "techcrunch.com", "theverge.com", "arstechnica.com",
    "wired.com", "engadget.com", "gizmodo.com",
    "protocol.com", "restofworld.org",
    # Investigative + longform
    "propublica.org", "vox.com", "axios.com", "qz.com",
    "theatlantic.com", "slate.com", "newyorker.com",
    "vanityfair.com",
    # Aggregators (treated as link-discovery, not authority)
    "news.ycombinator.com", "lobste.rs",
    "reddit.com",
    # Founder voice
    "medium.com", "substack.com",
    # Funding / company data
    "crunchbase.com",
    # Podcast transcripts
    "acquired.fm",
}

# Known paywalls. Body NEVER fetched at canonical URL; re-routed to
# Wayback or persisted as title-only.
PAYWALL_DOMAINS = {
    "nytimes.com", "wsj.com", "ft.com", "bloomberg.com", "economist.com",
    "businessinsider.com", "forbes.com", "fortune.com", "barrons.com",
    "nikkei.com", "lemonde.fr",
    "theinformation.com", "pitchbook.com",
    "washingtonpost.com",
    "latimes.com", "chicagotribune.com", "bostonglobe.com",
}


# ---------------- search recipes ----------------

# Each recipe: (query_template, default_tier_hint, target_subset)
#   target_subset ∈ {"open_tier1", "open_tier2", "all"}
# The query is .format()'d with {company}, {year}, {founder}, {story_kind}.
BUSINESS_RECIPES: list[tuple[str, int, str]] = [
    # --- Primary-source filings + court records (open_tier1) ---
    ('"{company}" 10-K OR S-1 OR DEF 14A OR 8-K site:sec.gov', 1, "open_tier1"),
    ('"{company}" complaint OR indictment OR settlement '
     'site:courtlistener.com OR site:justice.gov OR site:sec.gov', 1, "open_tier1"),
    ('"{company}" site:companieshouse.gov.uk', 1, "open_tier1"),

    # --- Wikipedia anchor (open_tier1) ---
    ('"{company}" history founding site:en.wikipedia.org', 1, "open_tier1"),
    ('"{founder}" biography site:en.wikipedia.org', 1, "open_tier1"),

    # --- Wayback first-pass (open_tier1) ---
    ('"{company}" "{year}" site:web.archive.org', 1, "open_tier1"),
    ('"{founder}" "{year}" site:web.archive.org', 1, "open_tier1"),

    # --- Open business / public press (open_tier2) ---
    ('"{company}" {story_kind} (founder OR origin OR pivot OR collapse) '
     'site:apnews.com OR site:reuters.com OR site:npr.org OR site:bbc.com '
     'OR site:cnbc.com OR site:theguardian.com', 2, "open_tier2"),

    # --- Investigative longform (open_tier2) ---
    ('"{company}" site:propublica.org OR site:theatlantic.com '
     'OR site:newyorker.com OR site:vanityfair.com', 2, "open_tier2"),

    # --- Tech press for tech-origin stories (open_tier2) ---
    ('"{company}" founding site:techcrunch.com OR site:arstechnica.com '
     'OR site:wired.com OR site:theverge.com', 2, "open_tier2"),

    # --- Podcast transcripts (open_tier2) ---
    ('"{company}" transcript site:acquired.fm OR site:npr.org', 2, "open_tier2"),

    # --- Founder voice (open_tier2) ---
    ('"{founder}" interview OR essay OR memoir site:medium.com '
     'OR site:substack.com OR site:archive.org', 2, "open_tier2"),

    # --- Open-ended sweep (all, paywall hits routed to Wayback) ---
    ('"{company}" "{founder}" {story_kind}', 3, "all"),
    ('"{company}" "{year}" history OR background OR profile', 3, "all"),
]


# ---------------- main entry ----------------

def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    browser = Browser()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace; S01 must run first"

    incident = episode["incident"]
    company = incident["company_name"]
    founder = incident.get("founder_or_protagonist") or ""
    year = incident.get("year_anchor")
    story_kind = incident.get("story_kind") or ""

    raw_dir = ws / "00_research" / "raw"
    ext_dir = ws / "00_research" / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    seen_urls: set[str] = set()
    inventory: list[dict] = []
    src_idx = 0

    max_sources = int(cfg.quality_gates.get("max_sources", 100))
    n_results = int(cfg.search.get("results_per_query", 30))

    logger.info("S02: company=%r year=%s founder=%r story=%s",
                company, year, founder, story_kind)
    logger.info("S02: %d recipes", len(BUSINESS_RECIPES))

    for recipe_idx, (query_template, tier_hint, subset) in enumerate(BUSINESS_RECIPES, 1):
        try:
            query = query_template.format(
                company=company,
                year=year if year is not None else "",
                founder=founder,
                story_kind=story_kind,
            )
        except KeyError as e:
            logger.warning("recipe %d format failed: %s", recipe_idx, e)
            continue

        logger.info("[recipe %d/%d %s] %s",
                    recipe_idx, len(BUSINESS_RECIPES), subset, query[:120])
        try:
            results = browser.search(query, n_results=n_results)
        except Exception as e:
            logger.warning("search failed for %r: %s", query, e)
            continue

        for r in results:
            if r.url in seen_urls or len(inventory) >= max_sources:
                continue
            pub = (r.publisher or extract_publisher(r.url)).lower()
            bucket = _classify_domain(pub)

            # Subset gate — drop hits that don't belong to the recipe's
            # intended bucket. "all" passes everything.
            if subset == "open_tier1" and bucket != "open_tier1":
                continue
            if subset == "open_tier2" and bucket not in ("open_tier1", "open_tier2"):
                continue
            # subset == "all": no bucket filter

            seen_urls.add(r.url)

            # Route paywalls through Wayback, never the canonical URL.
            if bucket == "paywall":
                entry = _ingest_paywall(
                    r=r, pub=pub, browser=browser,
                    incident=incident, raw_dir=raw_dir, ext_dir=ext_dir,
                    src_idx=src_idx + 1, ws=ws,
                )
            else:
                entry = _ingest_open(
                    r=r, pub=pub, browser=browser, bucket=bucket,
                    incident=incident, raw_dir=raw_dir, ext_dir=ext_dir,
                    src_idx=src_idx + 1, ws=ws, tier_hint=tier_hint,
                )

            if entry is None:
                continue
            src_idx += 1
            inventory.append(entry)
            logger.info("captured %s (tier=%s, %d words) %s",
                        entry["id"], entry["tier"], entry["word_count"], pub)

        if len(inventory) >= max_sources:
            break

    # --- quality gates ---
    min_sources = cfg.quality_gates["min_sources"]
    min_tier1 = cfg.quality_gates.get("min_tier1_sources", 1)
    tier1_count = sum(
        1 for e in inventory
        if e["tier"] in ("open_tier1",)
    )
    if len(inventory) < min_sources:
        return (f"only {len(inventory)} sources gathered "
                f"(need {min_sources}); company may be too obscure or too recent")
    if tier1_count < min_tier1:
        return (f"only {tier1_count} open_tier1 sources "
                f"(need {min_tier1}); no SEC filing, court record, or "
                f"Wikipedia anchor found for {company}")

    (ws / "00_research" / "source_inventory.json").write_text(
        json.dumps({"sources": inventory}, indent=2)
    )
    logger.info("S02 complete: %d sources (%d open_tier1)",
                len(inventory), tier1_count)
    return None


# ---------------- domain classification ----------------

def _classify_domain(publisher: str) -> str:
    """Return 'open_tier1' | 'open_tier2' | 'paywall' | 'unknown'."""
    p = (publisher or "").lower().lstrip(".")
    # Strip leading "www."
    if p.startswith("www."):
        p = p[4:]

    # Exact / suffix match against the OPEN_TIER1 set first (most
    # specific). subdomain matches: "efts.sec.gov" -> sec.gov.
    for d in OPEN_TIER1_DOMAINS:
        if p == d or p.endswith("." + d):
            return "open_tier1"
    for d in PAYWALL_DOMAINS:
        if p == d or p.endswith("." + d):
            return "paywall"
    for d in OPEN_TIER2_DOMAINS:
        if p == d or p.endswith("." + d):
            return "open_tier2"
    # .gov / .mil / .edu fallbacks
    if p.endswith(".gov") or p.endswith(".mil"):
        return "open_tier1"
    if p.endswith(".edu"):
        return "open_tier2"
    return "unknown"


# ---------------- ingest paths ----------------

def _ingest_open(
    *, r, pub, browser, bucket, incident, raw_dir, ext_dir,
    src_idx, ws, tier_hint,
) -> dict | None:
    """Fetch the canonical URL of an open source, gate on relevance,
    persist raw + extracted, return inventory entry."""
    try:
        fetched = browser.fetch(r.url, timeout=30)
    except Exception as e:
        logger.warning("fetch failed for %s: %s", r.url, e)
        return None
    if fetched.status != 200 or not fetched.text.strip():
        return None

    extracted = _extract_text(fetched.text)
    extracted = " ".join(extracted.split()[:15000])
    if not _is_relevant(extracted, incident):
        logger.info("dropped off-topic: %s", r.url[:80])
        return None

    sid = f"src_{src_idx:03d}"
    url_hash = hashlib.sha256(r.url.encode()).hexdigest()[:12]
    raw_path = raw_dir / f"{sid}_{url_hash}.html"
    try:
        raw_path.write_text(fetched.text)
    except Exception:
        pass
    ext_path = ext_dir / f"{sid}.txt"
    ext_path.write_text(extracted)

    # Tier: if the bucket is open_tier1 use that, else fall back to
    # bucket; finally fall back to the recipe hint.
    if bucket == "open_tier1":
        tier = "open_tier1"
    elif bucket == "open_tier2":
        tier = "open_tier2"
    else:
        tier = "open_tier2" if tier_hint <= 2 else "open_tier3"

    return {
        "id": sid,
        "url": r.url,
        "original_url": r.url,
        "publisher": pub,
        "tier": tier,
        "title": r.title,
        "byline": None,
        "date": None,
        "local_path": str(ext_path.relative_to(ws)),
        "raw_path": str(raw_path.relative_to(ws)),
        "word_count": len(extracted.split()),
    }


def _ingest_paywall(
    *, r, pub, browser, incident, raw_dir, ext_dir, src_idx, ws,
) -> dict | None:
    """Skip canonical URL; try Wayback. If Wayback yields no usable
    body, persist a title-only stub."""
    wb = wayback_url(r.url, when="2*")
    logger.info("paywall %s → wayback %s", pub, wb[:120])
    try:
        fetched = browser.fetch(wb, timeout=45)
    except Exception as e:
        logger.warning("wayback fetch failed for %s: %s", r.url, e)
        fetched = None

    if fetched and fetched.status == 200 and fetched.text.strip():
        extracted = _extract_text(fetched.text)
        extracted = " ".join(extracted.split()[:15000])
        if _is_relevant(extracted, incident) and len(extracted.split()) >= 100:
            sid = f"src_{src_idx:03d}"
            url_hash = hashlib.sha256(r.url.encode()).hexdigest()[:12]
            raw_path = raw_dir / f"{sid}_{url_hash}.html"
            try:
                raw_path.write_text(fetched.text)
            except Exception:
                pass
            ext_path = ext_dir / f"{sid}.txt"
            ext_path.write_text(extracted)
            return {
                "id": sid,
                "url": wb,
                "original_url": r.url,
                "publisher": pub,
                "tier": "paywall_wayback",
                "title": r.title,
                "byline": None,
                "date": None,
                "local_path": str(ext_path.relative_to(ws)),
                "raw_path": str(raw_path.relative_to(ws)),
                "word_count": len(extracted.split()),
            }

    # Title-only stub: keep the headline + snippet for downstream.
    title = (r.title or "").strip()
    snippet = (r.snippet or "").strip()
    if not (title or snippet):
        return None
    sid = f"src_{src_idx:03d}"
    ext_path = ext_dir / f"{sid}.txt"
    body = f"# PAYWALL — title and snippet only\n\nTitle: {title}\nSnippet: {snippet}\nOriginal URL: {r.url}\n"
    ext_path.write_text(body)
    return {
        "id": sid,
        "url": r.url,
        "original_url": r.url,
        "publisher": pub,
        "tier": "paywall_title_only",
        "title": title,
        "byline": None,
        "date": None,
        "local_path": str(ext_path.relative_to(ws)),
        "raw_path": None,
        "word_count": len(body.split()),
    }


# ---------------- text + relevance helpers ----------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_BUSINESS_STOPWORDS = {
    "inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "corporation",
    "company", "co", "co.", "ag", "sa", "se", "plc", "gmbh",
    "ceo", "cfo", "cto", "coo", "ipo", "founder", "founders",
    "the", "of", "and", "a", "an", "for",
    "business", "history", "founding", "founded", "story",
}


def _is_relevant(text: str, incident: dict) -> bool:
    """Two cheap signals:
      - the company name appears verbatim (case-insensitive); or
      - at least 2 distinguishing tokens from name AND the year both
        appear in the text; or
      - founder name appears AND at least one company-name token.

    Designed to filter out hits where SearXNG fell back to broader
    matches (e.g. unrelated companies with one shared word).
    """
    t = text.lower()
    name = (incident.get("company_name") or "").lower().strip()
    founder = (incident.get("founder_or_protagonist") or "").lower().strip()
    if name and name in t:
        return True
    name_tokens = [
        w for w in re.findall(r"[a-z0-9]+", name)
        if len(w) > 2 and w not in _BUSINESS_STOPWORDS
    ]
    year = str(incident.get("year_anchor") or "").strip()
    matched = sum(1 for w in set(name_tokens) if w in t)
    if year and year in t and matched >= 1:
        return True
    if matched >= 2:
        return True
    if founder and founder in t and matched >= 1:
        return True
    return False


def _extract_text(html: str) -> str:
    """Cheap HTML→text. Drops scripts/nav/header/footer/aside chrome."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return _WS_RE.sub(" ", soup.get_text(" ")).strip()
    except Exception:
        return _WS_RE.sub(" ", _HTML_TAG_RE.sub(" ", html)).strip()
