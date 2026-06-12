"""Wikimedia Commons direct API adapter.

The earlier S05 PD search route ran every query through SearXNG with a
`site:commons.wikimedia.org` filter, then expected direct image URLs
back. Both pieces are unreliable:

- SearXNG / its upstream engines do not consistently honor `site:`, so
  Commons pages often were not even in the result set.
- General-search results return HTML PAGE urls (e.g.
  `commons.wikimedia.org/wiki/File:Foo.svg`), not the actual image
  file URL. Phase 1 was filtering them all out at
  `_looks_like_image_url`.

This adapter hits the Commons MediaWiki API directly. It returns image
URLs together with structured license metadata (extmetadata), so the
caller does not have to scrape anything. No API key required.

API reference: https://commons.wikimedia.org/w/api.php
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger("hermes.wikimedia")

API_URL = "https://commons.wikimedia.org/w/api.php"
DEFAULT_UA = "BusinessStoriesPipeline/0.1 (+research; uses public domain media)"

# License-short-name fragments we accept. Commons uses values like
# "PD", "PD-old", "CC0", "CC BY 4.0", "CC BY-SA 3.0", "No restrictions".
ACCEPTABLE_LICENSE_FRAGMENTS = (
    "pd", "public domain", "cc0",
    "cc by", "cc-by",          # CC BY (any version), incl. CC BY-SA
    "no restrictions", "no known", "no copyright",
)

# Substrings that disqualify even if "CC BY" appears (NC = non-commercial,
# ND = no-derivatives are unsuitable for monetized video).
DISQUALIFYING_FRAGMENTS = (
    "nc", "non-commercial", "noncommercial",
    "nd", "no-derivatives", "noderivatives",
)


@dataclass
class CommonsImage:
    title: str                       # "File:Foo.jpg"
    url: str                         # full-resolution image URL
    description_url: str             # the human-readable Commons page
    width: int
    height: int
    mime: str
    license_short: str               # e.g. "PD-old-70", "CC BY-SA 4.0"
    license_url: str
    artist: str                      # plain-text artist credit
    credit: str
    attribution_required: bool


def search(
    query: str,
    *,
    limit: int = 20,
    user_agent: str = DEFAULT_UA,
    timeout: int = 30,
) -> list[CommonsImage]:
    """Search Commons for files matching `query` and return image
    metadata. Up to `limit` results."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    titles = _search_titles(session, query, limit=limit, timeout=timeout)
    if not titles:
        return []
    return _imageinfo(session, titles, timeout=timeout)


def is_license_acceptable(license_short: str) -> bool:
    """True if the license string indicates monetization-safe re-use."""
    norm = (license_short or "").lower().strip()
    if not norm:
        return False
    if any(d in norm for d in DISQUALIFYING_FRAGMENTS):
        return False
    return any(a in norm for a in ACCEPTABLE_LICENSE_FRAGMENTS)


# ------------------------------ internals ------------------------------

_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_RE.sub("", text or "").strip()


def _search_titles(
    session: requests.Session, query: str, *, limit: int, timeout: int,
) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,         # File: namespace only
        "srlimit": limit,
        "srprop": "snippet",
    }
    hits = _get_json(session, params=params, timeout=timeout).get("query", {}).get("search", []) or []
    return [h["title"] for h in hits if h.get("title", "").startswith("File:")]


def _imageinfo(
    session: requests.Session, titles: list[str], *, timeout: int,
) -> list[CommonsImage]:
    """Fetch imageinfo + extmetadata for up to 50 files in one call."""
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles[:50]),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1600,
    }
    pages = _get_json(session, params=params, timeout=timeout).get("query", {}).get("pages", {}) or {}

    out: list[CommonsImage] = []
    for page in pages.values():
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue
        info = imageinfo[0]
        ext = info.get("extmetadata") or {}

        license_short = (ext.get("LicenseShortName", {}).get("value") or "").strip()
        license_url = (ext.get("LicenseUrl", {}).get("value") or "").strip()
        artist = _strip_html(ext.get("Artist", {}).get("value") or "")
        credit = _strip_html(ext.get("Credit", {}).get("value") or "")

        attr_raw = ext.get("AttributionRequired", {}).get("value")
        attribution_required = (
            (isinstance(attr_raw, str) and attr_raw.lower() in ("true", "yes", "1"))
            or attr_raw is True
            or "cc by" in license_short.lower()
            or "cc-by" in license_short.lower()
        )

        mime = info.get("mime", "")
        url = info.get("url", "")
        if mime and "svg" in mime.lower():
            url = info.get("thumburl") or url

        out.append(CommonsImage(
            title=page.get("title", ""),
            url=url,
            description_url=info.get("descriptionurl", ""),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            mime=mime,
            license_short=license_short,
            license_url=license_url,
            artist=artist,
            credit=credit,
            attribution_required=bool(attribution_required),
        ))
    return out


def _get_json(
    session: requests.Session,
    *,
    params: dict,
    timeout: int,
    max_attempts: int = 4,
) -> dict:
    """GET Commons JSON with polite handling for transient 429s."""
    last_response: requests.Response | None = None
    for attempt in range(max_attempts):
        r = session.get(API_URL, params=params, timeout=timeout)
        last_response = r
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()

        if attempt >= max_attempts - 1:
            break

        retry_after = r.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
        except ValueError:
            delay = 1.5 * (attempt + 1)
        delay = max(1.0, min(delay, 8.0))
        logger.warning(
            "wikimedia rate limited for %r; retrying in %.1fs",
            params.get("srsearch") or params.get("titles") or "imageinfo",
            delay,
        )
        time.sleep(delay)

    assert last_response is not None
    last_response.raise_for_status()
    return {}
