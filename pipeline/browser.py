"""Browser adapter.

Three operations:
  - search(query, n_results=10): returns list of SearchResult
  - fetch(url): returns FetchResult (status, content_type, text, bytes_len)
  - download(url, dest_path): saves a binary asset

Real backend is the local SearXNG instance (JSON output) for search,
plus plain `requests` for fetch/download. Endpoint and tuning live
under the `search:` block in config.yaml.

In mock_mode, returns canned business-story content so the pipeline
runs offline.

A standalone helper `wayback_url(original_url)` is also exported; S2
uses it to re-issue paywalled URLs against the Wayback Machine.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote as _urlquote

import requests

from .config import load_config

logger = logging.getLogger("hermes.browser")


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    publisher: str = ""


@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    text: str
    bytes_len: int


class Browser:
    def __init__(self):
        cfg = load_config()
        self.mock_mode = cfg.mock_mode
        self._search_cfg = cfg.search
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self._search_cfg.get(
                "user_agent", "BusinessStoriesPipeline/0.1 (+research)"
            ),
        })

    # ------------------ public API ------------------

    def search(self, query: str, n_results: int = 10,
               *, categories: str = "") -> list[SearchResult]:
        """Generic SearXNG search.

        Set `categories="images"` to route the query through SearXNG's
        image-search engines (Google Images, Bing Images, etc.) instead
        of the default general-web engines. Image results carry the
        DIRECT image URL in the `url` field rather than the HTML page
        URL — S05 Phase 2 relies on this to actually download images.
        """
        if self.mock_mode:
            return self._mock_search(query, n_results)
        return self._invoke_search(query, n_results, categories=categories)

    def fetch(self, url: str, timeout: int = 30) -> FetchResult:
        if self.mock_mode:
            return self._mock_fetch(url)
        return self._invoke_fetch(url, timeout)

    def download(self, url: str, dest: Path, timeout: int = 60) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.mock_mode:
            return self._mock_download(url, dest)
        try:
            return self._invoke_download(url, dest, timeout)
        except Exception as e:
            logger.warning("download failed for %s: %s", url, e)
            return False

    # ------------------ implementation hooks ------------------

    def _invoke_search(
        self, query: str, n: int, *, categories: str = "",
    ) -> list[SearchResult]:
        s = self._search_cfg
        if s.get("backend") != "searxng":
            raise NotImplementedError(
                f"search backend {s.get('backend')!r} not implemented"
            )
        url = s["searxng_url"].rstrip("/") + "/search"
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "safesearch": s.get("safesearch", 0),
        }
        engines = (s.get("engines") or "").strip()
        if engines and not categories:
            # Don't pin engines when the caller requested a category
            # (categories trumps engines in SearXNG anyway, and engine
            # names from the default set may not be in the image set).
            params["engines"] = engines
        if categories:
            params["categories"] = categories

        r = self._session.get(
            url, params=params,
            timeout=s.get("request_timeout_seconds", 30),
        )
        r.raise_for_status()
        data = r.json()
        out: list[SearchResult] = []
        for item in (data.get("results") or [])[:n]:
            # Image-search results put the canonical image URL in
            # `img_src` and the source HTML page in `url`. We want
            # the image URL when categories=images, the page URL
            # otherwise.
            if categories == "images":
                link = item.get("img_src") or item.get("url") or ""
            else:
                link = item.get("url") or ""
            if not link:
                continue
            out.append(SearchResult(
                url=link,
                title=(item.get("title") or "").strip(),
                snippet=(item.get("content") or "").strip(),
                publisher=extract_publisher(link),
            ))
        return out

    def _invoke_fetch(self, url: str, timeout: int) -> FetchResult:
        r = self._session.get(url, timeout=timeout, allow_redirects=True)
        content_type = r.headers.get("Content-Type", "")
        ct_low = content_type.lower()
        is_textual = any(t in ct_low for t in ("text/", "html", "json", "xml"))
        text = r.text if is_textual else ""
        return FetchResult(
            url=str(r.url),
            status=r.status_code,
            content_type=content_type,
            text=text,
            bytes_len=len(r.content),
        )

    def _invoke_download(self, url: str, dest: Path, timeout: int) -> bool:
        with self._session.get(
            url, stream=True, timeout=timeout, allow_redirects=True
        ) as r:
            if r.status_code != 200:
                return False
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True

    # ------------------ mocks ------------------

    def _mock_search(self, query: str, n: int) -> list[SearchResult]:
        seed = int(hashlib.sha256(query.encode()).hexdigest()[:6], 16)
        domains = ["en.wikipedia.org", "sec.gov", "apnews.com",
                   "propublica.org", "archive.org"]
        results = []
        for i in range(min(n, 5)):
            d = domains[i % len(domains)]
            results.append(SearchResult(
                url=f"https://{d}/mock/{seed}/{i}",
                title=f"Mock result {i} for: {query[:60]}",
                snippet="Mock business-story snippet for pipeline testing. "
                        "Replace browser._invoke_search to get real data.",
                publisher=d,
            ))
        return results

    def _mock_fetch(self, url: str) -> FetchResult:
        body = (
            f"# Mock content for {url}\n\n"
            "Acme Corp was incorporated in Delaware on March 12, 1998 by Jordan Lee, "
            "who had spent the previous five years at a much larger competitor. "
            "The company's first prototype was assembled in a converted garage in Palo Alto, "
            "California. By 2001 Acme had raised a Series A of $4.2 million from Sequoia. "
            "In 2003 the company pivoted from enterprise software to a consumer subscription model, "
            "a move that the founder later called 'the only decision that mattered.' "
            "By 2007 Acme had three hundred employees and a public listing on the NASDAQ. "
            "In 2014, after a series of executive departures and a securities-fraud investigation, "
            "the company was delisted. Jordan Lee resigned in 2015. "
            "The Securities and Exchange Commission case (SEC v. Acme Corp, 2016) "
            "concluded with a $42 million settlement.\n\n"
            "(End of mock content.)"
        )
        return FetchResult(url=url, status=200, content_type="text/html",
                           text=body, bytes_len=len(body))

    def _mock_download(self, url: str, dest: Path) -> bool:
        from PIL import Image
        img = Image.new("RGB", (1600, 1200), color=(40, 50, 70))
        img.save(dest, "PNG")
        return True


# ------------------ helpers ------------------

def extract_publisher(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return m.group(1) if m else url


def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", s)
    return s[:max_len]


def wayback_url(original_url: str, when: str = "2*") -> str:
    """Build a Wayback Machine URL for a given source URL.

    `when` is a partial year token (e.g. "2024" or "2*" for any 2-prefix
    year). The web.archive.org `web/<token>/<url>` form returns the
    closest snapshot whose timestamp begins with `token` — handy for
    routing paywalled URLs through cached pre-paywall snapshots.
    """
    return f"https://web.archive.org/web/{when}/{_urlquote(original_url, safe=':/?&=%')}"
