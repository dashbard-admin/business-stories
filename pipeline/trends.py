"""Topic-validation signals for S01.

S01 picks "interesting" business stories via the writer LLM. Interesting
to an LLM ≠ what audiences search for. Without an external sanity
check the pipeline will happily burn 3-4 hours rendering an episode
on a topic that has no search demand at all, or that has so much
existing video coverage your upload gets buried on page 5.

This module provides the pre-commit demand check. S01 calls it
*after* schema/dedup/recency gates but *before* persisting the topic,
so the trend-validation rejection feeds back into the prompt as a
retry hint ("the LLM previously suggested X but it failed the
saturation gate — pick something else").

Two signals, both via SearXNG (the same backend S02 uses for source
gathering — no new API key, no new infrastructure):

  1. YouTube/video saturation:
       SearXNG `categories=videos` aggregates YouTube, Vimeo,
       Dailymotion, Bing Video. We query `"<company>" documentary`
       and count distinct URLs. Below min → too obscure (no audience
       demand). Above max → too saturated (cold-start audience can't
       find us).

  2. Recent news activity:
       SearXNG `categories=news` aggregates Google News, Bing News,
       Reuters, AP. We query the company name and count distinct
       URLs. This is an ADVISORY signal only — high counts mean
       "trending right now" (recent bankruptcy filing, breaking
       scandal, acquisition news) which is a goldmine, but low
       counts don't disqualify (a 1998 origin story might have no
       fresh news and still be a great pick).

The validation result returns the signals dict so S01 can attach it
to the persisted incident record, giving the operator post-hoc
visibility into why each topic was accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hermes.trends")


@dataclass
class ValidationResult:
    ok: bool
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------

def youtube_video_count(query: str, browser, *, n_results: int = 30) -> int:
    """Number of distinct video URLs SearXNG returns for `query` under
    the `videos` category. Returns -1 on adapter failure (caller treats
    -1 as "unknown, don't reject" — we'd rather let a topic through
    than block the whole pipeline on a transient SearXNG outage).

    Caveat: the count is an upper bound on YouTube saturation, not a
    direct YouTube result count — SearXNG includes Vimeo, Dailymotion,
    Bing Video etc. in this category. Empirically the spread is
    YouTube-dominated for English business topics, so it works as a
    saturation proxy.
    """
    try:
        results = browser.search(query, n_results=n_results, categories="videos")
        urls = {r.url for r in results if r.url}
        return len(urls)
    except Exception as e:
        logger.warning("youtube_video_count failed for %r: %s", query, e)
        return -1


def recent_news_count(query: str, browser, *, n_results: int = 20) -> int:
    """Number of distinct news URLs SearXNG returns for `query` under
    the `news` category. Returns -1 on adapter failure.

    SearXNG's news engines surface recent items by default; we don't
    do client-side date filtering because per-result published-at
    fields are unreliable across engine backends (some give RFC3339,
    some give human strings, some give nothing). The count is a
    "is this topic currently in the news cycle?" indicator.
    """
    try:
        results = browser.search(query, n_results=n_results, categories="news")
        urls = {r.url for r in results if r.url}
        return len(urls)
    except Exception as e:
        logger.warning("recent_news_count failed for %r: %s", query, e)
        return -1


# ----------------------------------------------------------------------
# Composite gate
# ----------------------------------------------------------------------

def validate_candidate(
    candidate: dict[str, Any],
    cfg_validation: dict[str, Any],
    browser,
) -> ValidationResult:
    """Pre-commit validation gate. Called once per candidate, after the
    cheap schema/dedup/recency gates have already passed.

    Returns ok=False with a human-readable reason if the candidate
    fails the saturation window. The signals dict is always populated
    (even on failure) so S01 can log it.
    """
    name = (candidate.get("company_name") or "").strip()
    if not name:
        return ValidationResult(False, "missing company_name", {})

    if not cfg_validation.get("enabled", True):
        return ValidationResult(True, "validation disabled", {})

    # Saturation probe — phrase the query the way an actual viewer
    # might. "<Company> documentary" is the closest match to the
    # search intent of a person who'd watch our channel.
    yt_query = f"{name} documentary"
    yt_count = youtube_video_count(yt_query, browser)

    # Trending probe — bare company name catches recent filings,
    # acquisition announcements, scandal breaks.
    news_count = recent_news_count(name, browser)

    signals: dict[str, Any] = {
        "youtube_query": yt_query,
        "youtube_count": yt_count,
        "news_count": news_count,
    }

    min_yt = int(cfg_validation.get("min_youtube_results", 3))
    max_yt = int(cfg_validation.get("max_youtube_results", 50))

    # Sentinel: search backend unreachable. Don't reject — degrade
    # gracefully. The other S01 gates still apply and the operator
    # will see this in the logs.
    if yt_count < 0:
        signals["validation_note"] = "youtube_count unavailable; gate skipped"
        return ValidationResult(True, "search service unreachable", signals)

    if yt_count < min_yt:
        return ValidationResult(
            False,
            f"obscure: only {yt_count} videos for {yt_query!r} "
            f"(min={min_yt}) — no audience demand",
            signals,
        )

    if yt_count > max_yt:
        return ValidationResult(
            False,
            f"saturated: {yt_count} videos for {yt_query!r} "
            f"(max={max_yt}) — channel can't break in",
            signals,
        )

    return ValidationResult(True, "passed", signals)


# ----------------------------------------------------------------------
# Non-US ratio enforcement helper (called by S01 before the LLM call)
# ----------------------------------------------------------------------

def non_us_required(
    queue: dict[str, Any],
    *,
    ratio: float,
    lookback: int,
) -> bool:
    """True iff the rolling window of recent picks is too US-heavy and
    the next pick must be non-US to keep the channel's geographic
    spread on target.

    The rolling-window key `countries` is appended to in S01 on every
    successful topic commit. We look at the last `lookback` entries
    and require non-US iff non-US count is below `round(N * ratio)`.

    Behaviour at the cold start (fewer than 2 entries): always return
    False — too early to enforce. This lets the first one or two
    episodes go with whatever the LLM picks (probably US), then the
    enforcement kicks in.
    """
    rw = queue.get("rolling_window") or {}
    countries = (rw.get("countries") or [])[-lookback:]
    if len(countries) < 2:
        return False
    non_us = sum(1 for c in countries if (c or "").strip().upper() not in ("US", "USA", ""))
    target = max(1, int(round(len(countries) * float(ratio))))
    return non_us < target
