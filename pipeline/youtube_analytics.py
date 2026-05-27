"""YouTube Analytics writeback (Batch E 2026-05-27).

Pulls per-episode performance metrics (views, CTR, AVD, retention
curve, traffic sources) from the YouTube Data + Analytics APIs and
writes them back to the pipeline so future S1/S6/S8 prompts can learn
from what worked.

Setup (one-time):
  1. Create an OAuth client in Google Cloud Console (Desktop app type).
     Enable the YouTube Data API v3 and YouTube Analytics API on the
     same project.
  2. Put the client ID + secret in .env:
       YOUTUBE_OAUTH_CLIENT_ID=...
       YOUTUBE_OAUTH_CLIENT_SECRET=...
  3. Run:
       python -m pipeline.hermes_orchestrator --authorize-youtube
     This opens a browser, you approve, paste the auth code back.
     The refresh token gets cached at state/youtube_oauth_token.json
     (gitignored — it's a per-machine credential).

After each upload, bind the episode to its uploaded video id:
  python -m pipeline.hermes_orchestrator --set-video-id EP_017 <yt_video_id>

Then run the writeback (manual per Q-E1 confirmed):
  python -m pipeline.hermes_orchestrator --analyse-performance

In mock mode this returns canned metrics so the pipeline can
exercise the feedback path without a real OAuth dance.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config

logger = logging.getLogger("hermes.youtube_analytics")


OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
]


@dataclass
class EpisodePerformance:
    """Snapshot of a single episode's performance, pulled from
    YouTube Analytics API. Persisted to
    episodes/EP_NNN_<slug>/06_metadata/youtube_performance.json AND
    summarised into state/performance_history.json."""
    video_id: str
    fetched_at: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    ctr: float = 0.0           # impressions-clickthrough rate, 0..1
    avd_seconds: float = 0.0   # average view duration
    avg_view_pct: float = 0.0  # AVD / video length, 0..1
    # Retention curve: list of (relative_position 0..1, retention 0..1).
    # YouTube returns 100 buckets — we keep them all for analysis.
    retention_curve: list[dict] = field(default_factory=list)
    peak_drop_at_seconds: float = 0.0   # biggest single-bucket drop
    top_traffic_sources: list[dict] = field(default_factory=list)
    impressions: int = 0


# ----------------------------------------------------------------------
# OAuth installed-app flow
# ----------------------------------------------------------------------

def _token_path() -> Path:
    return load_config().state_dir / "youtube_oauth_token.json"


def _client_secrets() -> dict[str, Any] | None:
    """Construct an installed-app client_secrets JSON dict from env."""
    cid = (os.environ.get("YOUTUBE_OAUTH_CLIENT_ID") or "").strip()
    csec = (os.environ.get("YOUTUBE_OAUTH_CLIENT_SECRET") or "").strip()
    if not cid or not csec:
        return None
    return {
        "installed": {
            "client_id": cid,
            "client_secret": csec,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def authorize_oauth() -> int:
    """One-time interactive flow. Saves the refresh token to
    state/youtube_oauth_token.json. Returns 0 on success."""
    secrets = _client_secrets()
    if not secrets:
        print("--authorize-youtube: YOUTUBE_OAUTH_CLIENT_ID + "
              "YOUTUBE_OAUTH_CLIENT_SECRET must be set in .env first.")
        return 2
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("--authorize-youtube: install the optional youtube deps:")
        print("    pip install -e '.[youtube]'")
        return 2

    flow = InstalledAppFlow.from_client_config(secrets, OAUTH_SCOPES)
    creds = flow.run_local_server(port=0)

    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(token_data, indent=2))
    print(f"--authorize-youtube: token cached to {path}")
    return 0


def _load_credentials():
    """Load refreshed credentials from the cached token file. Returns
    None when the token file is missing or the optional dep isn't
    installed."""
    path = _token_path()
    if not path.exists():
        logger.warning(
            "YouTube OAuth token not found at %s — run "
            "`--authorize-youtube` first",
            path,
        )
        return None
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        logger.warning("youtube optional deps not installed — run "
                       "`pip install -e '.[youtube]'`")
        return None
    data = json.loads(path.read_text())
    creds = Credentials.from_authorized_user_info(data, OAUTH_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Persist refreshed token.
        data["token"] = creds.token
        path.write_text(json.dumps(data, indent=2))
    return creds


# ----------------------------------------------------------------------
# Analytics fetcher
# ----------------------------------------------------------------------

class YouTubeAnalytics:
    """Per-episode performance fetcher. Lazy auth — the first
    fetch_episode() call triggers OAuth load."""

    def __init__(self):
        cfg = load_config()
        self._mock = cfg.mock_mode
        self._channel_id = (cfg.youtube_analytics.get("channel_id") or "")
        self._creds = None
        self._yt = None
        self._analytics = None

    def _ensure_clients(self) -> bool:
        if self._mock:
            return True
        if self._yt is not None:
            return True
        try:
            from googleapiclient.discovery import build
        except ImportError:
            logger.warning("googleapiclient missing — install with "
                           "`pip install -e '.[youtube]'`")
            return False
        creds = _load_credentials()
        if creds is None:
            return False
        self._creds = creds
        self._yt = build("youtube", "v3", credentials=creds,
                         cache_discovery=False)
        self._analytics = build("youtubeAnalytics", "v2",
                                credentials=creds, cache_discovery=False)
        return True

    def fetch_episode(self, video_id: str) -> EpisodePerformance | None:
        """Pull metrics for one published video. Returns None on
        unavailable / API failure (caller skips that episode)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if self._mock:
            return _mock_performance(video_id, now)

        if not self._ensure_clients():
            return None

        try:
            # Basic stats from Data API v3.
            yt_resp = self._yt.videos().list(
                part="statistics,contentDetails",
                id=video_id,
            ).execute()
            items = yt_resp.get("items") or []
            if not items:
                logger.warning("fetch_episode: %s not found", video_id)
                return None
            stats = items[0].get("statistics", {})
            content = items[0].get("contentDetails", {})

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))
            duration_iso = content.get("duration", "PT0S")
            duration_seconds = _iso_duration_to_seconds(duration_iso)

            # CTR + AVD + impressions from Analytics API.
            metrics = (
                "views,impressions,averageViewDuration,"
                "averageViewPercentage,annotationClickThroughRate,"
                "cardClickRate,impressionsCtr"
            )
            an_resp = self._analytics.reports().query(
                ids=f"channel==MINE" if not self._channel_id else
                    f"channel=={self._channel_id}",
                startDate="2020-01-01",
                endDate=datetime.now(timezone.utc).date().isoformat(),
                metrics=metrics,
                filters=f"video=={video_id}",
            ).execute()

            row = (an_resp.get("rows") or [[0] * 7])[0]
            headers = [h["name"] for h in an_resp.get("columnHeaders", [])]
            by_name = dict(zip(headers, row))

            impressions = int(by_name.get("impressions", 0) or 0)
            ctr = float(by_name.get("impressionsCtr", 0.0) or 0.0) / 100.0
            avd_seconds = float(by_name.get("averageViewDuration", 0) or 0)
            avg_view_pct = float(
                by_name.get("averageViewPercentage", 0.0) or 0.0
            ) / 100.0

            # Retention curve via the audienceWatchRatio dimension.
            retention_curve = self._fetch_retention(video_id)
            peak_drop = _compute_peak_drop_seconds(
                retention_curve, duration_seconds,
            )

            # Top traffic sources.
            tsrc = self._fetch_traffic_sources(video_id)

            return EpisodePerformance(
                video_id=video_id,
                fetched_at=now,
                views=views,
                likes=likes,
                comments=comments,
                ctr=ctr,
                avd_seconds=avd_seconds,
                avg_view_pct=avg_view_pct,
                retention_curve=retention_curve,
                peak_drop_at_seconds=peak_drop,
                top_traffic_sources=tsrc,
                impressions=impressions,
            )
        except Exception as e:
            logger.warning("fetch_episode %s failed: %s", video_id, e)
            return None

    def _fetch_retention(self, video_id: str) -> list[dict]:
        if self._mock or not self._analytics:
            return []
        try:
            r = self._analytics.reports().query(
                ids="channel==MINE",
                startDate="2020-01-01",
                endDate=datetime.now(timezone.utc).date().isoformat(),
                metrics="audienceWatchRatio",
                dimensions="elapsedVideoTimeRatio",
                filters=f"video=={video_id}",
                sort="elapsedVideoTimeRatio",
            ).execute()
            rows = r.get("rows") or []
            return [
                {"position": float(rr[0]), "retention": float(rr[1])}
                for rr in rows
            ]
        except Exception as e:
            logger.warning("retention fetch failed: %s", e)
            return []

    def _fetch_traffic_sources(self, video_id: str) -> list[dict]:
        if self._mock or not self._analytics:
            return []
        try:
            r = self._analytics.reports().query(
                ids="channel==MINE",
                startDate="2020-01-01",
                endDate=datetime.now(timezone.utc).date().isoformat(),
                metrics="views",
                dimensions="insightTrafficSourceType",
                filters=f"video=={video_id}",
                sort="-views",
                maxResults=10,
            ).execute()
            rows = r.get("rows") or []
            return [
                {"source": rr[0], "views": int(rr[1])}
                for rr in rows
            ]
        except Exception as e:
            logger.warning("traffic-source fetch failed: %s", e)
            return []


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _iso_duration_to_seconds(iso: str) -> float:
    """Parse PT#H#M#S → seconds. Best-effort; returns 0 on failure."""
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0.0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + s


def _compute_peak_drop_seconds(
    curve: list[dict], duration_seconds: float,
) -> float:
    """Find the position (in seconds) of the largest single-bucket
    drop in the retention curve. Returns 0 if curve is empty."""
    if not curve or len(curve) < 2 or duration_seconds <= 0:
        return 0.0
    biggest_drop = 0.0
    biggest_at = 0.0
    for i in range(1, len(curve)):
        prev = curve[i - 1]["retention"]
        cur = curve[i]["retention"]
        d = prev - cur
        if d > biggest_drop:
            biggest_drop = d
            biggest_at = float(curve[i]["position"]) * duration_seconds
    return round(biggest_at, 1)


def _mock_performance(video_id: str, now: str) -> EpisodePerformance:
    return EpisodePerformance(
        video_id=video_id,
        fetched_at=now,
        views=12000,
        likes=420,
        comments=37,
        ctr=0.062,
        avd_seconds=540.0,
        avg_view_pct=0.49,
        retention_curve=[
            {"position": i / 20.0, "retention": max(0.1, 1.0 - i * 0.045)}
            for i in range(21)
        ],
        peak_drop_at_seconds=220.0,
        top_traffic_sources=[
            {"source": "YT_SEARCH", "views": 5400},
            {"source": "EXTERNAL", "views": 3200},
            {"source": "YT_OTHER_PAGE", "views": 2100},
        ],
        impressions=193000,
    )


def to_serialisable(perf: EpisodePerformance) -> dict[str, Any]:
    """Convert EpisodePerformance to a JSON-safe dict for persistence."""
    return {
        "video_id": perf.video_id,
        "fetched_at": perf.fetched_at,
        "views": perf.views,
        "likes": perf.likes,
        "comments": perf.comments,
        "ctr": round(perf.ctr, 5),
        "avd_seconds": round(perf.avd_seconds, 2),
        "avg_view_pct": round(perf.avg_view_pct, 4),
        "retention_curve": perf.retention_curve,
        "peak_drop_at_seconds": perf.peak_drop_at_seconds,
        "top_traffic_sources": perf.top_traffic_sources,
        "impressions": perf.impressions,
    }
