"""YouTube upload packaging and explicit publish step.

Two-stage operator flow:
  1. build_upload_package(episode_id)
     Copies final.mp4, captions, thumbnail, shorts, and metadata into
     06_metadata/youtube_upload_package/ for review.
  2. upload_approved_package(episode_id)
     Uploads package contents to YouTube only after the operator calls
     the explicit approval CLI.

Uploads use the same OAuth token file as youtube_analytics.py, but the
token must include the youtube.upload scope. Re-run --authorize-youtube
after this module ships if an older readonly token exists.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .state import (
    file_lock,
    find_episode,
    find_episode_workspace,
    load_queue,
    save_queue,
)
from .youtube_analytics import _load_credentials

logger = logging.getLogger("hermes.youtube_upload")


@dataclass
class UploadPackageResult:
    package_dir: Path
    manifest_path: Path
    long_form_ready: bool
    shorts_ready: int


def build_upload_package(episode_id: str) -> UploadPackageResult:
    """Build a reviewable upload package for one episode.

    The package is self-contained and intentionally duplicated from
    pipeline output paths so the operator can inspect exactly what
    will be uploaded.
    """
    cfg = load_config()
    queue = load_queue()
    episode = find_episode(queue, episode_id)
    if not episode:
        raise ValueError(f"no such episode: {episode_id}")
    ws = find_episode_workspace(episode_id)
    if not ws:
        raise ValueError(f"no episode workspace for {episode_id}")

    final_mp4 = ws / "05_video" / "final.mp4"
    if not final_mp4.exists():
        raise FileNotFoundError(f"missing final video: {final_mp4}")

    package_dir = ws / "06_metadata" / "youtube_upload_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    long_dir = package_dir / "long_form"
    shorts_dir = package_dir / "shorts"
    long_dir.mkdir(parents=True, exist_ok=True)
    shorts_dir.mkdir(parents=True, exist_ok=True)

    incident = episode.get("incident") or {}
    titles = _load_titles(ws)
    title = _pick_title(titles, incident)
    tags = _build_tags(incident, cfg.upload.get("tags") or [])
    description = _build_long_description(ws, incident)
    thumb = _pick_thumbnail(ws)

    long_video = long_dir / "final.mp4"
    shutil.copy2(final_mp4, long_video)
    _copy_if_exists(ws / "05_video" / "captions.srt", long_dir / "captions.srt")
    _copy_if_exists(ws / "05_video" / "captions.vtt", long_dir / "captions.vtt")
    thumbnail_rel = None
    if thumb:
        out_thumb = long_dir / thumb.name
        shutil.copy2(thumb, out_thumb)
        thumbnail_rel = _rel(out_thumb, package_dir)

    long_metadata = {
        "kind": "long_form",
        "title": title,
        "description_path": "long_form/description.txt",
        "tags": tags,
        "category_id": str(cfg.upload.get("category_id", "27")),
        "privacy_status": str(cfg.upload.get("default_privacy_status", "private")),
        "default_language": str(cfg.upload.get("default_language", "en")),
        "made_for_kids": bool(cfg.upload.get("made_for_kids", False)),
        "video_path": "long_form/final.mp4",
        "caption_path": (
            "long_form/captions.srt"
            if (long_dir / "captions.srt").exists() else None
        ),
        "thumbnail_path": thumbnail_rel,
        "playlist_id": (cfg.upload.get("long_form_playlist_id") or ""),
    }
    (long_dir / "description.txt").write_text(description)
    (long_dir / "metadata.json").write_text(json.dumps(long_metadata, indent=2))

    short_entries = _build_shorts_package(
        ws=ws,
        package_dir=package_dir,
        shorts_dir=shorts_dir,
        incident=incident,
        cfg_upload=cfg.upload,
        base_tags=tags,
    )

    manifest = {
        "schema_version": 1,
        "episode_id": episode_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved": False,
        "uploaded": False,
        "package_dir": str(package_dir),
        "long_form": long_metadata,
        "shorts": short_entries,
        "contents": {
            "long_form_video": long_metadata["video_path"],
            "long_form_caption": long_metadata["caption_path"],
            "long_form_thumbnail": long_metadata["thumbnail_path"],
            "shorts_count": len(short_entries),
        },
        "review_notes": [
            "Review long_form/metadata.json and long_form/description.txt.",
            "Review long_form thumbnail choice.",
            "Review each shorts/short_NN/metadata.json.",
            "Upload only with --upload-youtube-package EP_ID --approve-youtube-upload.",
        ],
    }
    manifest_path = package_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _write_summary(package_dir, manifest)

    return UploadPackageResult(
        package_dir=package_dir,
        manifest_path=manifest_path,
        long_form_ready=True,
        shorts_ready=len(short_entries),
    )


def upload_approved_package(
    episode_id: str,
    *,
    approve: bool,
    privacy_status: str | None = None,
) -> dict[str, Any]:
    """Upload a package to YouTube.

    `approve` must be True. This is deliberately awkward so upload
    cannot happen from a casual typo.
    """
    if not approve:
        raise ValueError(
            "refusing upload without explicit --approve-youtube-upload"
        )
    cfg = load_config()
    ws = find_episode_workspace(episode_id)
    if not ws:
        raise ValueError(f"no episode workspace for {episode_id}")
    package_dir = ws / "06_metadata" / "youtube_upload_package"
    manifest_path = package_dir / "package_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"upload package missing: {manifest_path}; build it first"
        )
    manifest = json.loads(manifest_path.read_text())

    yt = _youtube_client()
    if yt is None:
        raise RuntimeError(
            "YouTube client unavailable. Install optional deps and run "
            "--authorize-youtube again so the token has youtube.upload."
        )

    privacy = privacy_status or manifest["long_form"].get(
        "privacy_status"
    ) or cfg.upload.get("default_privacy_status", "private")

    results: dict[str, Any] = {
        "episode_id": episode_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "long_form": None,
        "shorts": [],
    }

    long_result = _upload_one_video(
        yt,
        package_dir=package_dir,
        metadata=manifest["long_form"],
        privacy_status=privacy,
    )
    results["long_form"] = long_result

    long_video_id = long_result.get("video_id")
    if long_video_id and manifest["long_form"].get("thumbnail_path"):
        _set_thumbnail(
            yt,
            video_id=long_video_id,
            path=package_dir / manifest["long_form"]["thumbnail_path"],
        )
    if long_video_id and manifest["long_form"].get("caption_path"):
        _insert_caption(
            yt,
            video_id=long_video_id,
            path=package_dir / manifest["long_form"]["caption_path"],
            language=manifest["long_form"].get("default_language", "en"),
            name="English",
        )
    if long_video_id and manifest["long_form"].get("playlist_id"):
        _add_to_playlist(
            yt,
            video_id=long_video_id,
            playlist_id=manifest["long_form"]["playlist_id"],
        )

    for short in manifest.get("shorts") or []:
        short_result = _upload_one_video(
            yt,
            package_dir=package_dir,
            metadata=short,
            privacy_status=privacy,
        )
        if short_result.get("video_id") and short.get("playlist_id"):
            _add_to_playlist(
                yt,
                video_id=short_result["video_id"],
                playlist_id=short["playlist_id"],
            )
        results["shorts"].append(short_result)

    manifest["approved"] = True
    manifest["uploaded"] = True
    manifest["upload_results"] = results
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (package_dir / "upload_results.json").write_text(json.dumps(results, indent=2))

    _bind_uploaded_ids(episode_id, results)
    return results


def _youtube_client():
    try:
        from googleapiclient.discovery import build
    except ImportError:
        logger.warning("googleapiclient missing; install with pip -e '.[youtube]'")
        return None
    creds = _load_credentials()
    if creds is None:
        return None
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _upload_one_video(
    yt,
    *,
    package_dir: Path,
    metadata: dict[str, Any],
    privacy_status: str,
) -> dict[str, Any]:
    from googleapiclient.http import MediaFileUpload

    video_path = package_dir / metadata["video_path"]
    description_path = package_dir / metadata["description_path"]
    description = description_path.read_text() if description_path.exists() else ""
    body = {
        "snippet": {
            "title": metadata["title"][:100],
            "description": description[:5000],
            "tags": metadata.get("tags") or [],
            "categoryId": str(metadata.get("category_id") or "27"),
            "defaultLanguage": metadata.get("default_language") or "en",
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": bool(metadata.get("made_for_kids", False)),
        },
    }
    media = MediaFileUpload(
        str(video_path), mimetype="video/mp4", chunksize=-1, resumable=True
    )
    request = yt.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )
    response = None
    while response is None:
        _status, response = request.next_chunk()
    video_id = response["id"]
    logger.info("YouTube upload complete: %s -> %s", video_path.name, video_id)
    return {
        "kind": metadata.get("kind"),
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "title": metadata.get("title"),
        "privacy_status": privacy_status,
    }


def _set_thumbnail(yt, *, video_id: str, path: Path) -> None:
    from googleapiclient.http import MediaFileUpload

    if not path.exists():
        logger.warning("thumbnail missing for %s: %s", video_id, path)
        return
    media = MediaFileUpload(str(path), mimetype="image/jpeg", resumable=False)
    yt.thumbnails().set(videoId=video_id, media_body=media).execute()


def _insert_caption(
    yt,
    *,
    video_id: str,
    path: Path,
    language: str,
    name: str,
) -> None:
    from googleapiclient.http import MediaFileUpload

    if not path.exists():
        logger.warning("caption missing for %s: %s", video_id, path)
        return
    body = {
        "snippet": {
            "videoId": video_id,
            "language": language,
            "name": name,
            "isDraft": False,
        }
    }
    media = MediaFileUpload(
        str(path), mimetype="application/octet-stream", resumable=False
    )
    yt.captions().insert(part="snippet", body=body, media_body=media).execute()


def _add_to_playlist(yt, *, video_id: str, playlist_id: str) -> None:
    body = {
        "snippet": {
            "playlistId": playlist_id,
            "resourceId": {
                "kind": "youtube#video",
                "videoId": video_id,
            },
        }
    }
    yt.playlistItems().insert(part="snippet", body=body).execute()


def _bind_uploaded_ids(episode_id: str, results: dict[str, Any]) -> None:
    cfg = load_config()
    lock_path = cfg.state_dir / "locks" / "orchestrator.lock"
    stale = cfg.orchestrator["stale_lock_hours"] * 3600
    with file_lock(lock_path, stale_seconds=stale):
        queue = load_queue()
        ep = find_episode(queue, episode_id)
        if ep is None:
            return
        long_form = results.get("long_form") or {}
        if long_form.get("video_id"):
            ep["youtube_video_id"] = long_form["video_id"]
            ep["published_at"] = results["uploaded_at"]
        ep["youtube_shorts_video_ids"] = [
            s["video_id"] for s in results.get("shorts") or []
            if s.get("video_id")
        ]
        save_queue(queue)


def _load_titles(ws: Path) -> list[dict[str, Any]]:
    path = ws / "06_metadata" / "titles.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("variants") or []
    except Exception:
        return []


def _pick_title(titles: list[dict[str, Any]], incident: dict[str, Any]) -> str:
    for t in titles:
        text = (t.get("text") or "").strip()
        if text:
            return text[:100]
    company = incident.get("company_name") or "Business Story"
    return f"The Story of {company}"[:100]


def _build_long_description(ws: Path, incident: dict[str, Any]) -> str:
    company = incident.get("company_name") or "this company"
    pitch = incident.get("one_line_pitch") or incident.get("conflict") or ""
    description = [
        f"The business story of {company}.",
        "",
    ]
    if pitch:
        description.extend([pitch.strip(), ""])
    description.extend([
        "Chapters and sources are generated by the Business Stories pipeline.",
        "",
    ])
    attr = ws / "06_metadata" / "license_attributions.txt"
    if attr.exists():
        description.extend(["Sources & Attributions:", attr.read_text().strip(), ""])
    description.append("#BusinessHistory #BrandStory #Documentary")
    return "\n".join(description).strip() + "\n"


def _build_short_description(incident: dict[str, Any], hashtags: list[str]) -> str:
    company = incident.get("company_name") or "Business History"
    tags = " ".join(f"#{h.lstrip('#')}" for h in hashtags)
    return f"{company} business history short.\n\n{tags}\n"


def _build_tags(incident: dict[str, Any], defaults: list[str]) -> list[str]:
    raw = list(defaults)
    for key in ("company_name", "story_kind", "founder_or_protagonist"):
        val = (incident.get(key) or "").strip()
        if val:
            raw.append(val)
    out: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for tag in raw:
        clean = " ".join(str(tag).replace("#", "").split())[:50]
        key = clean.lower()
        if not clean or key in seen:
            continue
        if total_chars + len(clean) > 450:
            break
        seen.add(key)
        out.append(clean)
        total_chars += len(clean)
    return out


def _pick_thumbnail(ws: Path) -> Path | None:
    thumb_dir = ws / "05_video" / "thumbnails"
    preferred = [
        "thumb_founder_closeup.jpg",
        "thumb_split_frame.jpg",
        "thumb_big_number.jpg",
    ]
    for name in preferred:
        p = thumb_dir / name
        if p.exists():
            return p
    thumbs = sorted(thumb_dir.glob("*.jpg"))
    return thumbs[0] if thumbs else None


def _build_shorts_package(
    *,
    ws: Path,
    package_dir: Path,
    shorts_dir: Path,
    incident: dict[str, Any],
    cfg_upload: dict[str, Any],
    base_tags: list[str],
) -> list[dict[str, Any]]:
    src_dir = ws / "05_video" / "shorts"
    manifest_path = src_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return []
    hashtags = cfg_upload.get("shorts_hashtags") or ["Shorts"]
    entries: list[dict[str, Any]] = []
    for item in manifest.get("shorts") or []:
        src = Path(item.get("path") or "")
        if not src.is_absolute():
            src = ws / src
        if not src.exists():
            continue
        rank = int(item.get("rank") or (len(entries) + 1))
        out_dir = shorts_dir / f"short_{rank:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"short_{rank:02d}.mp4"
        shutil.copy2(src, out_video)
        title = (item.get("title_hint") or incident.get("company_name") or "Short")
        if "#Shorts" not in title:
            title = f"{title} #Shorts"
        metadata = {
            "kind": "short",
            "rank": rank,
            "title": title[:100],
            "description_path": f"shorts/short_{rank:02d}/description.txt",
            "tags": _build_tags(incident, base_tags + hashtags),
            "category_id": str(cfg_upload.get("category_id", "27")),
            "privacy_status": str(
                cfg_upload.get("default_privacy_status", "private")
            ),
            "default_language": str(cfg_upload.get("default_language", "en")),
            "made_for_kids": bool(cfg_upload.get("made_for_kids", False)),
            "video_path": f"shorts/short_{rank:02d}/short_{rank:02d}.mp4",
            "playlist_id": (cfg_upload.get("shorts_playlist_id") or ""),
            "source_window": {
                "start_seconds": item.get("start_seconds"),
                "end_seconds": item.get("end_seconds"),
                "reasoning": item.get("reasoning"),
            },
        }
        (out_dir / "description.txt").write_text(
            _build_short_description(incident, hashtags)
        )
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        entries.append(metadata)
    return entries


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _write_summary(package_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"YouTube Upload Package: {manifest['episode_id']}",
        "",
        "Long form:",
        f"- video: {manifest['long_form']['video_path']}",
        f"- title: {manifest['long_form']['title']}",
        f"- thumbnail: {manifest['long_form'].get('thumbnail_path')}",
        f"- captions: {manifest['long_form'].get('caption_path')}",
        "",
        f"Shorts: {len(manifest.get('shorts') or [])}",
    ]
    for s in manifest.get("shorts") or []:
        lines.append(f"- {s['video_path']}: {s['title']}")
    lines.extend([
        "",
        "Review package_manifest.json before upload.",
        "Upload command:",
        (
            "python -m pipeline.hermes_orchestrator "
            f"--upload-youtube-package {manifest['episode_id']} "
            "--approve-youtube-upload"
        ),
    ])
    (package_dir / "PACKAGE_SUMMARY.md").write_text("\n".join(lines) + "\n")
