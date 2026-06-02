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
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .llm import LLM
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
    caption_tracks_ready: int


@dataclass
class CaptionBackfillResult:
    package_dir: Path
    attempted: int
    uploaded: int
    skipped_existing: int
    warnings: list[dict[str, str]]


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
    max_tags = int(cfg.upload.get("max_tags", 3))
    tags = _build_tags(incident, cfg.upload.get("tags") or [], max_tags=max_tags)
    description = _build_long_description(ws, incident)
    thumb = _pick_thumbnail(ws)
    publish_at = _normalize_publish_at(cfg.upload.get("publish_at"))
    thumbnails_dir_rel = _copy_thumbnail_folder(ws, package_dir)

    long_video = long_dir / "final.mp4"
    shutil.copy2(final_mp4, long_video)
    _copy_if_exists(ws / "05_video" / "captions.srt", long_dir / "captions.srt")
    _copy_if_exists(ws / "05_video" / "captions.vtt", long_dir / "captions.vtt")
    long_caption_tracks = _build_caption_tracks(
        source_srt=ws / "05_video" / "captions.srt",
        out_dir=long_dir / "subtitles",
        rel_prefix="long_form/subtitles",
        cfg_upload=cfg.upload,
    )
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
        "publish_at": publish_at,
        "default_language": str(cfg.upload.get("default_language", "en")),
        "made_for_kids": bool(cfg.upload.get("made_for_kids", False)),
        "video_path": "long_form/final.mp4",
        "caption_path": (
            "long_form/captions.srt"
            if (long_dir / "captions.srt").exists() else None
        ),
        "caption_tracks": long_caption_tracks,
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
        max_tags=max_tags,
        publish_at=publish_at,
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
            "long_form_caption_tracks": len(long_caption_tracks),
            "long_form_thumbnail": long_metadata["thumbnail_path"],
            "thumbnails_dir": thumbnails_dir_rel,
            "shorts_count": len(short_entries),
            "shorts_caption_tracks": sum(
                len(s.get("caption_tracks") or []) for s in short_entries
            ),
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
        caption_tracks_ready=(
            len(long_caption_tracks)
            + sum(len(s.get("caption_tracks") or []) for s in short_entries)
        ),
    )


def upload_approved_package(
    episode_id: str,
    *,
    approve: bool,
    privacy_status: str | None = None,
    publish_at: str | None = None,
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
    publish_at_norm = _normalize_publish_at(
        publish_at or manifest["long_form"].get("publish_at")
        or cfg.upload.get("publish_at")
    )

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
        publish_at=publish_at_norm,
    )
    results["long_form"] = long_result

    long_video_id = long_result.get("video_id")
    post_upload_warnings: list[dict[str, str]] = []
    if long_video_id and manifest["long_form"].get("thumbnail_path"):
        _safe_post_upload_step(
            post_upload_warnings,
            "long_form_thumbnail",
            lambda: _set_thumbnail(
                yt,
                video_id=long_video_id,
                path=package_dir / manifest["long_form"]["thumbnail_path"],
            ),
        )
    if long_video_id:
        for track in _caption_tracks_for_metadata(manifest["long_form"]):
            _safe_post_upload_step(
                post_upload_warnings,
                f"long_form_caption_{track['language']}",
                lambda tr=track: _insert_caption(
                    yt,
                    video_id=long_video_id,
                    path=package_dir / tr["path"],
                    language=tr["language"],
                    name=tr["name"],
                ),
            )
    if long_video_id and manifest["long_form"].get("playlist_id"):
        _safe_post_upload_step(
            post_upload_warnings,
            "long_form_playlist",
            lambda: _add_to_playlist(
                yt,
                video_id=long_video_id,
                playlist_id=manifest["long_form"]["playlist_id"],
            ),
        )

    for short in manifest.get("shorts") or []:
        short_result = _upload_one_video(
            yt,
            package_dir=package_dir,
            metadata=short,
            privacy_status=privacy,
            publish_at=publish_at_norm,
        )
        if short_result.get("video_id") and short.get("playlist_id"):
            _safe_post_upload_step(
                post_upload_warnings,
                f"short_{short.get('rank')}_playlist",
                lambda sr=short_result, sh=short: _add_to_playlist(
                    yt,
                    video_id=sr["video_id"],
                    playlist_id=sh["playlist_id"],
                ),
            )
        if short_result.get("video_id"):
            for track in _caption_tracks_for_metadata(short):
                _safe_post_upload_step(
                    post_upload_warnings,
                    f"short_{short.get('rank')}_caption_{track['language']}",
                    lambda tr=track, sr=short_result: _insert_caption(
                        yt,
                        video_id=sr["video_id"],
                        path=package_dir / tr["path"],
                        language=tr["language"],
                        name=tr["name"],
                    ),
                )
        results["shorts"].append(short_result)
    results["post_upload_warnings"] = post_upload_warnings

    manifest["approved"] = True
    manifest["uploaded"] = True
    manifest["upload_results"] = results
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (package_dir / "upload_results.json").write_text(json.dumps(results, indent=2))

    _bind_uploaded_ids(episode_id, results)
    return results


def backfill_caption_tracks(
    episode_id: str,
    *,
    approve: bool,
    target: str = "all",
    languages: list[str] | None = None,
) -> CaptionBackfillResult:
    """Upload caption tracks from an existing package without videos.

    Intended for retrying caption failures after quota reset or after
    YouTube channel permissions/scopes have been fixed.
    """
    if not approve:
        raise ValueError(
            "refusing caption backfill without explicit "
            "--approve-youtube-upload"
        )
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
    upload_results = manifest.get("upload_results") or {}
    if not upload_results:
        raise ValueError(
            "package has no upload_results; upload videos once before "
            "caption backfill"
        )

    yt = _youtube_client()
    if yt is None:
        raise RuntimeError(
            "YouTube client unavailable. Install optional deps and run "
            "--authorize-youtube again so the token has caption scopes."
        )

    wanted_langs = {x.strip() for x in (languages or []) if x.strip()}
    warnings: list[dict[str, str]] = []
    attempted = 0
    uploaded = 0
    skipped_existing = 0

    for label, video_id, metadata in _caption_backfill_items(
        manifest, upload_results, target=target, episode_id=episode_id,
    ):
        if not video_id:
            warnings.append({
                "step": f"{label}_captions",
                "error": "missing uploaded video_id",
            })
            continue
        existing = _list_caption_languages(yt, video_id=video_id)
        for track in _caption_tracks_for_metadata(metadata):
            language = str(track.get("language") or "").strip()
            if wanted_langs and language not in wanted_langs:
                continue
            step = f"{label}_caption_{language}"
            if language in existing:
                skipped_existing += 1
                logger.info("caption backfill skipped existing: %s", step)
                continue
            attempted += 1
            try:
                _insert_caption(
                    yt,
                    video_id=video_id,
                    path=package_dir / track["path"],
                    language=language,
                    name=track["name"],
                )
                uploaded += 1
                logger.info("caption backfill uploaded: %s", step)
            except Exception as e:
                msg = str(e)
                warnings.append({"step": step, "error": msg})
                logger.warning("caption backfill failed: %s: %s", step, e)
                if "quotaExceeded" in msg:
                    logger.warning("caption backfill stopping: quotaExceeded")
                    run = _caption_backfill_run(
                        target=target,
                        languages=sorted(wanted_langs),
                        attempted=attempted,
                        uploaded=uploaded,
                        skipped_existing=skipped_existing,
                        warnings=warnings,
                    )
                    manifest.setdefault("caption_backfill_runs", []).append(run)
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    (package_dir / "caption_backfill_results.json").write_text(
                        json.dumps(run, indent=2)
                    )
                    return CaptionBackfillResult(
                        package_dir=package_dir,
                        attempted=attempted,
                        uploaded=uploaded,
                        skipped_existing=skipped_existing,
                        warnings=warnings,
                    )

    run = _caption_backfill_run(
        target=target,
        languages=sorted(wanted_langs),
        attempted=attempted,
        uploaded=uploaded,
        skipped_existing=skipped_existing,
        warnings=warnings,
    )
    manifest.setdefault("caption_backfill_runs", []).append(run)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (package_dir / "caption_backfill_results.json").write_text(
        json.dumps(run, indent=2)
    )
    return CaptionBackfillResult(
        package_dir=package_dir,
        attempted=attempted,
        uploaded=uploaded,
        skipped_existing=skipped_existing,
        warnings=warnings,
    )


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


def _caption_backfill_items(
    manifest: dict[str, Any],
    upload_results: dict[str, Any],
    *,
    target: str,
    episode_id: str,
) -> list[tuple[str, str | None, dict[str, Any]]]:
    target_norm = (target or "all").strip().lower().replace("-", "_")
    items: list[tuple[str, str | None, dict[str, Any]]] = []
    long_result = upload_results.get("long_form") or {}
    episode_record = _episode_record_for_upload(episode_id)
    if target_norm in {"all", "long", "long_form"}:
        items.append((
            "long_form",
            _video_id_from_result(long_result)
            or (episode_record or {}).get("youtube_video_id"),
            manifest.get("long_form") or {},
        ))
    shorts_meta = manifest.get("shorts") or []
    shorts_results = upload_results.get("shorts") or []
    if target_norm in {"all", "shorts"} or target_norm.startswith("short_"):
        by_rank = {
            int(s.get("rank") or idx): s
            for idx, s in enumerate(shorts_results, start=1)
            if isinstance(s, dict)
        }
        queued_short_ids = list(
            (episode_record or {}).get("youtube_shorts_video_ids") or []
        )
        for meta in shorts_meta:
            rank = int(meta.get("rank") or 0)
            label = f"short_{rank:02d}"
            if target_norm.startswith("short_") and target_norm != label:
                continue
            result = by_rank.get(rank) or {}
            queued_id = (
                str(queued_short_ids[rank - 1])
                if 0 < rank <= len(queued_short_ids) else None
            )
            items.append((
                label,
                _video_id_from_result(result) or queued_id,
                meta,
            ))
    if not items:
        raise ValueError(
            "--youtube-caption-target must be all, long, shorts, or short_NN"
        )
    return items


def _episode_record_for_upload(episode_id: str) -> dict[str, Any] | None:
    try:
        return find_episode(load_queue(), episode_id)
    except Exception:
        logger.debug("episode lookup failed for upload IDs", exc_info=True)
        return None


def _video_id_from_result(result: dict[str, Any]) -> str | None:
    video_id = str(result.get("video_id") or "").strip()
    if video_id:
        return video_id
    url = str(result.get("url") or "").strip()
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    return None


def _list_caption_languages(yt, *, video_id: str) -> set[str]:
    try:
        resp = yt.captions().list(part="snippet", videoId=video_id).execute()
    except Exception as e:
        logger.warning("caption list failed for %s: %s", video_id, e)
        return set()
    languages: set[str] = set()
    for item in resp.get("items") or []:
        snippet = item.get("snippet") or {}
        language = str(snippet.get("language") or "").strip()
        if language:
            languages.add(language)
    return languages


def _caption_backfill_run(
    *,
    target: str,
    languages: list[str],
    attempted: int,
    uploaded: int,
    skipped_existing: int,
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "languages": languages,
        "attempted": attempted,
        "uploaded": uploaded,
        "skipped_existing": skipped_existing,
        "warnings": warnings,
    }


def _safe_post_upload_step(
    warnings: list[dict[str, str]],
    step: str,
    fn,
) -> None:
    try:
        fn()
    except Exception as e:
        msg = str(e)
        logger.warning("YouTube post-upload step failed: %s: %s", step, msg)
        warnings.append({"step": step, "error": msg[:500]})


def _upload_one_video(
    yt,
    *,
    package_dir: Path,
    metadata: dict[str, Any],
    privacy_status: str,
    publish_at: str | None,
) -> dict[str, Any]:
    from googleapiclient.http import MediaFileUpload

    video_path = package_dir / metadata["video_path"]
    description_path = package_dir / metadata["description_path"]
    description = description_path.read_text() if description_path.exists() else ""
    effective_privacy = "private" if publish_at else privacy_status
    status_body: dict[str, Any] = {
        "privacyStatus": effective_privacy,
        "selfDeclaredMadeForKids": bool(metadata.get("made_for_kids", False)),
    }
    if publish_at:
        status_body["publishAt"] = publish_at
    body = {
        "snippet": {
            "title": metadata["title"][:100],
            "description": description[:5000],
            "tags": metadata.get("tags") or [],
            "categoryId": str(metadata.get("category_id") or "27"),
            "defaultLanguage": metadata.get("default_language") or "en",
        },
        "status": status_body,
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
        "privacy_status": effective_privacy,
        "publish_at": publish_at,
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


def _build_tags(
    incident: dict[str, Any],
    defaults: list[str],
    *,
    max_tags: int,
) -> list[str]:
    """Build a short, story-specific tag list.

    YouTube tags are weak discovery signals and spammy tag blocks can
    look low-quality, so we keep this intentionally tiny: company,
    one hooky story-kind phrase, then one broad niche tag.
    """
    company = (incident.get("company_name") or "").strip()
    story_kind = (incident.get("story_kind") or "").strip()
    hook = _tag_for_story_kind(story_kind)
    raw = [company, hook] + list(defaults)
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
        if len(out) >= max(1, int(max_tags)):
            break
    return out


def _tag_for_story_kind(story_kind: str) -> str:
    sk = story_kind.strip().lower()
    mapping = {
        "origin": "brand origin story",
        "rise_and_fall": "business rise and fall",
        "disruption": "market disruption story",
        "pivot": "impossible business pivot",
        "underdog_comeback": "underdog brand comeback",
        "founder_drama": "founder drama",
        "scandal_postmortem": "corporate scandal explained",
    }
    return mapping.get(sk, "business history")


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
    max_tags: int,
    publish_at: str | None,
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
        src = _resolve_short_source(ws=ws, src_dir=src_dir, item=item)
        if src is None:
            logger.warning(
                "youtube package: short source missing for rank=%s path=%r",
                item.get("rank"),
                item.get("path"),
            )
            continue
        rank = int(item.get("rank") or (len(entries) + 1))
        out_dir = shorts_dir / f"short_{rank:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"short_{rank:02d}.mp4"
        shutil.copy2(src, out_video)
        title_card_src = _resolve_short_title_card_source(
            ws=ws, src_dir=src_dir, item=item, rank=rank,
        )
        title_card_rel = None
        if title_card_src:
            out_title_card = out_dir / f"short_{rank:02d}_title_card.png"
            shutil.copy2(title_card_src, out_title_card)
            title_card_rel = _rel(out_title_card, package_dir)
        caption_src = _resolve_short_caption_source(
            ws=ws, src_dir=src_dir, item=item, rank=rank
        )
        caption_tracks = _build_caption_tracks(
            source_srt=caption_src,
            out_dir=out_dir / "subtitles",
            rel_prefix=f"shorts/short_{rank:02d}/subtitles",
            cfg_upload=cfg_upload,
        ) if caption_src else []
        title = (item.get("title_hint") or incident.get("company_name") or "Short")
        if "#Shorts" not in title:
            title = f"{title} #Shorts"
        metadata = {
            "kind": "short",
            "rank": rank,
            "title": title[:100],
            "description_path": f"shorts/short_{rank:02d}/description.txt",
            "tags": _build_tags(
                incident, base_tags + hashtags, max_tags=max_tags,
            ),
            "category_id": str(cfg_upload.get("category_id", "27")),
            "privacy_status": str(
                cfg_upload.get("default_privacy_status", "private")
            ),
            "publish_at": publish_at,
            "default_language": str(cfg_upload.get("default_language", "en")),
            "made_for_kids": bool(cfg_upload.get("made_for_kids", False)),
            "video_path": f"shorts/short_{rank:02d}/short_{rank:02d}.mp4",
            "thumbnail_path": title_card_rel,
            "caption_tracks": caption_tracks,
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


def _copy_thumbnail_folder(ws: Path, package_dir: Path) -> str | None:
    src = ws / "05_video" / "thumbnails"
    if not src.exists() or not src.is_dir():
        return None
    dst = package_dir / "thumbnails"
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            shutil.copy2(p, dst / p.name)
            copied += 1
    return "thumbnails" if copied else None


def _resolve_short_source(
    *, ws: Path, src_dir: Path, item: dict[str, Any]
) -> Path | None:
    """Resolve a Short video path, tolerating moved episode workspaces."""
    raw = str(item.get("path") or "").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else ws / p)
        if p.name:
            candidates.append(src_dir / p.name)

    try:
        rank = int(item.get("rank") or 0)
    except (TypeError, ValueError):
        rank = 0
    if rank > 0:
        candidates.append(src_dir / f"short_{rank:02d}.mp4")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_short_caption_source(
    *, ws: Path, src_dir: Path, item: dict[str, Any], rank: int
) -> Path | None:
    raw = str(item.get("caption_path") or "").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else ws / p)
        if p.name:
            candidates.append(src_dir / p.name)
    candidates.append(src_dir / f"short_{rank:02d}.srt")
    candidates.append(src_dir / "teaser_captions.srt")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_short_title_card_source(
    *, ws: Path, src_dir: Path, item: dict[str, Any], rank: int
) -> Path | None:
    raw = str(item.get("title_card_path") or "").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else ws / p)
        if p.name:
            candidates.append(src_dir / p.name)
    candidates.append(src_dir / f"short_{rank:02d}_title_card.png")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _build_caption_tracks(
    *,
    source_srt: Path,
    out_dir: Path,
    rel_prefix: str,
    cfg_upload: dict[str, Any],
) -> list[dict[str, str]]:
    if not source_srt.exists():
        return []
    specs = _language_specs(cfg_upload)
    if not specs:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks: list[dict[str, str]] = []
    source_lang = str(cfg_upload.get("default_language") or "en")
    for spec in specs:
        code = str(spec.get("code") or "").strip()
        name = str(spec.get("name") or code).strip()
        if not code:
            continue
        out_file = out_dir / f"captions.{_safe_lang_code(code)}.srt"
        is_source = bool(spec.get("source")) or code == source_lang
        if is_source:
            shutil.copy2(source_srt, out_file)
        else:
            translated = _translate_srt(
                source_srt=source_srt,
                language_code=code,
                language_name=name,
            )
            if not translated:
                logger.warning("subtitle translation skipped: %s", code)
                continue
            out_file.write_text(translated)
        tracks.append({
            "language": code,
            "name": name,
            "path": f"{rel_prefix}/{out_file.name}",
        })
    return tracks


def _language_specs(cfg_upload: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(cfg_upload.get("multilingual_subtitles_enabled", True)):
        return [{
            "code": str(cfg_upload.get("default_language") or "en"),
            "name": "English",
            "source": True,
        }]
    raw = cfg_upload.get("subtitle_languages") or []
    if not isinstance(raw, list):
        return []
    specs = [x for x in raw if isinstance(x, dict)]
    if not specs:
        return [{
            "code": str(cfg_upload.get("default_language") or "en"),
            "name": "English",
            "source": True,
        }]
    return specs


def _caption_tracks_for_metadata(metadata: dict[str, Any]) -> list[dict[str, str]]:
    tracks = metadata.get("caption_tracks") or []
    if isinstance(tracks, list) and tracks:
        return [
            {
                "language": str(t.get("language") or "en"),
                "name": str(t.get("name") or t.get("language") or "Caption"),
                "path": str(t.get("path") or ""),
            }
            for t in tracks
            if isinstance(t, dict) and t.get("path")
        ]
    path = metadata.get("caption_path")
    if not path:
        return []
    language = str(metadata.get("default_language") or "en")
    return [{"language": language, "name": "English", "path": str(path)}]


def _translate_srt(
    *, source_srt: Path, language_code: str, language_name: str
) -> str | None:
    blocks = _parse_srt(source_srt.read_text())
    if not blocks:
        return None
    translations: list[str] = []
    chunk_size = 30
    for start in range(0, len(blocks), chunk_size):
        texts = [b["text"] for b in blocks[start:start + chunk_size]]
        translated = _translate_caption_texts(
            texts=texts,
            language_code=language_code,
            language_name=language_name,
        )
        if len(translated) != len(texts):
            logger.warning(
                "subtitle translation count mismatch for %s: %d != %d",
                language_code, len(translated), len(texts),
            )
            return None
        translations.extend(translated)
    lines: list[str] = []
    for block, text in zip(blocks, translations):
        lines.extend([
            block["index"],
            block["timing"],
            text.strip(),
            "",
        ])
    return "\n".join(lines)


def _translate_caption_texts(
    *, texts: list[str], language_code: str, language_name: str
) -> list[str]:
    prompt = (
        "Translate these YouTube subtitle cue texts into "
        f"{language_name} ({language_code}). Preserve meaning, names, "
        "numbers, punchiness, and line-level order. Do not add commentary. "
        "Return JSON only as {\"translations\": [..]} with exactly the "
        "same number of strings.\n\n"
        f"TEXTS:\n{json.dumps(texts, ensure_ascii=False)}"
    )
    try:
        result = LLM(role="writer").complete_json(
            prompt, temperature=0.25, max_tokens=3500
        )
    except Exception as e:
        logger.warning("subtitle translation failed for %s: %s", language_code, e)
        return []
    values = result.get("translations") or []
    return [str(v).strip() for v in values if str(v).strip()]


def _parse_srt(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), -1)
        if timing_idx <= 0:
            continue
        out.append({
            "index": lines[0],
            "timing": lines[timing_idx],
            "text": " ".join(lines[timing_idx + 1:]).strip(),
        })
    return out


def _safe_lang_code(code: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", code.strip())
    return safe or "unknown"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _normalize_publish_at(value: Any) -> str | None:
    """Return an RFC3339-ish UTC timestamp string or None.

    Accepts empty values, a trailing-Z timestamp, or a Python ISO
    timestamp with +00:00. Validation is intentionally light; YouTube
    remains the final authority on whether the scheduled time is valid
    and far enough in the future.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        return text
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _write_summary(package_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"YouTube Upload Package: {manifest['episode_id']}",
        "",
        "Long form:",
        f"- video: {manifest['long_form']['video_path']}",
        f"- title: {manifest['long_form']['title']}",
        f"- tags: {', '.join(manifest['long_form'].get('tags') or [])}",
        f"- publish_at: {manifest['long_form'].get('publish_at')}",
        f"- thumbnail: {manifest['long_form'].get('thumbnail_path')}",
        f"- thumbnail candidates: {manifest['contents'].get('thumbnails_dir')}",
        "- captions: "
        f"{len(manifest['long_form'].get('caption_tracks') or [])} tracks",
        "",
        f"Shorts: {len(manifest.get('shorts') or [])}",
    ]
    for s in manifest.get("shorts") or []:
        lines.append(
            f"- {s['video_path']}: {s['title']} "
            f"({len(s.get('caption_tracks') or [])} subtitle tracks; "
            f"thumbnail={s.get('thumbnail_path')})"
        )
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
