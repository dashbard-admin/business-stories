"""Skeleton-manifest scanner for assets/sfx_library/.

Walks the SFX library directory and:
  1. For each audio file present on disk that has a matching manifest
     entry, backfills `duration_seconds` from ffprobe (overwrites any
     existing duration_seconds value).
  2. For each audio file present on disk WITHOUT a matching manifest
     entry, appends a minimal stub: `{file, cue: "?", license:
     "unknown", attribution: "", source_url: "", duration_seconds: <secs>}`
     so the operator can fill in the metadata.
  3. For each manifest entry whose `file` is missing from disk, logs
     a warning (but leaves the entry intact — the operator might be
     about to drop the file in).

Symmetric with pipeline/tools/scan_music_library.py.

Run:
    python -m pipeline.tools.scan_sfx_library
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ..config import load_config
from ..ffmpeg_builder import require_ffprobe

logger = logging.getLogger("hermes.scan_sfx_library")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


def _probe_duration(path: Path) -> float | None:
    """Return the duration in seconds via ffprobe. None on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        return round(float(out), 2)
    except Exception as e:
        logger.warning("ffprobe failed for %s: %s", path.name, e)
        return None


def main() -> int:
    require_ffprobe()
    cfg = load_config()
    sl = cfg.sfx_library
    sfx_dir = Path(sl["path"])
    manifest_path = Path(sl["manifest"])

    if not sfx_dir.exists():
        logger.error("SFX directory does not exist: %s", sfx_dir)
        return 1

    # Load existing manifest (or create the scaffold).
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
    else:
        data = {"sounds": []}
    sounds: list[dict] = list(data.get("sounds") or [])
    by_file = {(s.get("file") or ""): s for s in sounds}

    # Walk the directory.
    files_on_disk: dict[str, Path] = {}
    for p in sfx_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        files_on_disk[p.name] = p

    appended = 0
    backfilled = 0
    for name, path in sorted(files_on_disk.items()):
        existing = by_file.get(name)
        dur = _probe_duration(path)
        if existing is None:
            stub = {
                "file": name,
                "cue": "?",
                "intensity": "medium",
                "license": "unknown",
                "attribution": "",
                "source_url": "",
                "duration_seconds": dur,
            }
            sounds.append(stub)
            by_file[name] = stub
            appended += 1
            logger.info("appended stub for %s (%.2fs)", name, dur or 0)
        else:
            old = existing.get("duration_seconds")
            if dur is not None and old != dur:
                existing["duration_seconds"] = dur
                backfilled += 1
                logger.info("backfilled duration for %s (%s -> %.2fs)",
                            name, old, dur)

    # Warn about manifest entries with no file on disk.
    missing_count = 0
    for s in sounds:
        fn = s.get("file") or ""
        if fn and fn not in files_on_disk:
            logger.warning("manifest entry %s has no file on disk", fn)
            missing_count += 1

    data["sounds"] = sounds
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    logger.info(
        "scan complete: %d files on disk, %d appended, %d duration-"
        "backfilled, %d missing files",
        len(files_on_disk), appended, backfilled, missing_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
