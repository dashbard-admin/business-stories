"""Skeleton-manifest generator for assets/music_library/.

For every audio file in assets/music_library/ that is NOT already in
manifest.json, append a minimal entry (file, duration_seconds) so the
operator can fill in mood/instruments/tags by hand. Existing entries
are left untouched.

Run:
    python -m pipeline.tools.scan_music_library
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from ..config import load_config
from ..ffmpeg_builder import require_ffprobe

logger = logging.getLogger("hermes.scan_music_library")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


def _duration_seconds(path: Path) -> float:
    cmd = [require_ffprobe(),
           "-v", "error",
           "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1",
           str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception as e:
        logger.warning("ffprobe failed on %s: %s", path.name, e)
        return 0.0


def main() -> int:
    cfg = load_config()
    ml = cfg.music_library
    lib_dir = Path(ml["path"])
    manifest_path = Path(ml["manifest"])

    if not lib_dir.exists():
        logger.error("music library directory %s does not exist", lib_dir)
        return 2

    manifest: dict = {"tracks": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as e:
            logger.warning("manifest unreadable (%s) — starting fresh", e)
            manifest = {"tracks": []}

    by_file: dict[str, dict] = {t.get("file", ""): t for t in manifest.get("tracks", []) if t.get("file")}

    added = 0
    for f in sorted(lib_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if f.name in by_file:
            continue
        dur = _duration_seconds(f)
        entry = {
            "file": f.name,
            "mood": [],
            "tempo": "mid",
            "instruments": [],
            "tags": [],
            "duration_seconds": round(dur, 2),
            "suits_narrators": [],
            "suits_archetypes": [],
            "suits_styles": [],
            "suits_story_kinds": [],
            "gain_db_hint": 0.0,
        }
        by_file[f.name] = entry
        added += 1
        logger.info("added: %s (%.1fs)", f.name, dur)

    manifest["tracks"] = list(by_file.values())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest %s now has %d track(s) (+%d new)",
                manifest_path, len(manifest["tracks"]), added)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
