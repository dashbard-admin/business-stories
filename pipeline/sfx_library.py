"""Local SFX library matcher.

S08 emits a `sfx_cue` per beat (from a fixed catalog: typewriter,
keyboard, phone_ring, applause, door_slam, traffic_hum, market_bell,
newsprint, clock_tick, silence). S11 Phase 2 (added Batch C
2026-05-26) looks up the matching short SFX file from the operator-
curated library and mixes it under voice at the beat's start
timestamp, ducked well below the voice level.

Library layout:
    ${root}/assets/sfx_library/
      ├── manifest.json
      └── *.wav | *.mp3 | *.flac | *.aiff | *.ogg

Manifest schema (`sounds`):
    {
      "sounds": [
        {
          "file": "typewriter_01.wav",
          "cue": "typewriter",                  # one of the cue names
          "intensity": "soft" | "medium" | "sharp",
          "license": "CC0" | "CC-BY" | "freesound" | "Pixabay" | "unknown",
          "attribution": "Composer Name — Sound Title",  # required for CC-BY
          "source_url": "https://freesound.org/...",
          "duration_seconds": 0.9
        },
        ...
      ]
    }

The picker is deterministic per (cue, beat_id) so repeated S11 runs
on the same episode produce the same mix.

Symmetric with pipeline/music_library.py — same shape, same defaults,
same license_report() interface.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from .config import load_config

logger = logging.getLogger("hermes.sfx_library")


@dataclass
class SFXPick:
    path: Path
    cue: str
    duration_seconds: float
    gain_db_hint: float = 0.0
    meta: dict | None = None


class SFXLibrary:
    """Pure-Python matcher over the operator-curated SFX library."""

    def __init__(self):
        cfg = load_config()
        sl = cfg.sfx_library
        self._enabled = bool(sl.get("enabled", False))
        self._mock = cfg.mock_mode
        self._dir = Path(sl["path"])
        self._manifest_path = Path(sl["manifest"])
        self._sounds: list[dict] = []
        self._default_gain_db = float(sl.get("default_gain_db", -18.0))
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self._enabled:
            return
        if not self._manifest_path.exists():
            logger.warning("sfx_library: manifest %s not present",
                           self._manifest_path)
            return
        try:
            data = json.loads(self._manifest_path.read_text())
            self._sounds = list(data.get("sounds") or [])
        except Exception as e:
            logger.error("sfx_library: manifest parse failed: %s", e)
            self._sounds = []
        logger.info("sfx_library: loaded %d sounds", len(self._sounds))

    def _resolve(self, file_name: str) -> Path | None:
        p = self._dir / file_name
        if p.exists():
            return p
        logger.warning("sfx_library: sound file missing on disk: %s", p)
        return None

    # ----- public picker -----

    def pick_cue(
        self,
        cue: str,
        *,
        beat_id: str = "",
        max_duration_seconds: float = 6.0,
    ) -> SFXPick | None:
        """Return a single SFXPick for the named cue, or None when the
        library is disabled / has no match. The `silence` cue always
        returns None (no SFX mixed for silence beats). Deterministic
        per (cue, beat_id) pair so repeated S11 runs are stable."""
        if not self._enabled or self._mock or not self._sounds:
            return None
        cue = (cue or "").strip().lower()
        if not cue or cue == "silence":
            return None

        matching = [
            s for s in self._sounds
            if (s.get("cue") or "").strip().lower() == cue
            and (s.get("duration_seconds") or 0) <= max_duration_seconds
        ]
        if not matching:
            logger.info("sfx_library: no match for cue %r within "
                        "%.1fs cap", cue, max_duration_seconds)
            return None

        rng = random.Random(f"{cue}|{beat_id}")
        choice = rng.choice(matching)
        path = self._resolve(choice.get("file", ""))
        if not path:
            return None
        return SFXPick(
            path=path,
            cue=cue,
            duration_seconds=float(choice.get("duration_seconds") or 0.0),
            gain_db_hint=float(choice.get("gain_db_hint",
                                          self._default_gain_db)),
            meta=choice,
        )

    # ----- license reporting -----

    def license_report(self, file_names: list[str]) -> list[dict]:
        """For each file in `file_names`, return its license / attribution
        entry. Mirrors MusicLibrary.license_report. S12 emits the
        combined music + SFX report to
        06_metadata/license_attributions.txt."""
        by_file = {(s.get("file") or ""): s for s in self._sounds}
        out: list[dict] = []
        for fn in file_names:
            s = by_file.get(fn)
            if not s:
                out.append({
                    "file": fn,
                    "license": "unknown",
                    "attribution": "",
                    "source_url": "",
                    "warning": "not in manifest",
                })
                continue
            out.append({
                "file": fn,
                "cue": s.get("cue", ""),
                "license": s.get("license") or "unknown",
                "attribution": s.get("attribution") or "",
                "source_url": s.get("source_url") or "",
            })
        return out
