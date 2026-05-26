"""Local music-library matcher.

Replaces the maritime pipeline's audio_gen (MusicGen + Stable Audio).
S11 picks a sequence of tracks from a curated local library whose
total duration covers the voice track plus a tail, then ffmpeg
concatenates them with crossfade and sidechain-ducks under the voice.

Library layout:
    ${root}/assets/music_library/
      ├── manifest.json
      └── *.wav | *.mp3 | *.flac | *.aiff | *.ogg

Manifest schema (`tracks`):
    {
      "tracks": [
        {
          "file": "ambient_underdog_01.wav",
          "mood": ["hopeful", "rising", "determined"],
          "tempo": "slow" | "mid" | "fast",
          "instruments": ["piano", "strings", "synth_pad"],
          "tags": ["origin", "founder", "underdog"],
          "duration_seconds": 312.4,
          "suits_narrators": ["N1", "N3"],   # optional pin list
          "suits_archetypes": ["A1", "A5"],  # optional pin list
          "suits_styles": ["V1"],            # optional pin list
          "suits_story_kinds": ["origin", "underdog"],  # optional
          "gain_db_hint": 0.0                # optional per-track gain trim
        },
        ...
      ]
    }

The picker scores each track by:
  - keyword overlap between the topic keyword set and track
    mood + instruments + tags
  - +1 if narrator is in track.suits_narrators (when present)
  - +1 if archetype is in track.suits_archetypes (when present)
  - +1 if visual_style is in track.suits_styles (when present)
  - +2 if story_kind is in track.suits_story_kinds (when present)

It then greedily fills target_seconds with the highest-scoring tracks,
avoiding immediate repetition.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .config import load_config

logger = logging.getLogger("hermes.music_library")

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


@dataclass
class TrackPick:
    path: Path
    duration_seconds: float
    score: float
    gain_db_hint: float = 0.0
    meta: dict | None = None


def _kebab_tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", (text or "").lower()))


class MusicLibrary:
    """Pure-Python matcher over the operator-curated music library."""

    def __init__(self):
        cfg = load_config()
        ml = cfg.music_library
        self._enabled = bool(ml.get("enabled", True))
        self._mock = cfg.mock_mode
        self._dir = Path(ml["path"])
        self._manifest_path = Path(ml["manifest"])
        self._tracks: list[dict] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self._enabled:
            return
        if not self._manifest_path.exists():
            logger.warning("music_library: manifest %s not present", self._manifest_path)
            return
        try:
            data = json.loads(self._manifest_path.read_text())
            self._tracks = list(data.get("tracks") or [])
        except Exception as e:
            logger.error("music_library: manifest parse failed: %s", e)
            self._tracks = []
        logger.info("music_library: loaded %d tracks", len(self._tracks))

    def _resolve(self, file_name: str) -> Path | None:
        p = self._dir / file_name
        if p.exists():
            return p
        logger.warning("music_library: track file missing on disk: %s", p)
        return None

    # ----- license reporting (added Batch A 2026-05-26) -----

    def license_report(self, file_names: list[str]) -> list[dict]:
        """For each file in `file_names` that resolves to a manifest
        track, return a dict with `{file, license, attribution,
        source_url}`. Missing / unknown licenses surface as
        `license="unknown"` so the operator sees them at attribution
        time. Unknown tracks (not in the manifest at all) get a
        warning and a sentinel entry.

        S12 calls this with the list of bed-track filenames actually
        used in the episode (from mix_manifest.json), emits the
        result to 06_metadata/license_attributions.txt for paste
        into the YouTube description.
        """
        by_file = {(t.get("file") or ""): t for t in self._tracks}
        out: list[dict] = []
        for fn in file_names:
            t = by_file.get(fn)
            if not t:
                logger.warning(
                    "music_library.license_report: track %s not in manifest",
                    fn,
                )
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
                "license": (t.get("license") or "unknown"),
                "attribution": (t.get("attribution") or ""),
                "source_url": (t.get("source_url") or ""),
            })
        return out

    # ----- public picker -----

    def pick_bed(
        self,
        *,
        topic_keywords: list[str],
        narrator_id: str,
        archetype_id: str,
        visual_style_id: str,
        story_kind: str = "",
        target_seconds: float = 900.0,
        crossfade_seconds: float = 4.0,
        seed: int | None = None,
    ) -> list[TrackPick]:
        """Return an ordered list of TrackPick that totals at least
        `target_seconds` (plus crossfade slack).

        In mock mode (or when no tracks exist), returns an empty list —
        S11 then falls back to a voice-only mix.
        """
        if not self._enabled or self._mock or not self._tracks:
            return []

        rng = random.Random(seed)
        topic_set = set(t.lower() for t in (topic_keywords or []) if t)

        scored: list[tuple[float, dict, Path, float]] = []
        for t in self._tracks:
            file_name = t.get("file") or ""
            p = self._resolve(file_name)
            if p is None:
                continue
            dur = float(t.get("duration_seconds") or 0)
            if dur <= 0:
                continue
            score = _score_track(
                track=t,
                topic_keywords=topic_set,
                narrator_id=narrator_id,
                archetype_id=archetype_id,
                visual_style_id=visual_style_id,
                story_kind=story_kind,
            )
            scored.append((score, t, p, dur))

        if not scored:
            return []

        # Stable random tiebreak so two equally-good tracks don't always
        # play in the same order across episodes.
        scored.sort(key=lambda x: (x[0], rng.random()), reverse=True)

        picks: list[TrackPick] = []
        running = 0.0
        # Each track contributes (dur - crossfade) of *new* audio since
        # the crossfade overlaps the previous track's tail.
        for (sc, meta, path, dur) in scored:
            # Avoid immediate file repeat unless library is too small
            if picks and picks[-1].path == path and len(scored) > 1:
                continue
            picks.append(TrackPick(
                path=path,
                duration_seconds=dur,
                score=sc,
                gain_db_hint=float(meta.get("gain_db_hint", 0.0) or 0.0),
                meta=meta,
            ))
            running += max(0.0, dur - crossfade_seconds)
            if running >= target_seconds:
                break

        # If we still haven't filled the duration, loop through the
        # scored list again (skipping immediate repeats) — better to
        # repeat a track than to silence the bed.
        if running < target_seconds and scored:
            i = 0
            while running < target_seconds and i < 256:  # hard safety cap
                cand = scored[i % len(scored)]
                _, meta, path, dur = cand
                if picks and picks[-1].path == path:
                    i += 1
                    continue
                picks.append(TrackPick(
                    path=path,
                    duration_seconds=dur,
                    score=cand[0],
                    gain_db_hint=float(meta.get("gain_db_hint", 0.0) or 0.0),
                    meta=meta,
                ))
                running += max(0.0, dur - crossfade_seconds)
                i += 1

        logger.info("music_library: picked %d tracks (target=%.0fs, picked=%.0fs)",
                    len(picks), target_seconds, running)
        return picks


def _score_track(
    *,
    track: dict,
    topic_keywords: set[str],
    narrator_id: str,
    archetype_id: str,
    visual_style_id: str,
    story_kind: str,
) -> float:
    """Score a track against an episode's signal."""
    score = 0.0

    # Free-text overlap (mood + tags + instruments + filename stem)
    bag = set()
    for field in ("mood", "tags", "instruments"):
        for v in (track.get(field) or []):
            bag.update(_kebab_tokens(str(v)))
    bag.update(_kebab_tokens(str(track.get("file") or "").rsplit(".", 1)[0]))

    if topic_keywords and bag:
        overlap = len(topic_keywords & bag)
        # Mild diminishing returns — a track that matches every keyword
        # shouldn't dominate so completely that it monopolises the bed.
        score += overlap * 0.7

    # Explicit pins
    if narrator_id and narrator_id in (track.get("suits_narrators") or []):
        score += 1.0
    if archetype_id and archetype_id in (track.get("suits_archetypes") or []):
        score += 1.0
    if visual_style_id and visual_style_id in (track.get("suits_styles") or []):
        score += 1.0
    if story_kind and story_kind in (track.get("suits_story_kinds") or []):
        score += 2.0

    return score
