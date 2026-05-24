"""S11 — Audio Post.

Picks N tracks from the operator-curated local music library
(assets/music_library/) whose total duration covers voice_full.wav
plus a tail, concatenates them with a 4-second crossfade into a
single music_bed.wav, then sidechain-ducks the bed under the voice
and applies EBU R128 loudnorm to YouTube spec.

Unlike the maritime pipeline this stage does NOT:
  - run MusicGen or any other audio synthesis
  - generate or layer SFX
  - hit any external audio-gen HTTP server

If the music library is empty or the manifest is missing, S11 falls
back to a voice-only mix (still loudnormed). Operator can populate
the library and re-run.

Inputs:  04_audio/voice_full.wav, 04_audio/voice_timing.json
Outputs: 04_audio/final_mix.wav  +  04_audio/mix_manifest.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import soundfile as sf

from ..config import load_config
from ..ffmpeg_builder import (
    AudioMixSpec,
    audio_post_mix,
    concat_music_with_crossfade,
    get_duration_seconds,
)
from ..music_library import MusicLibrary
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s11")


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    voice_path = ws / "04_audio" / "voice_full.wav"
    if not voice_path.exists():
        return "no voice_full.wav; S10 must run first"

    # Voice duration drives target_seconds for the bed.
    try:
        voice_seconds = get_duration_seconds(voice_path)
    except Exception:
        info = sf.info(str(voice_path))
        voice_seconds = info.frames / info.samplerate
    logger.info("voice duration: %.1fs", voice_seconds)

    ml_cfg = cfg.music_library
    target_seconds = voice_seconds + max(0.0, float(ml_cfg.get("music_start_offset_seconds", 20.0))) + 6.0
    crossfade_s = float(ml_cfg.get("crossfade_seconds", 4.0))

    # ----- pick music tracks -----
    incident = episode["incident"]
    library = MusicLibrary()
    topic_keywords = _episode_keywords(incident)
    picks = library.pick_bed(
        topic_keywords=topic_keywords,
        narrator_id=episode["narrator"],
        archetype_id=episode["archetype"],
        visual_style_id=episode["visual_style"],
        story_kind=incident.get("story_kind", ""),
        target_seconds=target_seconds,
        crossfade_seconds=crossfade_s,
        seed=hash(incident["company_name"]) & 0xffff,
    )

    music_dir = ws / "04_audio" / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    final_mix = ws / "04_audio" / "final_mix.wav"

    music_wavs: list[Path] = []
    if picks:
        # Concat picked tracks with crossfade into a single bed file.
        music_bed = music_dir / "music_bed.wav"
        try:
            concat_music_with_crossfade(
                [p.path for p in picks], music_bed, crossfade_seconds=crossfade_s,
            )
            music_wavs = [music_bed]
            logger.info("S11 music bed assembled: %d tracks", len(picks))
        except Exception as e:
            logger.warning("music bed assembly failed (%s); falling back to voice-only",
                           e)
            music_wavs = []
    else:
        logger.info("S11: no tracks picked (mock mode, empty library, or no match) — voice-only mix")

    spec = AudioMixSpec(
        voice_wav=voice_path,
        music_wavs=music_wavs,
        out_wav=final_mix,
        voice_gain_db=-18.0,
        music_gain_db=float(ml_cfg.get("music_gain_db", -28.0)),
        ambient_gain_db=-30.0,
        duck_depth_db=6.0,
        duck_attack_ms=100,
        duck_release_ms=800,
        lufs_target=float(cfg.quality_gates.get("audio_lufs_target", -14.0)),
        true_peak_dbtp=float(cfg.quality_gates.get("audio_peak_max_dbtp", -1.0)),
        lra=11.0,
        voice_dynaudnorm_enabled=bool(ml_cfg.get("voice_dynaudnorm_enabled", True)),
        voice_dynaudnorm_framelen_ms=int(ml_cfg.get("voice_dynaudnorm_framelen_ms", 200)),
        voice_dynaudnorm_gauss=int(ml_cfg.get("voice_dynaudnorm_gauss", 11)),
        voice_dynaudnorm_max_gain=float(ml_cfg.get("voice_dynaudnorm_max_gain", 15.0)),
        music_start_offset_seconds=float(ml_cfg.get("music_start_offset_seconds", 20.0)),
    )

    try:
        audio_post_mix(spec)
    except Exception as e:
        # Voice-only retry — sometimes ffmpeg sidechain misbehaves on
        # very short voice tracks or odd music sample-rates. Voice-only
        # is the safest fallback and the rest of the pipeline still works.
        if music_wavs:
            logger.warning("audio_post_mix failed with music (%s); retry voice-only", e)
            spec.music_wavs = []
            try:
                audio_post_mix(spec)
            except Exception as e2:
                return f"audio_post_mix failed in voice-only fallback: {e2}"
        else:
            return f"audio_post_mix failed: {e}"

    # ----- write manifest -----
    mix_manifest = {
        "voice_path": str(voice_path.relative_to(ws)),
        "final_mix_path": str(final_mix.relative_to(ws)),
        "voice_seconds": round(voice_seconds, 3),
        "tracks_used": [
            {
                "file": p.path.name,
                "duration_seconds": p.duration_seconds,
                "score": round(p.score, 3),
                "gain_db_hint": p.gain_db_hint,
                "meta": p.meta,
            }
            for p in picks
        ],
        "music_start_offset_seconds": spec.music_start_offset_seconds,
        "lufs_target": spec.lufs_target,
        "music_gain_db": spec.music_gain_db,
    }
    (ws / "04_audio" / "mix_manifest.json").write_text(json.dumps(mix_manifest, indent=2))
    logger.info("S11 complete: final_mix.wav (%d music tracks, voice-only=%s)",
                len(picks), not music_wavs)
    return None


def _episode_keywords(incident: dict) -> list[str]:
    """Pull free-text tokens from the incident's hero/conflict/story_kind."""
    bag: list[str] = []
    for field in ("hero", "conflict", "one_line_pitch", "company_name",
                  "founder_or_protagonist", "story_kind"):
        v = (incident.get(field) or "").strip()
        if v:
            bag.extend(v.lower().split())
    # Cheap stop filter + dedup
    stop = {"the", "of", "and", "a", "an", "to", "in", "for", "with", "on",
            "by", "at", "as", "is", "was", "be", "this", "that", "from"}
    out: list[str] = []
    seen = set()
    for w in bag:
        w2 = "".join(c for c in w if c.isalnum())
        if not w2 or w2 in stop or w2 in seen:
            continue
        seen.add(w2)
        out.append(w2)
    return out
