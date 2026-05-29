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
    pad_audio_silence,
    render_sfx_track,
)
from ..music_library import MusicLibrary
from ..sfx_library import SFXLibrary
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

    # ----- voice padding for title + closing cards (Batch J 2026-05-29) -----
    # S12 prepends an optional title card and appends an optional closing
    # source-attribution card around the per-beat clips. Without padding
    # the voice with silence, final_mix.wav is voice_seconds long, but
    # the assembled video timeline is title + voice + closing.
    # S12's concat uses `-shortest` which then truncates the video at
    # audio length and the closing card never appears. Padding the voice
    # by exactly those amounts makes audio ≥ video and lets the closing
    # card render — and, because there's no voice to sidechain against
    # during the tail, the music bed swells to full level there, which
    # is the "music keeps playing through credits" behavior the operator
    # asked for.
    prod_cfg = cfg.production
    head_pad = max(0.0, float(prod_cfg.get("opening_title_card_seconds", 0)))
    closing_card = max(0.0, float(prod_cfg.get("closing_card_seconds", 0)))
    # +1.0s tail buffer absorbs ffmpeg rounding so -shortest doesn't
    # nibble the last frame of the closing card.
    tail_pad = closing_card + (1.0 if closing_card > 0 else 0.0)

    if head_pad > 0 or tail_pad > 0:
        voice_padded = ws / "04_audio" / "voice_padded.wav"
        try:
            pad_audio_silence(voice_path, voice_padded, head_pad, tail_pad)
            voice_for_mix = voice_padded
            voice_padded_seconds = voice_seconds + head_pad + tail_pad
            logger.info(
                "S11 voice padded: +%.1fs head / +%.1fs tail "
                "(total %.1fs for title+voice+closing cover)",
                head_pad, tail_pad, voice_padded_seconds,
            )
        except Exception as e:
            logger.warning("voice padding failed (%s); using raw voice", e)
            voice_for_mix = voice_path
            voice_padded_seconds = voice_seconds
    else:
        voice_for_mix = voice_path
        voice_padded_seconds = voice_seconds

    ml_cfg = cfg.music_library
    target_seconds = voice_padded_seconds + max(0.0, float(ml_cfg.get("music_start_offset_seconds", 20.0))) + 6.0
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

    # ----- Phase 2: SFX track (Batch C 2026-05-26) -----
    # head_pad shifts voice content forward inside the padded mix, so
    # SFX cue start offsets (which come from voice_timing.json in raw-
    # voice frame of reference) need the same shift to stay in sync.
    sfx_wav, sfx_picks = _build_sfx_track(
        ws=ws, episode=episode, cfg=cfg,
        voice_seconds=voice_seconds, music_dir=music_dir,
        voice_start_offset_seconds=head_pad,
        total_seconds_full=voice_padded_seconds,
    )

    spec = AudioMixSpec(
        voice_wav=voice_for_mix,
        music_wavs=music_wavs,
        out_wav=final_mix,
        voice_gain_db=-18.0,
        music_gain_db=float(ml_cfg.get("music_gain_db", -28.0)),
        ambient_gain_db=-30.0,
        # Batch F 2026-05-27: duck depth raised 6→10. The shallower 6dB
        # duck was contributing to flat dynamics — when voice paused,
        # the music barely swelled back, so the mix felt evenly loud
        # throughout. 10 dB gives audible swell-back during voice gaps
        # which restores emotional contour at act boundaries.
        duck_depth_db=10.0,
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
        sfx_wav=sfx_wav,
        sfx_gain_db=0.0,   # gain is baked into the rendered sfx_track.wav
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
        # Padding applied to cover title + closing cards in S12.
        # Added Batch J 2026-05-29.
        "voice_padding_head_seconds": round(head_pad, 3),
        "voice_padding_tail_seconds": round(tail_pad, 3),
        "voice_padded_seconds": round(voice_padded_seconds, 3),
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
        "sfx_used": [
            {
                "file": sp["file"],
                "cue": sp["cue"],
                "start_seconds": sp["start_seconds"],
                "gain_db": sp["gain_db"],
            }
            for sp in sfx_picks
        ],
        "music_start_offset_seconds": spec.music_start_offset_seconds,
        "lufs_target": spec.lufs_target,
        "music_gain_db": spec.music_gain_db,
    }
    (ws / "04_audio" / "mix_manifest.json").write_text(json.dumps(mix_manifest, indent=2))
    logger.info("S11 complete: final_mix.wav (%d music tracks, %d SFX cues, "
                "voice-only=%s)",
                len(picks), len(sfx_picks), not music_wavs)
    return None


# ----------------------------------------------------------------------
# S11 Phase 2: SFX track (Batch C 2026-05-26)
# ----------------------------------------------------------------------

def _build_sfx_track(
    *,
    ws,
    episode: dict,
    cfg,
    voice_seconds: float,
    music_dir,
    voice_start_offset_seconds: float = 0.0,
    total_seconds_full: float | None = None,
) -> tuple[Path | None, list[dict]]:
    """Build sfx_track.wav with all SFX cues placed at their beat-start
    offsets, gain pre-applied. Returns (path, sfx_picks) where path
    is None when SFX is disabled / no matches / no library.

    Per Q-C1 (confirmed): beat-anchored offsets. We read
    voice_timing.json (emitted by S10) for accurate beat-start
    timestamps. If voice_timing.json is missing, we fall back to
    cumulative estimated_seconds from beat_sheet.json.
    """
    sfx_cfg = cfg.sfx_library
    if not sfx_cfg.get("enabled", False):
        return None, []

    sfx_lib = SFXLibrary()

    # Read beat sheet + voice timing.
    beat_sheet_path = ws / "02_script" / "beat_sheet.json"
    if not beat_sheet_path.exists():
        logger.info("S11 SFX: no beat_sheet.json; skipping SFX")
        return None, []
    beat_sheet = json.loads(beat_sheet_path.read_text())
    beats = beat_sheet.get("beats", [])

    voice_timing_path = ws / "04_audio" / "voice_timing.json"
    starts_by_id: dict[str, float] = {}
    if voice_timing_path.exists():
        try:
            vt = json.loads(voice_timing_path.read_text())
            for b in vt.get("beats", []):
                bid = b.get("beat_id", "")
                if bid:
                    starts_by_id[bid] = float(b.get("start_seconds", 0.0))
        except Exception as e:
            logger.warning("voice_timing.json unreadable: %s", e)

    # Build the cue list.
    cues: list[dict] = []
    sfx_picks_meta: list[dict] = []
    cumulative = 0.0
    for b in beats:
        bid = b.get("beat_id", "")
        cue = (b.get("sfx_cue") or "").strip().lower()
        # Beat start: prefer voice_timing (exact); else cumulative
        # estimated_seconds (approximate).
        start = starts_by_id.get(bid, cumulative)
        cumulative += float(b.get("estimated_seconds", 0.0))

        if not cue or cue == "silence":
            continue
        pick = sfx_lib.pick_cue(cue, beat_id=bid)
        if pick is None:
            continue
        # Shift by voice_start_offset_seconds so SFX lands inside the
        # padded mix, not under the silent title-card head (Batch J).
        effective_start = start + voice_start_offset_seconds
        cues.append({
            "path": pick.path,
            "start_seconds": effective_start,
            "gain_db": pick.gain_db_hint,
        })
        sfx_picks_meta.append({
            "file": pick.path.name,
            "cue": pick.cue,
            "start_seconds": round(effective_start, 3),
            "gain_db": pick.gain_db_hint,
        })

    if not cues:
        logger.info("S11 SFX: no matching cues found in library; skipping")
        return None, []

    sfx_path = music_dir / "sfx_track.wav"
    ok = render_sfx_track(
        cues, sfx_path,
        total_seconds=float(total_seconds_full if total_seconds_full is not None
                            else voice_seconds),
    )
    if not ok:
        logger.warning("S11 SFX: render_sfx_track failed; mixing without SFX")
        return None, []
    logger.info("S11 SFX: %d cues placed in sfx_track.wav", len(cues))
    return sfx_path, sfx_picks_meta


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
