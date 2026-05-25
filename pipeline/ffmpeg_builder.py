"""Programmatic ffmpeg command builder.

All ffmpeg invocations in this pipeline funnel through this module so
escaping, filter-graph composition, and the loudnorm two-pass dance
live in one place. Functions:

  - ken_burns_clip(image_path, duration, motion, out_path)
  - concat_clips(clip_paths, audio_path, out_path)
  - audio_post_mix(spec) — sidechain duck + EBU R128 loudnorm
  - extract_audio(video_path, out_path)
  - probe(path) / get_duration_seconds(path)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("hermes.ffmpeg")

FPS = 30
OUT_W = 1920
OUT_H = 1080
OVERSCAN_W = 2880
OVERSCAN_H = 1620

# Supersample width passed to scale= before zoompan. Smooths Ken Burns
# motion at the cost of one extra upscale per clip. See maritime's
# notes — the integer-pixel crop window of zoompan steps visibly at
# native 1920px source.
KEN_BURNS_SUPERSAMPLE_W = 8000


def require_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError("ffmpeg not found on PATH")
    return p


def require_ffprobe() -> str:
    p = shutil.which("ffprobe")
    if not p:
        raise RuntimeError("ffprobe not found on PATH")
    return p


# -------------------- Ken Burns --------------------

@dataclass
class KenBurnsParams:
    start_scale: float = 1.00
    end_scale: float = 1.18
    start_x: float = 0.50
    end_x: float = 0.50
    start_y: float = 0.50
    end_y: float = 0.50


def motion_to_params(motion: str) -> KenBurnsParams:
    motion = motion.lower()
    if motion == "slow_zoom_in":
        return KenBurnsParams(1.00, 1.18, 0.5, 0.5, 0.5, 0.45)
    if motion == "slow_zoom_out":
        return KenBurnsParams(1.18, 1.00, 0.5, 0.5, 0.5, 0.50)
    if motion == "slow_pan_left":
        return KenBurnsParams(1.10, 1.10, 0.55, 0.45, 0.5, 0.5)
    if motion == "slow_pan_right":
        return KenBurnsParams(1.10, 1.10, 0.45, 0.55, 0.5, 0.5)
    if motion == "hold_still":
        return KenBurnsParams(1.03, 1.03, 0.5, 0.5, 0.5, 0.5)
    return KenBurnsParams()


def ken_burns_clip(
    image_path: Path,
    duration: float,
    motion: str,
    out_path: Path,
    *,
    fade_in_seconds: float = 0.0,
    fade_out_seconds: float = 0.0,
) -> None:
    """Render a single Ken Burns clip from a still image.

    `fade_in_seconds` / `fade_out_seconds` add ffmpeg `fade` filters at
    the start / end of the clip respectively. Each fade is internally
    clamped to at most 40% of `duration` so very short beats remain
    visible. Pass 0 (default) to disable either side.
    """
    p = motion_to_params(motion)
    n_frames = max(1, int(duration * FPS))

    z_expr = (
        f"{p.start_scale}+(({p.end_scale}-{p.start_scale})/{n_frames-1})*on"
        if n_frames > 1 else f"{p.start_scale}"
    )
    xc = (
        f"({p.start_x}+(({p.end_x}-{p.start_x})/{n_frames-1})*on)"
        if n_frames > 1 else f"{p.start_x}"
    )
    yc = (
        f"({p.start_y}+(({p.end_y}-{p.start_y})/{n_frames-1})*on)"
        if n_frames > 1 else f"{p.start_y}"
    )
    x_expr = f"iw*{xc}-(iw/zoom)/2"
    y_expr = f"ih*{yc}-(ih/zoom)/2"

    # Clamp fade lengths so they cannot overlap or exceed the clip.
    # 40% of duration on each side leaves at least 20% solid hold for
    # any clip; for typical 12-18 s beats with 0.3 s fades this is a
    # no-op (clamp doesn't fire).
    cap = max(0.0, duration * 0.4)
    fi = max(0.0, min(float(fade_in_seconds), cap))
    fo = max(0.0, min(float(fade_out_seconds), cap))

    fade_chain = ""
    if fi > 0:
        fade_chain += f"fade=t=in:st=0:d={fi:.3f},"
    if fo > 0:
        fade_chain += f"fade=t=out:st={max(0.0, duration - fo):.3f}:d={fo:.3f},"

    vf = (
        f"scale={OVERSCAN_W}:{OVERSCAN_H}:force_original_aspect_ratio=increase,"
        f"crop={OVERSCAN_W}:{OVERSCAN_H},"
        f"setsar=1,"
        f"scale={KEN_BURNS_SUPERSAMPLE_W}:-2:flags=lanczos,"
        f"zoompan="
        f"z='{z_expr}':"
        f"x='{x_expr}':"
        f"y='{y_expr}':"
        f"d=1:s={OUT_W}x{OUT_H}:fps={FPS},"
        f"{fade_chain}"
        f"format=yuv420p"
    )

    cmd = [
        require_ffmpeg(), "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-an",
        str(out_path),
    ]
    _run(cmd)


# -------------------- concat --------------------

def concat_clips(clip_paths: list[Path], audio_path: Path, out_path: Path) -> None:
    """Concat pre-rendered clips and mux with final audio mix.

    Validates every input clip BEFORE invoking ffmpeg — the concat
    demuxer silently truncates at the first unreadable input, which
    would yield a final.mp4 that's the-first-N-clips-long without
    anyone noticing. We refuse to proceed on any ffprobe failure.
    """
    bad: list[tuple[Path, str]] = []
    for p in clip_paths:
        if not p.exists():
            bad.append((p, "missing"))
            continue
        try:
            dur = get_duration_seconds(p)
            if dur <= 0:
                bad.append((p, f"duration={dur}s"))
        except Exception as e:
            bad.append((p, str(e)[:80]))
    if bad:
        details = "; ".join(f"{p.name} ({why})" for p, why in bad[:5])
        raise RuntimeError(
            f"concat refused: {len(bad)} input clip(s) unreadable/invalid: "
            f"{details}. Delete the bad file(s) and re-run S12."
        )

    concat_file = out_path.parent / "concat.txt"
    with concat_file.open("w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        require_ffmpeg(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-c:a", "aac",
        "-b:a", "320k",
        "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",
        str(out_path),
    ]
    _run(cmd)


# -------------------- audio post --------------------

@dataclass
class AudioMixSpec:
    voice_wav: Path
    music_wavs: list[Path]
    out_wav: Path
    voice_gain_db: float = -18.0
    music_gain_db: float = -28.0
    ambient_gain_db: float = -30.0
    duck_depth_db: float = 6.0
    duck_attack_ms: int = 100
    duck_release_ms: int = 800
    lufs_target: float = -14.0
    true_peak_dbtp: float = -1.0
    lra: float = 11.0
    voice_dynaudnorm_enabled: bool = True
    voice_dynaudnorm_framelen_ms: int = 200
    voice_dynaudnorm_gauss: int = 11
    voice_dynaudnorm_max_gain: float = 15.0
    music_start_offset_seconds: float = 0.0


def audio_post_mix(spec: AudioMixSpec) -> None:
    """Mix voice with a music bed, sidechain duck the music to the
    voice, then EBU R128 loudnorm to YouTube spec (-14 LUFS / -1 dBTP).

    Uses two-pass loudnorm so the final file lands within ~0.1 LUFS of
    target; falls back to one-pass dynamic if the analysis JSON can't
    be parsed.

    Unlike the maritime version this module has NO SFX path — the
    business-stories pipeline does not generate or layer SFX.
    """
    inputs: list[str] = ["-i", str(spec.voice_wav)]
    for m in spec.music_wavs:
        inputs += ["-i", str(m)]
    n_music = len(spec.music_wavs)

    voice_filters: list[str] = []
    if spec.voice_dynaudnorm_enabled:
        voice_filters.append(
            f"dynaudnorm=f={spec.voice_dynaudnorm_framelen_ms}"
            f":g={spec.voice_dynaudnorm_gauss}"
            f":m={spec.voice_dynaudnorm_max_gain}"
        )
    voice_filters.append(f"volume={spec.voice_gain_db}dB")
    voice_filter_chain = ",".join(voice_filters)

    needs_voice_split = n_music >= 1
    if needs_voice_split:
        fc_parts = [f"[0:a]{voice_filter_chain},asplit=2[v_dry][v_side]"]
    else:
        fc_parts = [f"[0:a]{voice_filter_chain}[v_dry]"]

    music_delay_ms = int(max(0.0, spec.music_start_offset_seconds) * 1000)
    music_labels: list[str] = []
    for i in range(n_music):
        gain = spec.music_gain_db if i == 0 else spec.ambient_gain_db
        chain = []
        if music_delay_ms > 0:
            chain.append(f"adelay={music_delay_ms}|{music_delay_ms}")
        chain.append(f"volume={gain}dB")
        fc_parts.append(f"[{i+1}:a]{','.join(chain)}[m{i}_raw]")

    if n_music >= 1:
        threshold = 0.05
        ratio = max(1.5, 10 ** (spec.duck_depth_db / 20))
        fc_parts.append(
            f"[m0_raw][v_side]sidechaincompress="
            f"threshold={threshold}:ratio={ratio}:"
            f"attack={spec.duck_attack_ms}:release={spec.duck_release_ms}[m0]"
        )
        music_labels.append("[m0]")

    for i in range(1, n_music):
        music_labels.append(f"[m{i}_raw]")

    mix_inputs_list = ["[v_dry]"] + music_labels
    if len(mix_inputs_list) == 1:
        # Voice-only: skip amix — amix with inputs=1 deadlocks ffmpeg.
        fc_parts.append("[v_dry]anull[mixed]")
    else:
        mix_inputs = "".join(mix_inputs_list)
        fc_parts.append(
            f"{mix_inputs}amix=inputs={len(mix_inputs_list)}"
            f":duration=first:dropout_transition=0:normalize=0[mixed]"
        )
    filter_complex_mix = ";".join(fc_parts)

    # ---- Pass 0: mix to temp file (no loudnorm yet) ----
    temp_mix = spec.out_wav.parent / "_premix_for_loudnorm.wav"
    _run([require_ffmpeg(), "-y", *inputs,
          "-filter_complex", filter_complex_mix,
          "-map", "[mixed]",
          "-ar", "48000", "-ac", "2",
          "-c:a", "pcm_s24le",
          str(temp_mix)])

    # ---- Pass 1: analyze ----
    analyze_cmd = [
        require_ffmpeg(), "-nostats", "-i", str(temp_mix),
        "-af", (f"loudnorm=I={spec.lufs_target}:TP={spec.true_peak_dbtp}"
                f":LRA={spec.lra}:print_format=json"),
        "-f", "null", "-",
    ]
    proc = subprocess.run(
        analyze_cmd, capture_output=True, text=True,
        stdin=subprocess.DEVNULL,
    )
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr or "", re.DOTALL)
    measured: dict | None = None
    if m:
        try:
            measured = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.warning("loudnorm analysis JSON parse failed: %s", e)

    # ---- Pass 2: apply, with measured values for ~0.1 LUFS precision ----
    # Guard against pathological inputs (digital silence, etc.) where
    # ffmpeg reports measured_I=-inf and the two-pass linear=true
    # branch errors out with "Result too large". When any required
    # measurement field is non-finite, fall back to one-pass.
    def _finite(v) -> bool:
        try:
            f = float(v)
            return f == f and f not in (float("inf"), float("-inf"))
        except (TypeError, ValueError):
            return False

    measured_ok = bool(measured) and all(
        _finite(measured.get(k))
        for k in ("input_i", "input_lra", "input_tp", "input_thresh")
    )

    if measured_ok:
        af = (
            f"loudnorm=I={spec.lufs_target}:TP={spec.true_peak_dbtp}"
            f":LRA={spec.lra}"
            f":measured_I={measured['input_i']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured.get('target_offset', 0.0)}"
            f":linear=true:print_format=summary"
        )
    else:
        if measured:
            logger.warning(
                "loudnorm analysis returned non-finite values "
                "(silent/empty input?); falling back to one-pass"
            )
        else:
            logger.warning("loudnorm analysis unavailable; falling back to one-pass")
        af = (f"loudnorm=I={spec.lufs_target}:TP={spec.true_peak_dbtp}"
              f":LRA={spec.lra}")

    _run([require_ffmpeg(), "-y", "-i", str(temp_mix),
          "-af", af,
          "-ar", "48000", "-ac", "2",
          "-c:a", "pcm_s24le",
          str(spec.out_wav)])

    try:
        temp_mix.unlink(missing_ok=True)
    except Exception:
        pass


def extract_audio(video_path: Path, out_path: Path) -> None:
    cmd = [require_ffmpeg(), "-y",
           "-i", str(video_path),
           "-vn", "-c:a", "aac", "-b:a", "192k",
           str(out_path)]
    _run(cmd)


# -------------------- music-bed concatenation --------------------

def concat_music_with_crossfade(
    music_paths: list[Path],
    out_path: Path,
    crossfade_seconds: float = 4.0,
) -> None:
    """Concatenate a list of music tracks with N-second crossfades.

    Used by S11 to glue the picked tracks from the local music library
    into a single bed that's then sidechain-ducked under the voice.

    For a single track, just copies it to out_path.
    """
    if not music_paths:
        raise ValueError("concat_music_with_crossfade: no tracks given")

    if len(music_paths) == 1:
        # Just resample/normalize to a known PCM format for the mix.
        cmd = [
            require_ffmpeg(), "-y",
            "-i", str(music_paths[0]),
            "-ar", "48000", "-ac", "2",
            "-c:a", "pcm_s24le",
            str(out_path),
        ]
        _run(cmd)
        return

    # Build a filter graph that crossfades each pair sequentially:
    #   [0][1] -> acrossfade -> [x1]
    #   [x1][2] -> acrossfade -> [x2]
    #   ...
    inputs: list[str] = []
    for p in music_paths:
        inputs += ["-i", str(p)]

    parts: list[str] = []
    prev_label = "[0:a]"
    for i in range(1, len(music_paths)):
        out_label = f"[x{i}]"
        parts.append(
            f"{prev_label}[{i}:a]acrossfade=d={crossfade_seconds}:c1=tri:c2=tri{out_label}"
        )
        prev_label = out_label

    filter_complex = ";".join(parts)

    cmd = [
        require_ffmpeg(), "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", prev_label,
        "-ar", "48000", "-ac", "2",
        "-c:a", "pcm_s24le",
        str(out_path),
    ]
    _run(cmd)


# -------------------- inspection --------------------

def probe(path: Path) -> dict:
    cmd = [require_ffprobe(),
           "-v", "error",
           "-print_format", "json",
           "-show_format", "-show_streams",
           str(path)]
    out = subprocess.run(
        cmd, capture_output=True, text=True, check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    return json.loads(out)


def get_duration_seconds(path: Path) -> float:
    info = probe(path)
    return float(info["format"]["duration"])


# -------------------- internals --------------------

def _run(cmd: list[str], timeout: float = 1800.0) -> None:
    """Run an ffmpeg/ffprobe command and raise on non-zero exit.

    Uses Popen + manual SIGTERM → SIGKILL escalation rather than the
    high-level subprocess.run(timeout=) because the latter has been
    observed not to actually terminate ffmpeg when it's wedged in a
    filter-graph scheduler deadlock.
    """
    logger.debug("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Explicit DEVNULL stdin. Same reason as pipeline/flux.py —
        # protects child interpreters/binaries from inheriting a
        # broken parent fd 0 in cron / long-running daemon contexts.
        stdin=subprocess.DEVNULL,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out after %.0fs: %s", timeout, " ".join(cmd[:6]))
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg ignored SIGTERM; sending SIGKILL")
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("ffmpeg refused to die even on SIGKILL")
                stdout, stderr = "", ""
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    if proc.returncode != 0:
        logger.error("ffmpeg stderr:\n%s", (stderr or "")[-2000:])
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=stdout, stderr=stderr
        )
