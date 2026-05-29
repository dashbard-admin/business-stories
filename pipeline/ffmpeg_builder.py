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
    # Optional SFX track — pre-rendered by S11 Phase 2 with all
    # cue clips placed at beat-start offsets and gain already
    # baked in. Mixed in alongside voice + music. Added Batch C
    # 2026-05-26.
    sfx_wav: Path | None = None
    sfx_gain_db: float = 0.0   # SFX gain already baked into the
                                # rendered sfx_wav, so no extra
                                # attenuation here by default.


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

    # Optional SFX track (Batch C 2026-05-26). Always added LAST to
    # the input list so its index is predictable. Gain is baked into
    # the file by S11; we apply spec.sfx_gain_db as a final trim
    # (default 0 == no extra trim).
    has_sfx = spec.sfx_wav is not None and Path(spec.sfx_wav).exists()
    if has_sfx:
        inputs += ["-i", str(spec.sfx_wav)]

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

    # SFX label — input index is n_music + 1 (voice is index 0).
    sfx_labels: list[str] = []
    if has_sfx:
        sfx_idx = n_music + 1
        fc_parts.append(
            f"[{sfx_idx}:a]volume={spec.sfx_gain_db}dB[sfx]"
        )
        sfx_labels.append("[sfx]")

    mix_inputs_list = ["[v_dry]"] + music_labels + sfx_labels
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


def composite_callouts_onto_clip(
    src_clip: Path,
    out_clip: Path,
    callouts: list[dict],
    *,
    callout_cfg: dict,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> bool:
    """Overlay one or more callout text PNGs onto an existing clip.

    Each callout dict: `{text, offset_seconds, hold_seconds (optional),
    fade_ms (optional)}`. Renders the text as a transparent PNG via
    Pillow (yellow body, black stroke), then uses ffmpeg's overlay
    filter with timed visibility and pre-applied fade.

    Returns True on success, False on failure (in which case S12
    keeps the original clip and logs a warning). Added Batch C
    2026-05-26.
    """
    # Batch L 2026-05-29: every `return False` now carries a specific
    # warning so the operator can tell from `logs/orch.<date>.log`
    # WHICH failure mode fired. Pre-Batch-L the only signal was S12's
    # "callout compositing failed for BEAT_NN; using raw clip" — that
    # told you "something went wrong" but not whether the data was
    # missing (S08 emitted nothing), the font failed, the ffmpeg run
    # crashed, or the output was 0-byte.
    if not callouts:
        logger.warning(
            "composite_callouts: %s — empty callouts list "
            "(S08 found no markers for this beat)",
            src_clip.name,
        )
        return False
    from PIL import Image, ImageDraw, ImageFont

    try:
        # Default styling from callout_cfg.
        font_pct = float(callout_cfg.get("font_size_pct", 0.10))
        color = callout_cfg.get("color", "#FFE600")
        stroke_color = callout_cfg.get("stroke_color", "#000000")
        stroke_width = int(callout_cfg.get("stroke_width", 6))
        default_hold = float(callout_cfg.get("hold_seconds", 2.5))
        default_fade_ms = int(callout_cfg.get("fade_ms", 200))
        font_size_px = int(frame_height * font_pct)

        # Pillow font — try a few common system fonts (mac, linux).
        font = None
        font_path_used: str | None = None
        for candidate in (
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ):
            try:
                font = ImageFont.truetype(candidate, font_size_px)
                font_path_used = candidate
                break
            except (OSError, IOError):
                continue
        if font is None:
            # Batch L 2026-05-29: log when we fall through to PIL's
            # default font — at typical callout sizes (~108 px) the
            # default font renders as a TINY bitmap glyph that's
            # easy to miss in a 1080p frame, which would look exactly
            # like "no overlay at all".
            logger.warning(
                "composite_callouts: %s — Pillow font candidate list "
                "exhausted; falling back to ImageFont.load_default() "
                "(callouts will render at the default bitmap size, "
                "~10 px high, and will be nearly invisible at 1080p)",
                src_clip.name,
            )
            font = ImageFont.load_default()

        variants = {
            "comic_pop_lower": {
                "fill": color, "stroke": stroke_color, "stroke_width": stroke_width,
                "background": None, "text_fill": color,
                "position": "lower_center", "motion": "pop",
            },
            "stamp_red_angle": {
                "fill": "#FFFFFF", "stroke": "#B00020", "stroke_width": 7,
                "background": "#B00020", "text_fill": "#FFFFFF",
                "position": "upper_right", "motion": "stamp", "rotate": -5,
            },
            "ticker_slide_left": {
                "fill": "#FFE600", "stroke": "#000000", "stroke_width": 4,
                "background": "#050505", "text_fill": "#FFE600",
                "position": "lower_left", "motion": "slide_left",
            },
            "paper_strip_typeon": {
                "fill": "#111111", "stroke": "#F8F1D2", "stroke_width": 3,
                "background": "#F8F1D2", "text_fill": "#111111",
                "position": "upper_left", "motion": "slide_up", "rotate": 2,
            },
            "money_pulse": {
                "fill": "#7CFF6B", "stroke": "#00280B", "stroke_width": 6,
                "background": None, "text_fill": "#7CFF6B",
                "position": "lower_center", "motion": "pulse",
            },
            "corner_badge": {
                "fill": "#111111", "stroke": "#FFE600", "stroke_width": 4,
                "background": "#FFE600", "text_fill": "#111111",
                "position": "bottom_right", "motion": "slide_up",
            },
        }

        # Render each callout to its own PNG.
        overlay_pngs: list[tuple[Path, dict]] = []
        tmp_dir = out_clip.parent / "_callouts_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(callouts):
            text = (c.get("text") or "").strip()
            if not text:
                continue
            variant_name = str(c.get("variant") or "comic_pop_lower")
            variant = variants.get(variant_name, variants["comic_pop_lower"])
            # Measure
            local_stroke = int(variant.get("stroke_width", stroke_width))
            bbox = font.getbbox(text, stroke_width=local_stroke)
            pad_x = int(font_size_px * (0.18 if variant.get("background") else 0.04))
            pad_y = int(font_size_px * (0.12 if variant.get("background") else 0.04))
            tw = (bbox[2] - bbox[0]) + 2 * (local_stroke + pad_x)
            th = (bbox[3] - bbox[1]) + 2 * (local_stroke + pad_y)
            img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            if variant.get("background"):
                d.rounded_rectangle(
                    [0, 0, tw - 1, th - 1],
                    radius=max(4, int(th * 0.12)),
                    fill=variant["background"],
                    outline=variant.get("stroke", stroke_color),
                    width=max(2, local_stroke // 2),
                )
            d.text(
                (-bbox[0] + local_stroke + pad_x,
                 -bbox[1] + local_stroke + pad_y),
                text, font=font, fill=variant.get("text_fill", color),
                stroke_width=local_stroke if not variant.get("background") else 0,
                stroke_fill=variant.get("stroke", stroke_color),
            )
            rotate = float(variant.get("rotate", 0.0))
            if rotate:
                img = img.rotate(rotate, expand=True, resample=Image.Resampling.BICUBIC)
            png_path = tmp_dir / f"callout_{i}.png"
            img.save(png_path, "PNG")
            overlay_pngs.append((png_path, {**c, "_variant": variant}))

        if not overlay_pngs:
            logger.warning(
                "composite_callouts: %s — every callout text was empty "
                "after strip; nothing to overlay (input was %d items)",
                src_clip.name, len(callouts),
            )
            return False

        # Build the filter graph: chain overlays. PNG inputs must be
        # looped so the fade filter has frames across the whole
        # callout window; otherwise ffmpeg sees a single transparent
        # alpha-fade frame at t=0 and the overlay looks invisible even
        # though the command succeeds.
        inputs: list[str] = ["-i", str(src_clip)]
        for png_path, _ in overlay_pngs:
            inputs += ["-loop", "1", "-i", str(png_path)]

        fc_parts: list[str] = []
        prev_label = "[0:v]"
        for i, (_, c) in enumerate(overlay_pngs):
            start = float(c.get("offset_seconds", 0.0))
            hold = float(c.get("hold_seconds", default_hold))
            fade_s = float(c.get("fade_ms", default_fade_ms)) / 1000.0
            end = start + hold

            # Pre-fade the overlay PNG. Use overlay_in / overlay_out
            # via the fade filter with alpha=1 so transparency is
            # respected.
            ovl_in = f"[{i+1}:v]"
            ovl_lbl = f"[ovl{i}]"
            variant = c.get("_variant") or {}
            motion = str(variant.get("motion") or "fade")
            scale_expr = "1"
            if motion == "pulse":
                scale_expr = "1+0.035*sin(10*t)"
            elif motion in {"pop", "stamp"}:
                scale_expr = "1+0.045*sin(16*t)*exp(-2.6*t)"
            ovl_filters = [f"{ovl_in}format=rgba"]
            if scale_expr != "1":
                ovl_filters.append(
                    f"scale=w=iw*({scale_expr}):h=ih*({scale_expr}):eval=frame"
                )
            ovl_filters.append(
                f"fade=t=in:st={start:.3f}:d={fade_s:.3f}:alpha=1"
            )
            ovl_filters.append(
                f"fade=t=out:st={max(start, end - fade_s):.3f}"
                f":d={fade_s:.3f}:alpha=1{ovl_lbl}"
            )
            fc_parts.append(
                ",".join(ovl_filters)
            )

            # Overlay onto previous layer, centered horizontally,
            # near the lower-third of frame (vertical hot-spot).
            # shortest=1 keeps the looped PNG input from extending
            # output beyond the finite source clip.
            next_label = (
                "[vout]" if i == len(overlay_pngs) - 1 else f"[v{i+1}]"
            )
            position = str(variant.get("position") or "lower_center")
            if position == "upper_left":
                x_expr = "W*0.06"
                y_expr = "H*0.12"
            elif position == "upper_right":
                x_expr = "W-w-W*0.06"
                y_expr = "H*0.12"
            elif position == "lower_left":
                x_expr = "W*0.06"
                y_expr = "H*0.70"
            elif position == "bottom_right":
                x_expr = "W-w-W*0.06"
                y_expr = "H-h-H*0.12"
            else:
                x_expr = "(W-w)/2"
                y_expr = "H*0.62"
            if motion == "slide_left":
                x_expr = f"({x_expr})-80*exp(-8*(t-{start:.3f}))"
            elif motion == "slide_up":
                y_expr = f"({y_expr})+70*exp(-8*(t-{start:.3f}))"
            fc_parts.append(
                f"{prev_label}{ovl_lbl}overlay="
                f"x='{x_expr}':y='{y_expr}':"
                f"enable='between(t,{start:.3f},{end:.3f})'"
                f":shortest=1"
                f"{next_label}"
            )
            prev_label = next_label

        fc = ";".join(fc_parts)
        cmd = [
            require_ffmpeg(), "-y", *inputs,
            "-filter_complex", fc,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-an",
            str(out_clip),
        ]
        _run(cmd)

        # Cleanup tmp PNGs.
        for png_path, _ in overlay_pngs:
            try:
                png_path.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        if not (out_clip.exists() and out_clip.stat().st_size > 1000):
            logger.warning(
                "composite_callouts: %s — ffmpeg returned cleanly but "
                "%s is missing or < 1000 bytes",
                src_clip.name, out_clip.name,
            )
            return False
        # Batch L 2026-05-29: a success log so the operator can
        # confirm in the daily log that the overlay path actually
        # ran on the expected beats.
        logger.info(
            "composite_callouts: %s ← %d overlays via %s "
            "(font=%s @ %d px, variants=%s)",
            out_clip.name, len(overlay_pngs), src_clip.name,
            font_path_used or "PIL_default", font_size_px,
            ",".join(str(c.get("variant") or "comic_pop_lower")
                     for _p, c in overlay_pngs),
        )
        return True
    except Exception as e:
        logger.warning(
            "composite_callouts: %s — unhandled exception: %s",
            src_clip.name, e,
        )
        return False


def render_sfx_track(
    cues: list[dict],
    out_path: Path,
    *,
    total_seconds: float,
    sample_rate: int = 48000,
) -> bool:
    """Render an SFX bed track of `total_seconds` length with each
    SFX clip placed at its `start_seconds` offset.

    `cues` is a list of `{path: Path, start_seconds: float,
    gain_db: float}` dicts. Each clip is loaded, delayed via
    ffmpeg's `adelay` filter, gain-trimmed via `volume`, then all
    of them mixed into a single stereo track via `amix`.

    Returns True on success, False on failure. Empty `cues` is a
    no-op (returns False without rendering).

    Added Batch C 2026-05-26 for S11 Phase 2.
    """
    if not cues:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    fc_parts: list[str] = []
    mix_labels: list[str] = []

    # Anchor input: silence of total_seconds, so the output has the
    # expected duration even if no SFX fires at the very end.
    anchor_idx = 0
    inputs += [
        "-f", "lavfi", "-t", f"{total_seconds:.3f}",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
    ]
    fc_parts.append(f"[{anchor_idx}:a]anull[sfx_anchor]")
    mix_labels.append("[sfx_anchor]")

    for ci, cue in enumerate(cues):
        idx = ci + 1
        path = cue["path"]
        start_ms = int(max(0.0, float(cue["start_seconds"])) * 1000)
        gain_db = float(cue.get("gain_db", 0.0))
        inputs += ["-i", str(path)]
        chain = []
        if start_ms > 0:
            chain.append(f"adelay={start_ms}|{start_ms}:all=1")
        chain.append(f"volume={gain_db}dB")
        chain.append("aformat=channel_layouts=stereo:sample_rates="
                     + str(sample_rate))
        fc_parts.append(f"[{idx}:a]{','.join(chain)}[sfx{ci}]")
        mix_labels.append(f"[sfx{ci}]")

    n = len(mix_labels)
    fc_parts.append(
        f"{''.join(mix_labels)}amix=inputs={n}:duration=first"
        f":dropout_transition=0:normalize=0[sfx_out]"
    )
    fc = ";".join(fc_parts)

    cmd = [
        require_ffmpeg(), "-y", *inputs,
        "-filter_complex", fc,
        "-map", "[sfx_out]",
        "-ar", str(sample_rate), "-ac", "2",
        "-c:a", "pcm_s24le",
        str(out_path),
    ]
    try:
        _run(cmd)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        logger.warning("render_sfx_track failed: %s", e)
        return False


def extract_audio(video_path: Path, out_path: Path) -> None:
    cmd = [require_ffmpeg(), "-y",
           "-i", str(video_path),
           "-vn", "-c:a", "aac", "-b:a", "192k",
           str(out_path)]
    _run(cmd)


# -------------------- voice padding (Batch J 2026-05-29) --------------------

def pad_audio_silence(
    src: Path,
    dst: Path,
    head_seconds: float,
    tail_seconds: float,
) -> None:
    """Pad an audio file with silence at the head and/or tail.

    Used by S11 so that final_mix.wav covers the full video timeline
    (title card + voice + closing card) instead of just the voice
    duration. Without this, S12's `-shortest` mux truncates the
    output at audio length and the closing source-attribution card
    never makes it into the rendered MP4.

    head_seconds and tail_seconds are clamped to 0 if negative.
    When both are 0, the source file is copied through (re-encoded
    to PCM_S16LE so downstream filters see a known format).
    """
    head = max(0.0, float(head_seconds))
    tail = max(0.0, float(tail_seconds))

    filters: list[str] = []
    if head > 0:
        head_ms = int(round(head * 1000))
        # `adelay=N:all=1` prepends N ms of silence to every channel.
        filters.append(f"adelay={head_ms}:all=1")
    if tail > 0:
        # `apad=pad_dur=N` appends N seconds of silence.
        filters.append(f"apad=pad_dur={tail:.3f}")

    cmd: list[str] = [
        require_ffmpeg(), "-y",
        "-i", str(src),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += [
        "-c:a", "pcm_s16le",
        "-ar", "24000",  # match Kokoro's sample rate; concat downstream
        str(dst),
    ]
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
