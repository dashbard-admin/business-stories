"""S09 — FLUX Rendering with VLM-driven QA.

For every beat with no PD asset, render via the FLUX CLI with the
locked style profile applied, then submit the result to the VLM
(Qwen3-VL) for a quality verdict. Rejected images are re-rolled
with a new seed up to `image_qa.max_attempts_per_beat` times. The
highest-scoring attempt is kept even if every attempt is rejected.

In addition to the per-beat renders, this stage produces:
  - 03_assets/flux/title.png      — the title-card panel
  - 03_assets/flux/credits.png    — the closing source-attribution panel

Both use the locked visual style and a panel description derived from
the incident metadata (company + hero + conflict).

Idempotency mirrors maritime S9:
  - pass/borderline verdict on an existing file → skip (true no-op)
  - existing file but no verdict → run QA only (no re-render)
  - else: render-and-judge loop from scratch.

Inputs:  02_script/beat_sheet.json
Outputs: 03_assets/flux/BEAT_NN.png  +  image_qa per beat in beat_sheet.json
         03_assets/flux/title.png    +  03_assets/flux/credits.png
"""

from __future__ import annotations

import json
import logging
import shutil
import yaml
from pathlib import Path

from ..config import load_config
from ..flux import Flux, FluxRequest, compute_seed
from ..grok import Grok
from ..state import find_episode_workspace
from ..vlm import VLM, ImageVerdict

logger = logging.getLogger("hermes.stage.s09")


def _is_good_enough(verdict: ImageVerdict, strict_borderline: bool) -> bool:
    if verdict.verdict == "pass":
        return True
    if verdict.verdict == "borderline":
        if not strict_borderline:
            return True
        return (not verdict.artifacts) and verdict.anatomy_ok
    return False


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    flux = Flux()
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    beat_sheet_path = ws / "02_script" / "beat_sheet.json"
    if not beat_sheet_path.exists():
        return "no beat_sheet.json"
    beat_sheet = json.loads(beat_sheet_path.read_text())
    beats = beat_sheet["beats"]

    flux_dir = ws / "03_assets" / "flux"
    flux_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ws / "03_assets" / "asset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"pd_assets": []}
    manifest.setdefault("flux_assets", [])

    qa_enabled = bool(cfg.image_qa.get("enabled", True))
    max_attempts = max(1, int(cfg.image_qa.get("max_attempts_per_beat", 2)))
    strict_borderline = bool(cfg.image_qa.get("strict_borderline", True))
    vlm: VLM | None = VLM() if qa_enabled else None
    logger.info("S09: image QA %s, max_attempts_per_beat=%d, strict_borderline=%s",
                "enabled" if qa_enabled else "disabled",
                max_attempts, strict_borderline)

    # Grok text-correction sub-phase. Lazy: only spun up if available;
    # otherwise the per-beat loop skips correction transparently.
    grok = Grok()
    grok_prompt_template: str | None = None
    grok_dir = ws / "03_assets" / "grok"
    if grok.available:
        try:
            grok_prompt_template = (
                cfg.prompts_dir / "grok_text_correction.txt"
            ).read_text()
            grok_dir.mkdir(parents=True, exist_ok=True)
            logger.info("S09: Grok text-correction enabled "
                        "(model=%s, endpoint=%s%s)",
                        grok.model, grok.base_url, grok.endpoint_path)
        except Exception as e:
            logger.warning("S09: Grok prompt template unreadable "
                           "(%s); correction disabled", e)
            grok_prompt_template = None
    else:
        logger.info("S09: Grok text-correction disabled (%s)",
                    grok.unavailability_reason())

    # ----- title + credits cards (rendered first so S12 can pick them up
    # even if the per-beat loop bails mid-way) -----
    _render_title_card(ws=ws, episode=episode, cfg=cfg, flux=flux, flux_dir=flux_dir)
    _render_credits_card(ws=ws, episode=episode, cfg=cfg, flux=flux, flux_dir=flux_dir,
                         manifest=manifest)

    rendered = 0
    failed = 0
    rejected_accepted = 0

    def _persist() -> None:
        beat_sheet_path.write_text(json.dumps(beat_sheet, indent=2))
        manifest_path.write_text(json.dumps(manifest, indent=2))

    for b in beats:
        if "flux_render_request" not in b:
            continue
        beat_id = b["beat_id"]
        out_path = flux_dir / f"{beat_id}.png"
        fr = b["flux_render_request"]

        existing_qa = b.get("image_qa") or {}
        existing_verdict = (existing_qa.get("verdict") or "").lower()
        has_image = out_path.exists() and out_path.stat().st_size > 100

        if has_image and existing_verdict in ("pass", "borderline", "unjudged"):
            if b.get("flux_asset_path") != str(out_path.relative_to(ws)):
                b["flux_asset_path"] = str(out_path.relative_to(ws))
                _persist()
            continue

        if has_image and not existing_qa and vlm:
            verdict = vlm.critique_image(out_path, fr["prompt"])
            if verdict and _is_good_enough(verdict, strict_borderline):
                b["flux_asset_path"] = str(out_path.relative_to(ws))
                b["image_qa"] = {
                    **verdict.to_dict(),
                    "attempts": 0,
                    "seed": compute_seed(fr["prompt"], seed_offset=0),
                }
                logger.info("S09 %s existing image QA: %s (score=%d match=%d)",
                            beat_id, verdict.verdict, verdict.score, verdict.prompt_match)
                _persist()
                continue
            if verdict:
                logger.info("S09 %s existing image %s; re-rendering", beat_id, verdict.verdict)
            elif verdict is None:
                b["flux_asset_path"] = str(out_path.relative_to(ws))
                b["image_qa"] = {
                    "verdict": "unjudged", "attempts": 0,
                    "reasoning": "VLM unavailable",
                    "seed": compute_seed(fr["prompt"], seed_offset=0),
                }
                _persist()
                continue

        # Render-and-judge loop
        attempts: list[tuple[Path, ImageVerdict | None, int]] = []
        passing: tuple[Path, ImageVerdict | None, int] | None = None

        for attempt in range(max_attempts):
            seed_offset = attempt * 1000
            seed_used = compute_seed(fr["prompt"], seed_offset=seed_offset)
            attempt_target = flux_dir / f"{beat_id}_a{attempt}.png"
            req = FluxRequest(
                beat_id=beat_id,
                prompt=fr["prompt"],
                negative_prompt=fr.get("negative_prompt", ""),
                out_path=attempt_target,
                reference_image_path=fr.get("reference_image_path"),
                reference_strength=float(fr.get("reference_strength", 0.5)),
            )
            chosen = flux.render_batch_with_retry(
                req, num_candidates=1, seed_offset=seed_offset,
            )
            if not chosen or not chosen.exists():
                logger.warning("S09 %s render attempt %d failed", beat_id, attempt + 1)
                continue

            verdict = vlm.critique_image(chosen, fr["prompt"]) if vlm else None
            attempts.append((chosen, verdict, seed_used))

            if verdict is None:
                passing = (chosen, None, seed_used)
                break

            logger.info("S09 %s a%d (seed=%d): verdict=%s score=%d match=%d",
                        beat_id, attempt + 1, seed_used, verdict.verdict,
                        verdict.score, verdict.prompt_match)

            if _is_good_enough(verdict, strict_borderline):
                passing = (chosen, verdict, seed_used)
                break

        chosen_path: Path | None = None
        chosen_verdict: ImageVerdict | None = None
        chosen_seed: int = 0
        if passing is not None:
            chosen_path, chosen_verdict, chosen_seed = passing
        elif attempts:
            attempts.sort(
                key=lambda x: (x[1].score if x[1] else 5),
                reverse=True,
            )
            chosen_path, chosen_verdict, chosen_seed = attempts[0]
            rejected_accepted += 1
            logger.warning("S09 %s: all %d attempts rejected; keeping best (seed=%d)",
                           beat_id, len(attempts), chosen_seed)

        if chosen_path is None:
            failed += 1
            logger.error("S09 %s: every render attempt failed", beat_id)
            continue

        # ----- Grok text-correction sub-phase -----
        # If the VLM verdict on the winning image flags malformed /
        # illegible text artifacts AND the Grok adapter is available,
        # send the image + the FLUX prompt to xAI Grok for a text-fix
        # pass. The corrected output replaces chosen_path; both the
        # raw FLUX render and the Grok output are archived to
        # 03_assets/grok/<beat_id>_*.png for visual comparison.
        grok_correction_info: dict | None = None
        if grok_prompt_template and grok.available and chosen_verdict:
            # _has_malformed_text returns a (bool, list) TUPLE. Earlier
            # versions used the tuple directly in the boolean gate,
            # which is always truthy in Python (any 2-item tuple is)
            # — that meant Grok was attempted on every beat with the
            # only "successful" no-op being _correct_text_via_grok's
            # internal recheck. We unpack explicitly to make the gate
            # check the actual bool.
            triggered, triggers = _has_malformed_text(chosen_verdict)
            if triggered:
                grok_correction_info = _correct_text_via_grok(
                    src=chosen_path,
                    beat_id=beat_id,
                    flux_prompt=fr["prompt"],
                    verdict=chosen_verdict,
                    grok=grok,
                    grok_dir=grok_dir,
                    prompt_template=grok_prompt_template,
                )
                if grok_correction_info and grok_correction_info.get("corrected_path"):
                    # Promote the corrected image to be the canonical
                    # chosen path. The original FLUX render lives on at
                    # grok_correction_info["original_archive_path"].
                    chosen_path = Path(grok_correction_info["corrected_path"])
                    logger.info("S09 %s: Grok corrected (artifacts=%s)",
                                beat_id, grok_correction_info.get("triggering_artifacts"))
                else:
                    logger.warning("S09 %s: Grok correction attempted but failed; "
                                   "keeping uncorrected FLUX render. "
                                   "triggers: %s", beat_id, triggers[:3])

        try:
            if out_path.exists():
                out_path.unlink()
            shutil.move(str(chosen_path), str(out_path))
        except Exception as e:
            logger.warning("S09 %s could not promote: %s", beat_id, e)
            continue
        for p, _v, _s in attempts:
            if p != chosen_path and p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

        b["flux_asset_path"] = str(out_path.relative_to(ws))
        b["image_qa"] = {
            "attempts": len(attempts),
            "seed": chosen_seed,
            **(chosen_verdict.to_dict()
               if chosen_verdict
               else {"verdict": "unjudged", "reasoning": "VLM unavailable"}),
        }
        if grok_correction_info:
            # Preserve original verdict but flag that Grok rewrote
            # the pixels. Operator can audit which beats this hit by
            # grepping image_qa.grok_correction in beat_sheet.json.
            b["image_qa"]["grok_correction"] = {
                "applied": True,
                "triggering_artifacts": grok_correction_info.get("triggering_artifacts"),
                "original_archive": grok_correction_info.get("original_archive_path"),
                "corrected_archive": grok_correction_info.get("corrected_archive_path"),
            }
        manifest["flux_assets"].append({
            "id": beat_id,
            "local_path": b["flux_asset_path"],
            "prompt": fr["prompt"],
        })
        rendered += 1
        _persist()

    _persist()

    logger.info("S09 complete: %d rendered (%d kept-from-rejected), %d failed",
                rendered, rejected_accepted, failed)
    if failed > 0 and failed > rendered * 0.1:
        return f"FLUX rendering failed for {failed}/{rendered + failed} beats"
    return None


# ---------------- Grok text-correction helpers ----------------

# Keyword set used to decide whether a VLM verdict implicates
# malformed/illegible/garbled text in the rendered image. Checked
# case-insensitively against verdict.artifacts entries AND the
# verdict.reasoning string. The list is intentionally generous —
# false positives cost one extra Grok call per beat, false negatives
# leave bad text in the final video.
_MALFORMED_TEXT_KEYWORDS = (
    "malformed text", "illegible text", "incomprehensible text",
    "garbled text", "gibberish", "glyph", "glyphic", "glyph soup",
    "unreadable", "scribble", "scribbled", "scribble-like",
    "fake text", "nonsense text", "jumbled text", "text artifact",
    "text artifacts", "broken text", "text is unreadable",
    "letters", "lettering", "alphabet", "word salad",
    "garbled writing", "garbled letters", "garbled signage",
    "garbled lettering", "unintelligible text",
)


def _has_malformed_text(verdict: ImageVerdict) -> tuple[bool, list[str]]:
    """Scan the VLM verdict for malformed-text indicators.

    Returns (triggered, matching_phrases). `triggered` is True when
    at least one keyword from _MALFORMED_TEXT_KEYWORDS appears in
    either an artifact entry or in verdict.reasoning. The matching
    phrases (artifacts strings or "reasoning: ...") are returned so
    they can be logged + stamped into image_qa for audit.
    """
    matches: list[str] = []
    haystack: list[tuple[str, str]] = []
    for a in (verdict.artifacts or []):
        haystack.append(("artifact", str(a)))
    if verdict.reasoning:
        haystack.append(("reasoning", verdict.reasoning))
    for source, text in haystack:
        low = text.lower()
        for kw in _MALFORMED_TEXT_KEYWORDS:
            if kw in low:
                matches.append(f"{source}: {text[:160]}")
                break  # one match per haystack entry is enough
    return (bool(matches), matches)


def _correct_text_via_grok(
    *,
    src: Path,
    beat_id: str,
    flux_prompt: str,
    verdict: ImageVerdict,
    grok: Grok,
    grok_dir: Path,
    prompt_template: str,
) -> dict | None:
    """Archive the FLUX render, call Grok with the FLUX prompt + image,
    archive the Grok output. Returns a metadata dict for the caller
    to stamp into image_qa, or None on any failure.

    Output filenames (under grok_dir = 03_assets/grok/):
      - <beat_id>_flux_original.png  — pre-correction FLUX render
      - <beat_id>_grok_corrected.png — Grok output (also becomes the
                                       new canonical FLUX image at
                                       03_assets/flux/<beat_id>.png)

    On failure (Grok API error, malformed response, decode failure)
    returns None — caller keeps the uncorrected FLUX render.
    """
    triggered, matches = _has_malformed_text(verdict)
    if not triggered:
        return None
    logger.info("S09 %s: VLM flagged malformed text — routing to Grok. "
                "triggers: %s", beat_id, matches[:3])

    grok_dir.mkdir(parents=True, exist_ok=True)
    original_archive = grok_dir / f"{beat_id}_flux_original.png"
    corrected_archive = grok_dir / f"{beat_id}_grok_corrected.png"

    # Archive the FLUX original. shutil.copy keeps the src in place
    # so the caller's subsequent shutil.move(chosen_path, out_path)
    # still works if the Grok call fails and we fall back.
    try:
        import shutil as _sh
        _sh.copy(str(src), str(original_archive))
    except Exception as e:
        logger.warning("S09 %s: failed to archive FLUX original "
                       "to %s: %s", beat_id, original_archive, e)
        return None

    # Compose the Grok prompt with the original FLUX prompt embedded.
    grok_prompt = prompt_template.format(flux_prompt=flux_prompt)

    # Call Grok.
    result = grok.correct_image(
        image_path=src, prompt=grok_prompt, out_path=corrected_archive,
    )
    if result is None or not corrected_archive.exists():
        return None

    return {
        "triggering_artifacts": matches,
        "original_archive_path": str(original_archive),
        "corrected_archive_path": str(corrected_archive),
        "corrected_path": str(corrected_archive),
    }


# ---------------- title + credits cards ----------------

def _render_title_card(*, ws: Path, episode: dict, cfg, flux: Flux,
                       flux_dir: Path) -> None:
    """Render the opening title-card panel. Composed prompt uses the
    visual-style prefix + a description derived from the incident's
    hero/conflict so the panel evokes the episode's tension."""
    out_path = flux_dir / "title.png"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return  # idempotent

    visual_style = episode["visual_style"]
    style_yaml = yaml.safe_load(
        (cfg.style_profiles_dir / f"{visual_style}.yaml").read_text()
    )
    incident = episode["incident"]
    company = incident["company_name"]
    hero = (incident.get("hero") or "").strip()
    conflict = (incident.get("conflict") or "").strip()
    story_kind = incident.get("story_kind", "")

    # Hand-built title-card prompt. The companion S6/S13-equivalent
    # title_generate.txt is NOT invoked here — we don't need a YouTube
    # title yet, just a visual.
    scene_hint = (
        f"a cinematic wide comic panel evoking the story of {company}: "
        f"{hero or 'the founder'} set against {conflict or 'their adversary'}. "
        f"Composition emphasizes the tension of the {story_kind or 'business'} arc. "
        f"NO text, NO logos with letters, the panel must work as pure imagery."
    )
    style_prefix = (style_yaml.get("prefix") or "").strip()
    style_suffix = (style_yaml.get("suffix") or "").strip()
    style_neg = (style_yaml.get("negative_prompt") or "").strip()

    composed = f"{style_prefix} {scene_hint}, {style_suffix}"
    NO_TEXT_NEGATIVE = (
        "text, letters, words, writing, captions, watermark, logo with letters, "
        "signature, signage with legible text, alphabet, numbers"
    )
    negative = f"{NO_TEXT_NEGATIVE}, {style_neg}"

    req = FluxRequest(
        beat_id="title",
        prompt=composed,
        negative_prompt=negative,
        out_path=out_path,
    )
    chosen = flux.render_batch_with_retry(req, num_candidates=1, seed_offset=42)
    if chosen and chosen.exists():
        logger.info("S09 title card rendered: %s", chosen)
    else:
        logger.warning("S09 title card render failed; S12 will fall back to ffmpeg-drawn card")


def _render_credits_card(*, ws: Path, episode: dict, cfg, flux: Flux,
                         flux_dir: Path, manifest: dict) -> None:
    """Render the closing credits/source-attribution backdrop. The
    actual attribution TEXT is composited by S12 over this panel —
    the FLUX panel is the backdrop only."""
    out_path = flux_dir / "credits.png"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return

    visual_style = episode["visual_style"]
    style_yaml = yaml.safe_load(
        (cfg.style_profiles_dir / f"{visual_style}.yaml").read_text()
    )

    # A neutral closing scene works for any company. Empty boardroom,
    # late evening, suggests the story has ended — apt for both rise
    # and fall arcs.
    scene_hint = (
        "a wide quiet comic panel: an empty modern office or boardroom "
        "at late evening, low warm light through tall windows, no people, "
        "no signage, no readable text. Composition leaves the lower third "
        "free of visual incident — attribution credits will be composited "
        "into that area at video assembly time."
    )
    style_prefix = (style_yaml.get("prefix") or "").strip()
    style_suffix = (style_yaml.get("suffix") or "").strip()
    style_neg = (style_yaml.get("negative_prompt") or "").strip()

    composed = f"{style_prefix} {scene_hint}, {style_suffix}"
    NO_TEXT_NEGATIVE = (
        "text, letters, words, writing, captions, watermark, logo with letters, "
        "signature, signage with legible text, alphabet, numbers"
    )
    negative = f"{NO_TEXT_NEGATIVE}, {style_neg}"

    req = FluxRequest(
        beat_id="credits",
        prompt=composed,
        negative_prompt=negative,
        out_path=out_path,
    )
    chosen = flux.render_batch_with_retry(req, num_candidates=1, seed_offset=999)
    if chosen and chosen.exists():
        logger.info("S09 credits card rendered: %s", chosen)
    else:
        logger.warning("S09 credits card render failed; S12 will fall back to ffmpeg-drawn card")
