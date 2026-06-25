"""S09 — FLUX Rendering with VLM-driven QA.

For every beat with no PD asset, render via the FLUX CLI with the
locked style profile applied, then submit the result to the VLM
(Qwen3-VL) for a quality verdict. Rejected images are re-rolled
with a new seed up to `image_qa.max_attempts_per_beat` times. The
highest-scoring attempt is kept even if every attempt is rejected.

In addition to the per-beat renders, this stage produces:
  - 03_assets/flux/title.png      — the title-card panel
  - 03_assets/flux/credits.png    — the closing source-attribution panel
                                      only when closing card is enabled

These use the locked visual style and a panel description derived
from the incident metadata (company + hero + conflict).

Idempotency mirrors maritime S9:
  - pass/borderline verdict on an existing file → skip (true no-op)
  - existing file but no verdict → run QA only (no re-render)
  - else: render-and-judge loop from scratch.

Inputs:  02_script/beat_sheet.json
Outputs: 03_assets/flux/BEAT_NN.png  +  image_qa per beat in beat_sheet.json
         03_assets/flux/title.png    +  optional 03_assets/flux/credits.png
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import yaml
import hashlib
from pathlib import Path

from ..config import load_config
from ..flux import Flux, FluxRequest, compute_seed
from ..grok import Grok
from ..state import find_episode_workspace
from ..vlm import VLM, ImageVerdict

logger = logging.getLogger("hermes.stage.s09")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _is_good_enough(verdict: ImageVerdict, strict_borderline: bool) -> bool:
    if verdict.verdict == "pass":
        return True
    if verdict.verdict == "borderline":
        if not strict_borderline:
            return True
        return (not verdict.artifacts) and verdict.anatomy_ok
    return False


def _image_backend(cfg) -> str:
    raw = (cfg.raw.get("image_generation") or {}).get("backend", "both")
    backend = str(raw or "both").strip().lower()
    aliases = {
        "auto": "both",
        "fallback": "both",
        "flux_only": "flux",
        "only_flux": "flux",
        "grok_only": "grok",
        "only_grok": "grok",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"both", "flux", "grok"}:
        logger.warning(
            "S09: unknown image_generation.backend=%r; using 'both'",
            raw,
        )
        return "both"
    return backend


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    image_backend = _image_backend(cfg)
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
    logger.info("S09: image backend=%s, image QA %s, max_attempts_per_beat=%d, strict_borderline=%s",
                image_backend,
                "enabled" if qa_enabled else "disabled",
                max_attempts, strict_borderline)

    # Grok text-correction sub-phase. Lazy: only spun up if available;
    # otherwise the per-beat loop skips correction transparently.
    grok = Grok()
    grok_prompt_template: str | None = None
    grok_dir = ws / "03_assets" / "grok"
    grok_needed = image_backend in {"both", "grok"}
    if grok_needed and grok.available:
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
    elif grok_needed:
        logger.info("S09: Grok text-correction disabled (%s)",
                    grok.unavailability_reason())
        if image_backend == "grok":
            return f"image_generation.backend=grok but Grok is unavailable: {grok.unavailability_reason()}"
    else:
        logger.info("S09: Grok disabled by image_generation.backend=flux")

    # ----- title + credits cards (rendered first so S12 can pick them up
    # even if the per-beat loop bails mid-way) -----
    _render_title_card(
        ws=ws, episode=episode, cfg=cfg, flux=flux, flux_dir=flux_dir,
        image_backend=image_backend, grok=grok,
        grok_prompt_template=grok_prompt_template,
    )
    prod_cfg = cfg.production
    closing_card_enabled = bool(prod_cfg.get("closing_card_enabled", True))
    closing_card_seconds = float(prod_cfg.get("closing_card_seconds", 0))
    if closing_card_enabled and closing_card_seconds > 0:
        _render_credits_card(
            ws=ws, episode=episode, cfg=cfg, flux=flux, flux_dir=flux_dir,
            manifest=manifest, image_backend=image_backend, grok=grok,
            grok_prompt_template=grok_prompt_template,
        )
    else:
        logger.info("S09 credits card render skipped (closing card disabled)")

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
        prompt_hash = _prompt_hash(fr["prompt"])

        existing_qa = b.get("image_qa") or {}
        existing_verdict = (existing_qa.get("verdict") or "").lower()
        has_image = out_path.exists() and out_path.stat().st_size > 100
        stored_prompt_hash = existing_qa.get("prompt_hash")
        prompt_changed = bool(stored_prompt_hash and stored_prompt_hash != prompt_hash)
        existing_provider = (existing_qa.get("image_backend") or "").lower()
        provider_changed = (
            image_backend in {"flux", "grok"}
            and (existing_provider or "flux") != image_backend
        )

        if has_image and existing_verdict in ("pass", "borderline", "unjudged"):
            if not prompt_changed and not provider_changed:
                if b.get("flux_asset_path") != str(out_path.relative_to(ws)):
                    b["flux_asset_path"] = str(out_path.relative_to(ws))
                    _persist()
                continue
            logger.info(
                "S09 %s prompt/provider changed since existing render; re-rendering",
                beat_id,
            )

        if has_image and not existing_qa and b.get("visual_prompt_version"):
            logger.info(
                "S09 %s has existing pixels but no QA for storyboard prompt; "
                "re-rendering",
                beat_id,
            )
        elif has_image and not existing_qa and vlm:
            verdict = vlm.critique_image(out_path, fr["prompt"])
            if verdict:
                _log_verdict(beat_id, "existing", 0, 0, verdict)
            if verdict and _is_good_enough(verdict, strict_borderline):
                b["flux_asset_path"] = str(out_path.relative_to(ws))
                b["image_qa"] = {
                    **verdict.to_dict(),
                    "attempts": 0,
                    "seed": compute_seed(fr["prompt"], seed_offset=0),
                    "prompt_hash": prompt_hash,
                    "visual_prompt_version": b.get("visual_prompt_version"),
                }
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
                    "prompt_hash": prompt_hash,
                    "visual_prompt_version": b.get("visual_prompt_version"),
                }
                _persist()
                continue

        # Render-and-judge loop
        attempts: list[tuple[Path, ImageVerdict | None, int]] = []
        passing: tuple[Path, ImageVerdict | None, int] | None = None
        selected_backend = "grok" if image_backend == "grok" else "flux"

        for attempt in range(max_attempts):
            seed_offset = attempt * 1000
            seed_used = compute_seed(fr["prompt"], seed_offset=seed_offset)
            attempt_target = flux_dir / f"{beat_id}_a{attempt}.png"
            if image_backend == "grok":
                chosen = _render_prompt_with_grok(
                    grok=grok,
                    prompt_template=grok_prompt_template,
                    flux_prompt=fr["prompt"],
                    out_path=attempt_target,
                    label=beat_id,
                )
                seed_used = 0
            else:
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
                logger.warning(
                    "S09 %s %s render attempt %d failed",
                    beat_id, image_backend, attempt + 1,
                )
                continue

            verdict = vlm.critique_image(chosen, fr["prompt"]) if vlm else None
            attempts.append((chosen, verdict, seed_used))

            if verdict is None:
                passing = (chosen, None, seed_used)
                break

            _log_verdict(beat_id, "a", attempt + 1, seed_used, verdict)

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

        # ----- Grok regeneration sub-phase -----
        # If the VLM verdict on the winning image flags EITHER
        # malformed/illegible text artifacts OR anatomical issues
        # (anatomy_ok=False), AND the Grok adapter is available,
        # ask Grok to regenerate the panel from the FLUX prompt
        # (text-to-image). The Grok output replaces chosen_path;
        # both the raw FLUX render and the Grok output are archived
        # to 03_assets/grok/<beat_id>_*.png for visual comparison.
        grok_correction_info: dict | None = None
        if image_backend == "both" and grok_prompt_template and grok.available and chosen_verdict:
            # Two independent gate predicates. Either is enough to
            # route to Grok. Both checked here in the orchestrator
            # so the routing decision is auditable from one log line.
            text_triggered, text_triggers = _has_malformed_text(chosen_verdict)
            anatomy_bad = not chosen_verdict.anatomy_ok
            triggers: list[str] = list(text_triggers)
            if anatomy_bad:
                anat_msg = "anatomy_ok=False"
                if chosen_verdict.reasoning:
                    anat_msg = (
                        f"{anat_msg}: {chosen_verdict.reasoning[:160]}"
                    )
                # Anatomy goes to the front of the triggers list so
                # it's visible first in the log + the audit field.
                triggers.insert(0, anat_msg)
            if text_triggered or anatomy_bad:
                grok_correction_info = _correct_text_via_grok(
                    src=chosen_path,
                    beat_id=beat_id,
                    flux_prompt=fr["prompt"],
                    verdict=chosen_verdict,
                    grok=grok,
                    grok_dir=grok_dir,
                    prompt_template=grok_prompt_template,
                    triggers=triggers,
                )
                if grok_correction_info and grok_correction_info.get("corrected_path"):
                    # Promote the corrected image to be the canonical
                    # chosen path. The original FLUX render lives on at
                    # grok_correction_info["original_archive_path"].
                    chosen_path = Path(grok_correction_info["corrected_path"])
                    selected_backend = "grok"
                    logger.info("S09 %s: Grok corrected (triggers=%s)",
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
            "prompt_hash": prompt_hash,
            "image_backend": selected_backend,
            "visual_prompt_version": b.get("visual_prompt_version"),
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

    # Visual brand-safety pass (Batch F 2026-05-28). Runs the VLM
    # over every rendered beat to catch scene/topic mismatches that
    # the per-image critique can't see — e.g. a hacking-coded panel
    # in a non-crime story (the Pets.com review video had exactly
    # this), or era-anachronistic styling (a 1940s noir detective
    # figure in a 1999 dot-com bust). Output:
    # 03_assets/visual_brand_safety_flags.json. High-severity flags
    # gate the stage (mark needs_human) unless disabled in config.
    return _run_visual_brand_safety_pass(
        episode=episode, ws=ws, beats=beats, cfg=cfg, vlm=vlm,
    )


# ---------------- VLM verdict logging ----------------

def _log_verdict(
    beat_id: str,
    label: str,
    attempt_num: int,
    seed: int,
    verdict: ImageVerdict,
) -> None:
    """Emit the VLM's full assessment of one image attempt.

    Three lines per attempt:
      - one-line summary  (verdict + score + match + anatomy)
      - artifacts list    (only if non-empty)
      - reasoning prose   (only if non-empty)

    `label` distinguishes call sites in the log:
      - "a"        — per-attempt render in the main loop
      - "existing" — VLM judging an on-disk image without re-rendering

    Splitting onto three lines makes grep-by-purpose trivial:
        grep "artifacts:"   logs/orch.YYYY-MM-DD.log
        grep "reasoning:"   logs/orch.YYYY-MM-DD.log
    and keeps the summary line short enough to scan in a tail.
    """
    if label == "a":
        head = f"S09 {beat_id} a{attempt_num} (seed={seed})"
    else:
        head = f"S09 {beat_id} {label}"
    logger.info(
        "%s: verdict=%s score=%d match=%d anatomy=%s",
        head, verdict.verdict, verdict.score, verdict.prompt_match,
        "ok" if verdict.anatomy_ok else "BAD",
    )
    if verdict.artifacts:
        logger.info("%s artifacts: %s", head, verdict.artifacts)
    if verdict.reasoning:
        logger.info("%s reasoning: %s", head, verdict.reasoning)


# ---------------- Grok text-correction helpers ----------------

def _render_prompt_with_grok(
    *,
    grok: Grok,
    prompt_template: str | None,
    flux_prompt: str,
    out_path: Path,
    label: str,
) -> Path | None:
    """Render one prompt with Grok, using the same prompt wrapper as fallback."""
    if not grok.available:
        logger.warning("S09 %s: Grok unavailable (%s)", label, grok.unavailability_reason())
        return None
    if not prompt_template:
        logger.warning("S09 %s: Grok prompt template unavailable", label)
        return None
    grok_prompt = prompt_template.format(flux_prompt=flux_prompt)
    result = grok.regenerate_from_prompt(prompt=grok_prompt, out_path=out_path)
    if result and out_path.exists():
        return out_path
    if _grok_failure_wants_sanitized_retry(grok):
        safe_flux_prompt = _sanitize_prompt_for_grok_moderation(flux_prompt)
        if safe_flux_prompt != flux_prompt:
            logger.warning(
                "S09 %s: Grok moderation rejected prompt; retrying sanitized prompt",
                label,
            )
            safe_prompt = prompt_template.format(flux_prompt=safe_flux_prompt)
            result = grok.regenerate_from_prompt(
                prompt=safe_prompt, out_path=out_path,
            )
            if result and out_path.exists():
                return out_path
    return None


def _grok_failure_wants_sanitized_retry(grok: Grok) -> bool:
    body = (getattr(grok, "last_error_text", "") or "").lower()
    status = getattr(grok, "last_status_code", None)
    return status == 400 and (
        "content moderation" in body
        or "rejected" in body
        or "invalid argument" in body
    )


def _sanitize_prompt_for_grok_moderation(prompt: str) -> str:
    """Remove brand/IP and sensitive wording that xAI image moderation
    often rejects while preserving the broad movie-shot composition."""
    text = str(prompt or "")
    text = re.sub(
        r"Use only these selected continuity props in frame:.*?\.",
        "Use only generic story-specific props in frame.",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"Avoid recently used continuity props from the previous slide:.*?\.",
        "Avoid unrelated props from the previous slide.",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"Beat narration context:.*?Specific composition:",
        "Beat narration context: a non-branded business history moment. Specific composition:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    replacements = {
        r"\bNintendo\b": "the company",
        r"\bNintendo['’]s\b": "the company's",
        r"\bDonkey Kong\b": "a hit arcade game",
        r"\bGame Boy\b": "a handheld game device",
        r"\bTetris\b": "a falling-block puzzle game",
        r"\bMario\b": "a mascot character",
        r"\bPok[eé]mon\b": "a collectible creature game",
        r"\bFamicom\b": "a home game console",
        r"\bNES\b": "a home game console",
        r"\bSwitch\b": "a modern game console",
        r"\bWii\b": "a motion-control game console",
        r"\bHanafuda\b": "traditional illustrated playing cards",
        r"\bTake-Two Interactive\b": "a partner company",
        r"\bSeattle Mariners\b": "a professional sports team",
        r"\bSilicon Graphics\b": "a technology partner",
        r"\bYamauchi\b": "the executive",
        r"\bHiroshi\b": "the executive",
        r"\bgambling\b": "regulated play",
        r"\bgamble\b": "play",
        r"\bbetting\b": "wager-free play",
        r"\bbet\b": "choice",
        r"\bwater guns?\b": "water toys",
        r"\bguns?\b": "toy devices",
        r"\bfire\b": "sudden pressure",
        r"\bexplodes?\b": "grows quickly",
        r"\bban(?:s|ned|ning)?\b": "restriction",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    suffix = (
        " Keep it generic: no real logos, no copyrighted characters, "
        "no trademarked product likenesses, no readable text."
    )
    if "Keep it generic:" not in text:
        text = f"{text}.{suffix}"
    return text

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
    " illegible", " illegibility",
    # Phrases the VLM actually uses for "no text" directive
    # violations (observed in production):
    "no text", "no-text", "text directive", "text legibility",
    "text is present", "text is visible", "text is legible",
    "text legible", "real text", "real handwritten text",
    "text violation", "text violations",
)


# Word-boundary regex used by the fallback rule below. Matches the
# standalone word "text" but NOT embedded occurrences like "texture",
# "context", or "subtext".
_TEXT_WORD_RE = re.compile(r"\btext\b", re.IGNORECASE)


def _has_malformed_text(verdict: ImageVerdict) -> tuple[bool, list[str]]:
    """Scan the VLM verdict for malformed-text indicators.

    Returns (triggered, matching_phrases). `triggered` is True when:
      (a) any keyword in _MALFORMED_TEXT_KEYWORDS appears as a
          substring of an artifact entry or of verdict.reasoning, OR
      (b) the verdict is "borderline" or "reject" AND the standalone
          word "text" appears anywhere in artifacts/reasoning. Because
          the writer prompt prepends an explicit "no text" directive
          to every FLUX render, the VLM only ever mentions "text" in
          a non-pass verdict to complain that the directive was
          violated. This is the safety-net rule for cases where the
          VLM's phrasing doesn't match an explicit keyword (e.g. "Text
          is present despite prompt's 'no text' directive").

    The matching phrases are returned so they can be logged and
    stamped into image_qa for audit.
    """
    matches: list[str] = []
    haystack: list[tuple[str, str]] = []
    for a in (verdict.artifacts or []):
        haystack.append(("artifact", str(a)))
    if verdict.reasoning:
        haystack.append(("reasoning", verdict.reasoning))

    # Rule (a) — explicit keyword hits.
    for source, text in haystack:
        low = text.lower()
        for kw in _MALFORMED_TEXT_KEYWORDS:
            if kw in low:
                matches.append(f"{source}: {text[:160]}")
                break
    if matches:
        return (True, matches)

    # Rule (b) — word-boundary "text" fallback for non-pass verdicts.
    if (verdict.verdict or "").lower() in ("borderline", "reject"):
        for source, text in haystack:
            if _TEXT_WORD_RE.search(text):
                matches.append(f"{source} [text mentioned]: {text[:160]}")
                break  # one fallback match is enough

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
    triggers: list[str],
) -> dict | None:
    """Archive the FLUX render, ask Grok to regenerate from the same
    prompt (text-to-image — no image upload, no edit semantics), and
    archive the Grok output. Returns a metadata dict for the caller
    to stamp into image_qa, or None on any failure.

    `triggers` is the list of strings describing WHY the orchestrator
    routed to Grok (combined text + anatomy triggers, computed in
    the caller). Passed in so the gate decision and the function's
    logging stay in sync — earlier versions re-checked
    `_has_malformed_text` inside the function and silently no-oped
    when anatomy-only triggered the route.

    Output filenames (under grok_dir = 03_assets/grok/):
      - <beat_id>_flux_original.png  — pre-correction FLUX render
                                       (archived for visual comparison)
      - <beat_id>_grok_corrected.png — Grok regeneration (also becomes
                                       the new canonical FLUX image at
                                       03_assets/flux/<beat_id>.png)

    On failure (Grok API error, malformed response) returns None —
    the caller keeps the uncorrected FLUX render.
    """
    if not triggers:
        return None
    logger.info("S09 %s: VLM flagged issues — routing to Grok. "
                "triggers: %s", beat_id, triggers[:3])

    grok_dir.mkdir(parents=True, exist_ok=True)
    original_archive = grok_dir / f"{beat_id}_flux_original.png"
    corrected_archive = grok_dir / f"{beat_id}_grok_corrected.png"

    # Archive the FLUX original for side-by-side comparison. shutil.copy
    # keeps the src in place so the caller's subsequent
    # shutil.move(chosen_path, out_path) still works if the Grok call
    # fails and we fall back.
    try:
        import shutil as _sh
        _sh.copy(str(src), str(original_archive))
    except Exception as e:
        logger.warning("S09 %s: failed to archive FLUX original "
                       "to %s: %s", beat_id, original_archive, e)
        return None

    # The Grok prompt template wraps the FLUX prompt. Default template
    # is just {flux_prompt} pass-through — operator-editable in
    # pipeline/prompts/grok_text_correction.txt.
    grok_prompt = prompt_template.format(flux_prompt=flux_prompt)

    # Text-to-image regeneration. We deliberately do NOT send the FLUX
    # image as a reference — image-edit produced low-quality results
    # in practice. A fresh generation from the same prompt lets Grok
    # render text legibly without inheriting FLUX's artifacts.
    result = grok.regenerate_from_prompt(
        prompt=grok_prompt, out_path=corrected_archive,
    )
    if result is None or not corrected_archive.exists():
        return None

    return {
        "triggering_artifacts": triggers,
        "original_archive_path": str(original_archive),
        "corrected_archive_path": str(corrected_archive),
        "corrected_path": str(corrected_archive),
    }


def _rerender_single_beat_with_grok(
    *,
    target: dict,
    beat_sheet: dict,
    beat_sheet_path: Path,
    ws: Path,
    beat_id: str,
    fr: dict,
    out_path: Path,
    grok_dir: Path,
    ts: str,
) -> bool:
    """Force a one-beat Grok render and promote it to the FLUX path."""
    cfg = load_config()
    grok = Grok()
    if not grok.available:
        logger.warning(
            "S09 force-grok: Grok unavailable for %s (%s)",
            beat_id, grok.unavailability_reason(),
        )
        return False

    template_path = cfg.prompts_dir / "grok_text_correction.txt"
    try:
        prompt_template = template_path.read_text()
    except Exception as e:
        logger.warning(
            "S09 force-grok: prompt template unreadable for %s: %s",
            beat_id, e,
        )
        return False

    grok_dir.mkdir(parents=True, exist_ok=True)
    forced_archive = grok_dir / f"{beat_id}_grok_forced_{ts}.png"
    grok_prompt = prompt_template.format(flux_prompt=fr["prompt"])
    logger.info("S09 force-grok: rendering %s with Grok", beat_id)
    result = grok.regenerate_from_prompt(
        prompt=grok_prompt,
        out_path=forced_archive,
    )
    if not result or not forced_archive.exists():
        logger.warning("S09 force-grok: Grok failed for %s", beat_id)
        return False

    try:
        if out_path.exists():
            out_path.unlink()
        shutil.copy(str(forced_archive), str(out_path))
    except Exception as e:
        logger.warning("S09 force-grok: could not promote %s: %s", beat_id, e)
        return False

    target["flux_asset_path"] = str(out_path.relative_to(ws))
    target["image_qa"] = {
        "verdict": "unjudged",
        "reasoning": "operator forced Grok rerender",
        "rerendered_at": ts,
        "prompt_hash": _prompt_hash(fr["prompt"]),
        "image_backend": "grok",
        "visual_prompt_version": target.get("visual_prompt_version"),
        "grok_correction": {
            "applied": True,
            "forced": True,
            "triggering_artifacts": ["operator --force-grok"],
            "corrected_archive": str(forced_archive),
        },
    }
    beat_sheet_path.write_text(json.dumps(beat_sheet, indent=2))
    logger.info("S09 force-grok complete: %s -> %s",
                forced_archive.name, out_path.name)
    return True


# ---------------- title + credits cards ----------------

def _panel_backend_matches(out_path: Path, image_backend: str) -> bool:
    meta_path = out_path.with_suffix(out_path.suffix + ".backend")
    if not meta_path.exists():
        return image_backend in {"both", "flux"}
    return meta_path.read_text().strip().lower() == image_backend


def _write_panel_backend(out_path: Path, image_backend: str) -> None:
    try:
        out_path.with_suffix(out_path.suffix + ".backend").write_text(image_backend)
    except Exception:
        pass

def _render_title_card(
    *,
    ws: Path,
    episode: dict,
    cfg,
    flux: Flux,
    flux_dir: Path,
    image_backend: str,
    grok: Grok,
    grok_prompt_template: str | None,
) -> None:
    """Render the opening title-card panel. Composed prompt uses the
    visual-style prefix + a description derived from the incident's
    hero/conflict so the panel evokes the episode's tension."""
    out_path = flux_dir / "title.png"
    if (
        out_path.exists()
        and out_path.stat().st_size > 1000
        and _panel_backend_matches(out_path, image_backend)
    ):
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

    if image_backend == "grok":
        chosen = _render_prompt_with_grok(
            grok=grok,
            prompt_template=grok_prompt_template,
            flux_prompt=f"{composed}. Avoid: {negative}.",
            out_path=out_path,
            label="title",
        )
    else:
        req = FluxRequest(
            beat_id="title",
            prompt=composed,
            negative_prompt=negative,
            out_path=out_path,
        )
        chosen = flux.render_batch_with_retry(req, num_candidates=1, seed_offset=42)
    if chosen and chosen.exists():
        _write_panel_backend(out_path, image_backend)
        logger.info("S09 title card rendered: %s", chosen)
    else:
        logger.warning("S09 title card render failed; S12 will fall back to ffmpeg-drawn card")


def _render_credits_card(
    *,
    ws: Path,
    episode: dict,
    cfg,
    flux: Flux,
    flux_dir: Path,
    manifest: dict,
    image_backend: str,
    grok: Grok,
    grok_prompt_template: str | None,
) -> None:
    """Render the closing credits/source-attribution backdrop. The
    actual attribution TEXT is composited by S12 over this panel —
    the FLUX panel is the backdrop only."""
    out_path = flux_dir / "credits.png"
    if (
        out_path.exists()
        and out_path.stat().st_size > 1000
        and _panel_backend_matches(out_path, image_backend)
    ):
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

    if image_backend == "grok":
        chosen = _render_prompt_with_grok(
            grok=grok,
            prompt_template=grok_prompt_template,
            flux_prompt=f"{composed}. Avoid: {negative}.",
            out_path=out_path,
            label="credits",
        )
    else:
        req = FluxRequest(
            beat_id="credits",
            prompt=composed,
            negative_prompt=negative,
            out_path=out_path,
        )
        chosen = flux.render_batch_with_retry(req, num_candidates=1, seed_offset=999)
    if chosen and chosen.exists():
        _write_panel_backend(out_path, image_backend)
        logger.info("S09 credits card rendered: %s", chosen)
    else:
        logger.warning("S09 credits card render failed; S12 will fall back to ffmpeg-drawn card")


# ----------------------------------------------------------------------
# Per-beat re-render entry point (Batch B 2026-05-26)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Visual brand-safety pass (Batch F 2026-05-28)
# ----------------------------------------------------------------------

def _run_visual_brand_safety_pass(
    *,
    episode: dict,
    ws,
    beats: list,
    cfg,
    vlm,
) -> str | None:
    """For every rendered beat, ask the VLM whether the panel is
    brand-safe and topic-coherent for the episode's company / era /
    story_kind. Emits 03_assets/visual_brand_safety_flags.json with
    structured flags. Returns a needs_human reason if any high-
    severity flag fires AND cfg.visual_brand_safety.gate_on_severity
    is "high" (default). Operator clears via --approve.
    """
    vbs_cfg = cfg.visual_brand_safety
    out_path = ws / "03_assets" / "visual_brand_safety_flags.json"

    if not vbs_cfg.get("enabled", True):
        out_path.write_text(json.dumps({
            "verdict": "skipped",
            "reason": "visual_brand_safety.enabled=false",
            "flags": [],
            "high_severity_count": 0,
            "low_severity_count": 0,
        }, indent=2))
        return None

    if vlm is None:
        logger.info("S09 visual brand-safety: VLM disabled; skipping")
        return None

    incident = episode.get("incident") or {}
    company_name = incident.get("company_name", "")
    story_kind = incident.get("story_kind", "")
    year_anchor = incident.get("year_anchor", "")
    one_line_pitch = incident.get("one_line_pitch", "")

    sample_every = max(1, int(vbs_cfg.get("sample_every_n", 1)))
    flags: list[dict] = []
    high = 0
    low = 0
    checked = 0
    for i, b in enumerate(beats):
        # Sample every Nth beat to keep VLM cost bounded — Pets.com
        # video had 60+ beats so a per-beat pass adds 60 VLM calls.
        # sample_every=2 halves it without losing the per-pattern
        # signal we care about.
        if i % sample_every != 0:
            continue
        rel = b.get("flux_asset_path")
        if not rel:
            continue
        img_path = ws / rel
        if not img_path.exists():
            continue
        result = vlm.brand_safety_check(
            img_path,
            company_name=company_name,
            story_kind=story_kind,
            year_anchor=year_anchor,
            one_line_pitch=one_line_pitch,
        )
        checked += 1
        if result is None:
            continue
        severity = result.get("severity", "clean")
        if severity == "clean":
            continue
        flag = {
            "beat_id": b.get("beat_id", ""),
            "image_path": rel,
            "severity": severity,
            "category": result.get("category", ""),
            "issue": result.get("issue", ""),
            "suggestion": result.get("suggestion", ""),
        }
        flags.append(flag)
        if severity == "high":
            high += 1
        elif severity == "low":
            low += 1
        logger.info(
            "S09 visual-safety [%s]: %s — %s",
            severity, b.get("beat_id", "?"),
            (result.get("issue") or "")[:120],
        )

    out_path.write_text(json.dumps({
        "verdict": ("ship_blocker" if high > 0 else
                    "review_recommended" if low > 0 else "clean"),
        "high_severity_count": high,
        "low_severity_count": low,
        "checked_count": checked,
        "flags": flags,
    }, indent=2))
    logger.info(
        "S09 visual brand-safety: %d high, %d low, %d clean (of %d checked)",
        high, low, checked - high - low, checked,
    )

    # Stamp counts on the episode record for --status surfacing.
    # The dict is passed by reference; the orchestrator's
    # save_queue() call after dispatch_stage picks up the mutation.
    episode["visual_safety_flags_count"] = {"high": high, "low": low}

    # Apply the gate.
    gate = (vbs_cfg.get("gate_on_severity") or "high").lower()
    if gate == "off":
        return None
    if gate == "high" and high > 0:
        return (
            f"visual brand-safety: {high} high-severity flag(s) "
            f"require review. Inspect 03_assets/"
            f"visual_brand_safety_flags.json then run `--approve "
            f"{episode['id']}` to clear (or `--rerender` the "
            f"specific beats first)."
        )
    if gate == "low" and (high > 0 or low > 0):
        return (
            f"visual brand-safety: {high}H + {low}L flag(s) "
            f"(gate_on_severity=low). Inspect "
            f"03_assets/visual_brand_safety_flags.json then "
            f"`--approve {episode['id']}` to clear."
        )
    return None


def rerender_single_beat(
    episode: dict,
    beat_id: str,
    *,
    from_edited_prompt: bool = False,
    force_grok: bool = False,
) -> bool:
    """Re-run FLUX render + VLM judge + Grok fallback for a single
    beat. Called by hermes_orchestrator's --rerender CLI flow.

    `from_edited_prompt=True` re-reads the beat's
    `flux_render_request.prompt` fresh from beat_sheet.json on disk
    so an operator edit takes effect. When False, re-uses whatever
    prompt was in the in-memory beat record (which is also read
    from disk, just unmodified) — semantically equivalent here
    since we don't cache; the flag exists to make the operator's
    intent explicit.

    `force_grok=True` bypasses FLUX entirely and asks Grok to render
    the beat from the same prompt. Otherwise the configured
    `image_generation.backend` policy is used. The chosen output is
    promoted to the canonical 03_assets/flux/<beat_id>.png path so
    downstream S12 continues to read the usual asset field.

    Archives the existing render (and any Grok-corrected version)
    to 03_assets/quarantine/<beat_id>_<timestamp>.png before
    re-rendering, so nothing gets silently overwritten without a
    paper trail.

    Returns True on success, False on failure. Raises
    FileNotFoundError if the workspace or beat isn't found.
    """
    from datetime import datetime, timezone

    cfg = load_config()
    image_backend = "grok" if force_grok else _image_backend(cfg)
    ws = find_episode_workspace(episode["id"])
    if not ws:
        raise FileNotFoundError(
            f"no workspace for episode {episode['id']}"
        )

    beat_sheet_path = ws / "02_script" / "beat_sheet.json"
    if not beat_sheet_path.exists():
        raise FileNotFoundError(
            f"no beat_sheet.json at {beat_sheet_path}"
        )
    beat_sheet = json.loads(beat_sheet_path.read_text())
    beats = beat_sheet.get("beats", [])

    target = next((b for b in beats if b.get("beat_id") == beat_id), None)
    if target is None:
        raise FileNotFoundError(
            f"beat {beat_id} not found in beat_sheet.json"
        )
    if "flux_render_request" not in target:
        raise FileNotFoundError(
            f"beat {beat_id} has no flux_render_request "
            f"(routed to PD asset, not FLUX)"
        )

    flux_dir = ws / "03_assets" / "flux"
    grok_dir = ws / "03_assets" / "grok"
    quarantine_dir = ws / "03_assets" / "quarantine"
    flux_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Archive any existing renders.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for src_dir, suffix in [(flux_dir, "flux"), (grok_dir, "grok")]:
        existing = src_dir / f"{beat_id}.png"
        if existing.exists():
            archived = quarantine_dir / f"{beat_id}.{suffix}.{ts}.png"
            existing.rename(archived)
            logger.info("S09 rerender: archived %s -> %s",
                        existing.name, archived.name)

    # Build single-beat render request and run the same FLUX +
    # judge + Grok cycle. To keep this scope tight, we reuse the
    # existing helpers but inline a minimal version of the per-beat
    # loop from run() — the loop in run() is iterative across beats
    # and not currently factorable without a refactor we don't want
    # to do in Batch B.
    flux = Flux()
    fr = target["flux_render_request"]
    if from_edited_prompt:
        logger.info("S09 rerender: re-reading prompt from beat_sheet.json "
                    "for %s (edited-prompt path)", beat_id)

    out_path = flux_dir / f"{beat_id}.png"
    if image_backend == "grok":
        return _rerender_single_beat_with_grok(
            target=target,
            beat_sheet=beat_sheet,
            beat_sheet_path=beat_sheet_path,
            ws=ws,
            beat_id=beat_id,
            fr=fr,
            out_path=out_path,
            grok_dir=grok_dir,
            ts=ts,
        )

    qa_enabled = bool(cfg.image_qa.get("enabled", True))
    max_attempts = max(1, int(cfg.image_qa.get("max_attempts_per_beat", 2)))
    strict_borderline = bool(cfg.image_qa.get("strict_borderline", True))
    vlm: VLM | None = VLM() if qa_enabled else None

    attempts: list[tuple[Path, ImageVerdict | None]] = []
    chosen_path: Path | None = None
    chosen_verdict: ImageVerdict | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_path = flux_dir / f"{beat_id}_rerender_a{attempt}.png"
        req = FluxRequest(
            beat_id=beat_id,
            prompt=fr["prompt"],
            negative_prompt=fr.get("negative_prompt", ""),
            out_path=attempt_path,
        )
        candidate = flux.render_batch_with_retry(
            req, num_candidates=1, seed_offset=hash(beat_id) % 10000 + attempt,
        )
        if not candidate or not candidate.exists():
            logger.warning("S09 rerender: FLUX failed attempt %d for %s",
                           attempt, beat_id)
            continue
        if not vlm:
            chosen_path = candidate
            break
        verdict = vlm.critique_image(candidate, fr["prompt"])
        if verdict:
            _log_verdict(beat_id, "rerender", attempt, max_attempts, verdict)
        else:
            chosen_path = candidate
            logger.warning(
                "S09 rerender: VLM unavailable for %s attempt %d; "
                "keeping unjudged render",
                beat_id, attempt,
            )
            break
        if verdict and _is_good_enough(verdict, strict_borderline):
            chosen_path = candidate
            chosen_verdict = verdict
            break
        attempts.append((candidate, verdict))

    if chosen_path is None:
        if attempts:
            attempts.sort(
                key=lambda x: (x[1].score if x[1] else 5),
                reverse=True,
            )
            chosen_path, chosen_verdict = attempts[0]
            logger.warning(
                "S09 rerender: no acceptable render for %s after %d "
                "attempt(s); keeping best rejected attempt",
                beat_id, max_attempts,
            )
        else:
            logger.warning("S09 rerender: no render for %s after %d attempt(s)",
                           beat_id, max_attempts)
            return False

    # Grok fallback if anatomy/text issues remain.
    grok = Grok()
    selected_backend = "flux"
    if image_backend == "both" and grok.available and chosen_verdict:
        text_triggered, text_triggers = _has_malformed_text(chosen_verdict)
        anatomy_bad = not chosen_verdict.anatomy_ok
        triggers: list[str] = list(text_triggers)
        if anatomy_bad:
            anat_msg = "anatomy_ok=False"
            if chosen_verdict.reasoning:
                anat_msg = f"{anat_msg}: {chosen_verdict.reasoning[:160]}"
            triggers.insert(0, anat_msg)
        if text_triggered or anatomy_bad:
            grok_template_path = cfg.prompts_dir / "grok_text_correction.txt"
            if grok_template_path.exists():
                grok_template = grok_template_path.read_text()
                grok_correction_info = _correct_text_via_grok(
                    src=chosen_path, beat_id=beat_id,
                    flux_prompt=fr["prompt"], verdict=chosen_verdict,
                    grok=grok, grok_dir=grok_dir,
                    prompt_template=grok_template, triggers=triggers,
                )
                if grok_correction_info and grok_correction_info.get("corrected_path"):
                    corrected_path = Path(grok_correction_info["corrected_path"])
                    if corrected_path.exists():
                        if out_path.exists() and corrected_path != out_path:
                            out_path.unlink()
                        if corrected_path != out_path:
                            shutil.copy(str(corrected_path), str(out_path))
                        chosen_path = out_path
                        selected_backend = "grok"
                        logger.info(
                            "S09 rerender: promoted Grok correction for %s",
                            beat_id,
                        )

    if chosen_path != out_path:
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(chosen_path), str(out_path))
        chosen_path = out_path

    for attempt_path, _verdict in attempts:
        if attempt_path != chosen_path and attempt_path.exists():
            try:
                attempt_path.unlink()
            except OSError:
                pass
    for attempt_path in flux_dir.glob(f"{beat_id}_rerender_a*.png"):
        if attempt_path != chosen_path and attempt_path.exists():
            try:
                attempt_path.unlink()
            except OSError:
                pass

    # Update the beat record + persist.
    target["flux_asset_path"] = str(chosen_path.relative_to(ws))
    if chosen_verdict:
        target["image_qa"] = {
            "verdict": chosen_verdict.verdict,
            "score": chosen_verdict.score,
            "prompt_match": chosen_verdict.prompt_match,
            "anatomy_ok": chosen_verdict.anatomy_ok,
            "artifacts": chosen_verdict.artifacts,
            "reasoning": chosen_verdict.reasoning,
            "rerendered_at": ts,
            "image_backend": selected_backend,
        }
    else:
        target["image_qa"] = {
            "verdict": "unjudged",
            "reasoning": "VLM unavailable",
            "rerendered_at": ts,
            "image_backend": selected_backend,
        }
    beat_sheet_path.write_text(json.dumps(beat_sheet, indent=2))
    logger.info("S09 rerender complete: %s", chosen_path.name)
    return True
