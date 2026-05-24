"""S08 — Beat Sheet & Visual Plan.

Convert the BEAT markers in the script into a structured plan: each
beat gets a duration estimate, a visual intent, a Ken Burns motion,
and either a matched PD asset or a queued FLUX request.

Three asset-match passes (semantic similarity via sentence-transformers):
  Pass 1: direct PD use      sim ≥ pd_direct_use_threshold
  Pass 2: PD as reference    sim ≥ pd_reference_threshold
            (text grounding only when pd_image_reference_enabled=false;
             the FLUX CLI in this pipeline does NOT accept reference
             images, so img2img is always text-only here)
  Pass 2.5: generic stash    sim ≥ generic_stash.threshold

Inputs:  02_script/script.txt + 03_assets/asset_manifest.json
Outputs: 02_script/beat_sheet.json
"""

from __future__ import annotations

import json
import logging
import re

import yaml

from ..config import load_config
from ..llm import LLM
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s08")

WPM = 120  # anchored to script_generate.txt's hook-cadence math

BEAT_RE = re.compile(r"##\s*BEAT\s+(\d+)\s*##", re.IGNORECASE)


def _split_script_by_beats(script: str) -> dict[int, str]:
    matches = list(BEAT_RE.finditer(script))
    result: dict[int, str] = {}
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(script)
        result[num] = script[start:end].strip()
    return result


def _beat_id_to_int(beat_id: str) -> int | None:
    m = re.search(r"\d+", beat_id or "")
    return int(m.group(0)) if m else None


def _estimate_seconds(text: str) -> float:
    return (len(text.split()) / WPM) * 60.0


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    llm = LLM(role="writer")
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    script = (ws / "02_script" / "script.txt").read_text()
    manifest_path = ws / "03_assets" / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() \
        else {"pd_assets": []}
    pd_assets = manifest.get("pd_assets", [])

    visual_style = episode["visual_style"]
    style_yaml = yaml.safe_load(
        (cfg.style_profiles_dir / f"{visual_style}.yaml").read_text()
    )
    narrator = cfg.narrator_by_id(episode["narrator"])

    template = (cfg.prompts_dir / "beat_sheet.txt").read_text()
    prompt = template.format(
        visual_style_name=style_yaml["name"],
        visual_style_guidance=style_yaml.get("guidance_for_llm", ""),
        narrator_name=narrator["name"],
        script_with_beats=script,
    )

    try:
        beats = llm.complete_json(prompt, temperature=0.4, max_tokens=24000)
    except Exception as e:
        try:
            raw = llm.complete(prompt, temperature=0.4, max_tokens=24000)
            (ws / "02_script" / "beat_sheet_raw.txt").write_text(raw.text)
        except Exception:
            pass
        return f"beat-sheet generation failed: {e}"

    if not isinstance(beats, list) or not beats:
        return "beat-sheet output was empty or invalid"

    # Inject script_text from disk so the per-beat prose stays
    # authoritative (the LLM does not echo it back).
    beat_texts = _split_script_by_beats(script)
    for b in beats:
        if not isinstance(b, dict):
            continue
        if b.get("script_text"):
            continue
        num = _beat_id_to_int(b.get("beat_id", ""))
        if num is not None and num in beat_texts:
            b["script_text"] = beat_texts[num]

    for b in beats:
        b.setdefault("estimated_seconds", _estimate_seconds(b.get("script_text", "")))

    target_min = cfg.production["target_duration_seconds"] - cfg.production["duration_tolerance_seconds"]
    target_max = cfg.production["target_duration_seconds"] + cfg.production["duration_tolerance_seconds"]
    total = sum(b["estimated_seconds"] for b in beats)
    if total < target_min or total > target_max:
        logger.warning("beat total %.1fs outside %d-%d", total, target_min, target_max)

    # ----- asset matching -----
    sims = None
    direct_eligible: set[int] = set()
    stash_indices: set[int] = set()
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        beat_descs = [b.get("specific_visual_description", "") for b in beats]
        asset_descs = [
            (a.get("caption") or a.get("description") or a.get("title") or "")
            for a in pd_assets
        ]
        if asset_descs:
            beat_emb = st_model.encode(beat_descs, normalize_embeddings=True)
            asset_emb = st_model.encode(asset_descs, normalize_embeddings=True)
            import numpy as np
            sims = beat_emb @ asset_emb.T
            direct_eligible = {
                ai for ai, a in enumerate(pd_assets)
                if a.get("direct_use_eligible", True)
            }
            stash_indices = {
                ai for ai, a in enumerate(pd_assets) if a.get("is_generic_stash")
            }
    except Exception as e:
        logger.warning("sentence-transformers unavailable (%s); skipping asset match", e)

    iqa = cfg.image_qa
    SIM_THRESHOLD = float(iqa.get("pd_direct_use_threshold", 0.20))
    MAX_REUSES_PER_ASSET = int(iqa.get("pd_max_reuses_per_asset", 30))
    REFERENCE_THRESHOLD = float(iqa.get("pd_reference_threshold", 0.20))
    REFERENCE_STRENGTH = float(iqa.get("pd_reference_strength", 0.1))
    USE_IMAGE_REFERENCE = bool(iqa.get("pd_image_reference_enabled", False))
    asset_use_count: dict[int, int] = {}

    if sims is not None and len(pd_assets):
        import numpy as np
        # Pass 1: direct PD use
        for bi, b in enumerate(beats):
            row = sims[bi].copy()
            for ai, count in asset_use_count.items():
                if count >= MAX_REUSES_PER_ASSET:
                    row[ai] = -1.0
            for ai in range(len(pd_assets)):
                if ai not in direct_eligible:
                    row[ai] = -1.0
            for ai in stash_indices:
                row[ai] = -1.0
            if len(row) == 0:
                continue
            best = int(np.argmax(row))
            if row[best] >= SIM_THRESHOLD:
                b["pd_asset_id"] = pd_assets[best]["id"]
                b["pd_asset_path"] = pd_assets[best]["local_path"]
                asset_use_count[best] = asset_use_count.get(best, 0) + 1

        # Pass 2: loose-reference (text grounding into FLUX prompt)
        for bi, b in enumerate(beats):
            if "pd_asset_id" in b:
                continue
            row = sims[bi].copy()
            for ai in stash_indices:
                row[ai] = -1.0
            if len(row) == 0:
                continue
            best = int(np.argmax(row))
            if row[best] >= REFERENCE_THRESHOLD:
                b["reference_asset_id"] = pd_assets[best]["id"]
                b["reference_asset_path"] = pd_assets[best]["local_path"]
                b["reference_description"] = (
                    pd_assets[best].get("caption")
                    or pd_assets[best].get("description")
                    or pd_assets[best].get("title", "")
                )
                b["reference_strength"] = REFERENCE_STRENGTH

        # Pass 2.5: generic stash fallback
        if stash_indices:
            STASH_THRESHOLD = float(cfg.generic_stash.get("threshold", 0.18))
            STASH_MAX_REUSES = int(cfg.generic_stash.get("max_reuses_per_asset", 5))
            stash_use_count: dict[int, int] = {}
            stash_assigned = 0
            for bi, b in enumerate(beats):
                if "pd_asset_id" in b:
                    continue
                row = sims[bi].copy()
                for ai in range(len(pd_assets)):
                    if ai not in stash_indices:
                        row[ai] = -1.0
                for ai, c in stash_use_count.items():
                    if c >= STASH_MAX_REUSES:
                        row[ai] = -1.0
                if len(row) == 0:
                    continue
                best = int(np.argmax(row))
                if row[best] >= STASH_THRESHOLD:
                    b["pd_asset_id"] = pd_assets[best]["id"]
                    b["pd_asset_path"] = pd_assets[best]["local_path"]
                    stash_use_count[best] = stash_use_count.get(best, 0) + 1
                    stash_assigned += 1
                    b.pop("reference_asset_id", None)
                    b.pop("reference_asset_path", None)
                    b.pop("reference_description", None)
                    b.pop("reference_strength", None)
            logger.info("S08 stash fallback: %d beats covered by %d stash entries",
                        stash_assigned, len(stash_use_count))

    # ----- FLUX fallback for unmatched beats -----
    style_prefix = (style_yaml.get("prefix") or "").strip()
    style_suffix = (style_yaml.get("suffix") or "").strip()
    style_neg = (style_yaml.get("negative_prompt") or "").strip()

    force_no_text = bool(iqa.get("flux_force_no_text", True))
    NO_TEXT_POSITIVE = (
        "clean image with no text, no captions, no subtitles, no signs, "
        "no watermarks, no logos with letters, no writing of any kind"
    )
    NO_TEXT_NEGATIVE = (
        "text, letters, words, writing, captions, subtitles, watermark, "
        "logo with letters, signature, signage with legible text, "
        "readable lettering, alphabet, numbers, document text, "
        "printed sentences, typography, labels, name tags, banners with words"
    )

    for b in beats:
        if "pd_asset_id" in b:
            continue
        beat_prompt = b.get("flux_fallback_prompt", "") or b.get("specific_visual_description", "")

        # Path A — text grounding: weave PD reference description into prompt
        ref_desc = (b.get("reference_description") or "").strip()
        if ref_desc:
            beat_prompt = (
                f"Subject reference (from photographic record): {ref_desc}. "
                f"{beat_prompt}"
            )

        composed = f"{style_prefix} {beat_prompt}, {style_suffix}"
        negative = style_neg
        if force_no_text:
            composed = f"{composed}. {NO_TEXT_POSITIVE}."
            negative = f"{NO_TEXT_NEGATIVE}, {negative}"

        flux_req: dict = {
            "prompt": composed,
            "negative_prompt": negative,
        }
        # Path B — img2img is NOT supported by the CLI flux. Even when
        # USE_IMAGE_REFERENCE is on, the flux.py adapter logs an INFO
        # and proceeds text-only. Leaving the field here only as a
        # contract-compatibility breadcrumb; setting
        # pd_image_reference_enabled=false in config silences the log.
        ref_path = b.get("reference_asset_path")
        if ref_path and USE_IMAGE_REFERENCE:
            flux_req["reference_image_path"] = ref_path
            flux_req["reference_strength"] = b.get("reference_strength", REFERENCE_STRENGTH)
        b["flux_render_request"] = flux_req

    pd_matched = sum(1 for b in beats if "pd_asset_id" in b)
    flux_needed = sum(1 for b in beats if "flux_render_request" in b)

    (ws / "02_script" / "beat_sheet.json").write_text(
        json.dumps({
            "beats": beats,
            "total_estimated_seconds": sum(b["estimated_seconds"] for b in beats),
            "matched_pd_count": pd_matched,
            "flux_needed_count": flux_needed,
        }, indent=2)
    )
    logger.info("S08 complete: %d beats, %d direct PD, %d FLUX",
                len(beats), pd_matched, flux_needed)
    return None
