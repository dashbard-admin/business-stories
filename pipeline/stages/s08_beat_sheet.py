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


# [CALLOUT: "text"] marker parser (Batch C 2026-05-26).
# Matches both straight-quoted and curly-quoted forms; case-insensitive
# on "CALLOUT". Any whitespace around the colon is tolerated.
_CALLOUT_RE = re.compile(
    r"\[\s*CALLOUT\s*:\s*[\"“‘]?([^\"”’\]]+)[\"”’]?\s*\]",
    re.IGNORECASE,
)


def _extract_callouts(text: str, max_per_beat: int) -> tuple[str, list[dict]]:
    """Strip `[CALLOUT: "TEXT"]` markers from `text` and return
    (cleaned_text, callouts_list). Each callout dict has:
      - text: the bracketed string (uppercase-trimmed)
      - offset_seconds: 0.0 for v1 (beat-anchored)
    Caps at `max_per_beat`; extras stripped from text but dropped
    from the list."""
    matches = list(_CALLOUT_RE.finditer(text))
    if not matches:
        return text, []
    callouts: list[dict] = []
    for m in matches[:max_per_beat]:
        raw = m.group(1).strip().strip('"').strip("'")
        callouts.append({"text": raw, "offset_seconds": 0.0})
    # Always strip ALL CALLOUT markers from the script_text so
    # Kokoro doesn't read them aloud — even the ones we dropped
    # because they exceeded max_per_beat.
    cleaned = _CALLOUT_RE.sub("", text)
    # Collapse the empty whitespace the strip leaves behind.
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned, callouts


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

    # Load character iconography for hero-beat FLUX-prompt injection.
    # Missing file = no iconography injection, beats render normally.
    character_profile: dict = {}
    cp_path = ws / "01_factcheck" / "character_profile.json"
    if cp_path.exists():
        try:
            character_profile = json.loads(cp_path.read_text())
        except Exception as e:
            logger.warning("character_profile.json unreadable: %s", e)
    founder_name = (
        (episode.get("incident") or {}).get("founder_or_protagonist") or ""
    ).strip()
    # Tokens used to detect "this beat shows the hero". Filter out
    # short / common tokens; require length > 2 so "Adam" qualifies
    # but stop-words don't.
    founder_tokens = {
        t.lower() for t in re.findall(r"[A-Za-z][A-Za-z']+", founder_name)
        if len(t) > 2
    }

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

    # ----- CALLOUT markers (Batch C 2026-05-26) -----
    # The writer may emit inline `[CALLOUT: "$9 BILLION"]` markers
    # after high-impact concrete-number sentences. Parse them into
    # a per-beat list `callouts: [{text, offset_seconds}]`, capped
    # at config.callouts.max_per_beat. The markers are stripped from
    # script_text so Kokoro doesn't try to read them aloud. S12
    # composites the bracketed text as a Pillow overlay on the
    # beat's clip at offset_seconds from beat start (Q-C1: beat-
    # anchored). Offset for v1 is always 0.0 — voice-anchored is
    # a future polish that needs S10 word-level timing.
    callouts_cfg = cfg.callouts
    callouts_max = int(callouts_cfg.get("max_per_beat", 1))
    callouts_enabled = bool(callouts_cfg.get("enabled", True))
    callout_total = 0
    for b in beats:
        text = b.get("script_text", "") or ""
        callouts: list[dict] = []
        if callouts_enabled:
            stripped, callouts = _extract_callouts(text, callouts_max)
            if callouts:
                b["script_text"] = stripped
                b["callouts"] = callouts
                callout_total += len(callouts)
    if callouts_enabled and callout_total:
        logger.info("S08 callouts: parsed %d markers across beats",
                    callout_total)

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

    # Narrow PD path (Batch C 2026-05-26). When config asset_hunt has
    # `enabled_visual_intents` set to a non-empty list, only beats whose
    # visual_intent is in that set are eligible for PD routing. The
    # default list (founder_portrait, document_or_headline) targets
    # the two intents where real photos add value without breaking
    # the comic style. Wired but only meaningful when asset_hunt
    # master switch is also true (it's false by default, so this list
    # has no observable effect until both flip on).
    pd_intent_filter = set(
        (cfg.raw.get("asset_hunt") or {}).get("enabled_visual_intents") or []
    )
    if pd_intent_filter:
        logger.info("S08 narrow-PD filter: %s", sorted(pd_intent_filter))

    def _pd_eligible_beat(b: dict) -> bool:
        if not pd_intent_filter:
            return True
        return (b.get("visual_intent") or "") in pd_intent_filter

    if sims is not None and len(pd_assets):
        import numpy as np
        # Pass 1: direct PD use
        for bi, b in enumerate(beats):
            if not _pd_eligible_beat(b):
                continue
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
            if not _pd_eligible_beat(b):
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

    iconography = (character_profile.get("iconography") or "").strip()
    hero_inject_count = 0

    for b in beats:
        if "pd_asset_id" in b:
            continue
        beat_prompt = b.get("flux_fallback_prompt", "") or b.get("specific_visual_description", "")

        # Hero-beat iconography injection. If the beat's
        # visual_description references the founder by name OR the
        # visual_intent is one of the "hero is central" intents,
        # prepend the iconography paragraph so FLUX renders a
        # consistent character across beats. Cross-beat consistency,
        # not photorealistic likeness — base FLUX can't do faces
        # from text alone.
        if iconography and _beat_shows_hero(b, founder_tokens):
            beat_prompt = f"{iconography} {beat_prompt}"
            hero_inject_count += 1

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
    if iconography:
        logger.info("S08 iconography: injected into %d hero-beat FLUX prompts",
                    hero_inject_count)
    else:
        logger.info("S08 iconography: skipped (no character_profile.json)")

    (ws / "02_script" / "beat_sheet.json").write_text(
        json.dumps({
            "beats": beats,
            "total_estimated_seconds": sum(b["estimated_seconds"] for b in beats),
            "matched_pd_count": pd_matched,
            "flux_needed_count": flux_needed,
            "hero_iconography_injections": hero_inject_count,
        }, indent=2)
    )
    logger.info("S08 complete: %d beats, %d direct PD, %d FLUX",
                len(beats), pd_matched, flux_needed)

    # In-flight gate (Batch B 2026-05-26). Default off; flip on for
    # the first few episodes to calibrate operator intuition for beat
    # distribution / visual intent quality before committing FLUX compute.
    if cfg.orchestrator.get("gate_at_S08", False):
        # Compose a beat-distribution summary for the operator.
        intent_counts: dict[str, int] = {}
        for b in beats:
            vi = b.get("visual_intent") or "?"
            intent_counts[vi] = intent_counts.get(vi, 0) + 1
        dist = ", ".join(f"{k}={v}"
                         for k, v in sorted(intent_counts.items(),
                                            key=lambda kv: -kv[1]))
        return (
            f"S08 gate enabled: review 02_script/beat_sheet.json "
            f"({len(beats)} beats, {pd_matched} PD, {flux_needed} FLUX; "
            f"distribution: {dist}) then run `--approve "
            f"{episode['id']}` to advance to S09."
        )
    return None


# Visual intents that almost always centre on the hero — even when
# the founder's name isn't in the description (e.g. "founder at his
# workbench", "the founder in profile"). Iconography injection
# fires unconditionally on these.
_HERO_CENTRIC_INTENTS = {
    "founder_portrait",
    "boardroom_meeting",
    "courtroom_scene",
}


def _beat_shows_hero(beat: dict, founder_tokens: set[str]) -> bool:
    """Decide whether to prepend the character iconography to this
    beat's FLUX prompt. True iff:
      - the beat's visual_intent is one of the hero-centric intents
        (founder_portrait, boardroom_meeting, courtroom_scene), OR
      - any meaningful token of the founder's name appears in the
        beat's visual description / fallback prompt (case-insensitive).
    """
    intent = (beat.get("visual_intent") or "").strip().lower()
    if intent in _HERO_CENTRIC_INTENTS:
        return True
    if not founder_tokens:
        return False
    haystack = " ".join([
        beat.get("specific_visual_description") or "",
        beat.get("flux_fallback_prompt") or "",
        beat.get("script_text") or "",
    ]).lower()
    return any(t in haystack for t in founder_tokens)
