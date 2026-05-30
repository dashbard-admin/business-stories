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


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


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
      - sentence_index: zero-based index of the sentence that owns it
      - offset_seconds: fallback only; S12 recomputes sentence timing
    Caps at `max_per_beat`; extras stripped from text but dropped
    from the list."""
    matches = list(_CALLOUT_RE.finditer(text))
    if not matches:
        return text, []
    callouts: list[dict] = []
    for m in matches[:max_per_beat]:
        raw = m.group(1).strip().strip('"').strip("'")
        prefix = _CALLOUT_RE.sub("", text[:m.start()])
        sentence_index = max(0, len(_split_sentences(prefix)) - 1)
        callouts.append({
            "text": raw,
            "sentence_index": sentence_index,
            "offset_seconds": 0.0,
        })
    # Always strip ALL CALLOUT markers from the script_text so
    # Kokoro doesn't read them aloud — even the ones we dropped
    # because they exceeded max_per_beat.
    cleaned = _CALLOUT_RE.sub("", text)
    # Collapse the empty whitespace the strip leaves behind.
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned, callouts


# ----------------------------------------------------------------------
# Batch F 2026-05-27 helpers
# ----------------------------------------------------------------------

# Available Ken Burns motion variants (must match motion_to_params() in
# pipeline/ffmpeg_builder.py — keep them in sync).
_KEN_BURNS_MOTIONS = (
    "slow_zoom_in",
    "slow_zoom_out",
    "slow_pan_left",
    "slow_pan_right",
    "hold_still",
)


def _diversify_ken_burns_motion(beats: list[dict], *, episode_seed: int) -> None:
    """Re-distribute Ken Burns motion across beats so 60+ sequential
    panels don't all zoom in identically. Pet.com review showed
    `slow_zoom_in` dominated >80% of beats — monotonous.

    Strategy: walk beats in order, cycle through the 5 motions, with a
    deterministic offset seeded by episode_id so re-runs reproduce the
    same sequence. Hero-centric founder portraits keep face-friendly
    motions; the others rotate."""
    import random
    rng = random.Random(episode_seed)
    # Shuffle the motion order once per episode so episodes don't all
    # look identical, but the per-beat sequence is stable within an
    # episode.
    cycle = list(_KEN_BURNS_MOTIONS)
    rng.shuffle(cycle)

    cycle_idx = 0
    for b in beats:
        intent = (b.get("visual_intent") or "").strip().lower()
        # Hero-centric intents: keep the LLM's pick if it's already a
        # face-friendly motion; else force slow_zoom_in (cinematic for
        # portraits).
        if intent in _HERO_CENTRIC_INTENTS:
            current = (b.get("ken_burns_motion") or "").strip().lower()
            if current not in {"slow_zoom_in", "slow_zoom_out", "hold_still"}:
                b["ken_burns_motion"] = "slow_zoom_in"
            continue
        # All other beats: cycle through the shuffled motion list.
        b["ken_burns_motion"] = cycle[cycle_idx % len(cycle)]
        cycle_idx += 1


# Visual intents that work as a hook (face / object / dramatic scene
# the viewer can lock onto in the first 15 seconds).
_HOOK_SAFE_INTENTS = {
    "founder_portrait",
    "product_reveal",
    "office_environment",
    "boardroom_meeting",
    "street_scene",
    "crowd_or_market",
    "factory_or_workshop",
}

# Visual intents the hook MUST NOT use (flat, dark, text-heavy — the
# 0:30 mark of the Pets.com review video was a near-black document
# beat which is precisely the wrong frame at the retention decision
# point).
_HOOK_BANNED_INTENTS = {
    "document_or_headline",
    "chart_abstraction",
    "montage_panel",
}


def _enforce_hook_beat_intents(beats: list[dict]) -> None:
    """For the first 3 beats: if the LLM picked a banned intent,
    rewrite to the safest available alternative. This is a hard
    constraint, not a hint.

    Batch M 2026-05-30: do NOT turn legal/document hooks into
    founder_portraits. That made legal stories open on unrelated
    founder/app imagery. Keep cinematic document hooks as documents;
    only rewrite flat chart/montage hooks."""
    for b in beats[:3]:
        intent = (b.get("visual_intent") or "").strip().lower()
        if intent in {"chart_abstraction", "montage_panel"}:
            b["visual_intent"] = "document_or_headline"
            b["_hook_intent_overridden"] = intent


# Decline-arc story_kinds that get the era-aware style switch.
_DECLINE_STORY_KINDS = {
    "rise_and_fall",
    "scandal_postmortem",
    "founder_drama",
}


def _attach_act_and_style(
    beats: list[dict],
    *,
    story_kind: str,
    locked_style: str,
) -> None:
    """Tag each beat with `act` (0..5 inclusive, plus 3.5 for the
    investigation slot) and `effective_visual_style`.

    For decline stories: V1 on Acts 0-2 (the rise — bright, optimistic
    comic), V2 on Acts 3-5 (the fall — noir comic). For non-decline
    stories the locked episode-level style applies to every beat.

    The act lookup is based on the BEAT POSITION (rank within the
    beat list), mapped to the 7-act distribution from
    pipeline/prompts/script_generate.txt:
        Act 0:    1-2 beats
        Act 1:    ~12 beats
        Act 2:    ~11 beats
        Act 3:    ~16 beats
        Act 3.5:  ~9 beats
        Act 4:    ~12 beats
        Act 5:    ~5-6 beats
    Total ~67-72 mid. We compute the per-act cutoff PROPORTIONALLY so
    the mapping still works for episodes with 65-95 beats.
    """
    n = len(beats)
    if n == 0:
        return

    # Reference fractions (sum to 1.0) from the script_generate template.
    fractions = [
        ("0",   0.02),
        ("1",   0.18),
        ("2",   0.16),
        ("3",   0.23),
        ("3.5", 0.13),
        ("4",   0.18),
        ("5",   0.10),
    ]
    cutoffs: list[tuple[str, int]] = []
    acc = 0.0
    for label, frac in fractions:
        acc += frac
        cutoffs.append((label, int(round(n * acc))))

    is_decline = story_kind in _DECLINE_STORY_KINDS

    for i, b in enumerate(beats):
        act_label = "5"
        for label, ci in cutoffs:
            if i < ci:
                act_label = label
                break
        b["act"] = act_label

        if is_decline:
            # V1 for Acts 0, 1, 2; V2 from Act 3 onward.
            if act_label in {"0", "1", "2"}:
                b["effective_visual_style"] = "V1"
            else:
                b["effective_visual_style"] = "V2"
        else:
            b["effective_visual_style"] = locked_style


# ----------------------------------------------------------------------
# Batch M 2026-05-30: narrative visual continuity
# ----------------------------------------------------------------------

_LEGAL_TERMS = {
    "bankruptcy", "court", "filing", "petition", "judgment", "restitution",
    "liquidation", "trust", "fraud", "ponzi", "wire fraud", "mail fraud",
    "shapiro", "woodbridge", "mercer", "investor", "investors", "sentence",
    "sentenced", "prison", "debtor", "settlement", "claim", "claims",
}
_PRODUCT_TERMS = {
    "twitter", "vine", "costolo", "app", "video", "creator", "creators",
    "tiktok", "algorithm", "feed", "short-form", "monetize", "advertising",
    "product", "platform", "acquisition", "acquired",
}
_BRIDGE_TERMS = {
    "same name", "share a name", "two companies", "not the same",
    "confusion", "confused", "legal entity", "entity", "entities",
    "name was stolen", "corporate records", "registry", "record remains",
}


def _story_world_for_beat(beat: dict) -> str:
    text = " ".join([
        beat.get("script_text") or "",
        beat.get("specific_visual_description") or "",
        beat.get("flux_fallback_prompt") or "",
    ]).lower()
    legal_score = sum(1 for t in _LEGAL_TERMS if t in text)
    product_score = sum(1 for t in _PRODUCT_TERMS if t in text)
    bridge_score = sum(1 for t in _BRIDGE_TERMS if t in text)
    if bridge_score or (legal_score and product_score):
        return "name_confusion_bridge"
    if legal_score > product_score:
        return "legal_fraud"
    if product_score > 0:
        return "product_platform"
    if (beat.get("act") or "") == "5":
        return "closing_resolution"
    return "business_context"


def _scene_label(world: str) -> str:
    return {
        "legal_fraud": "The bankruptcy file",
        "product_platform": "The dying video app",
        "name_confusion_bridge": "Two names collide",
        "closing_resolution": "The record that remains",
        "business_context": "The market pressure",
    }.get(world, "The business story")


def _recurring_props(world: str) -> list[str]:
    return {
        "legal_fraud": [
            "red fraud case folder with no readable label",
            "Delaware bankruptcy court file",
            "investor packet",
            "amber desk lamp",
        ],
        "product_platform": [
            "cracked smartphone with six blank video tiles",
            "green product folder with no readable label",
            "dim startup office monitors",
            "empty creator workspace",
        ],
        "name_confusion_bridge": [
            "red fraud folder with no readable label",
            "green product folder with no readable label",
            "evidence board with red string",
            "blurred corporate registry printouts",
        ],
        "closing_resolution": [
            "two separated folders, one red and one green",
            "closed court archive box",
            "empty office chair",
            "fading nameplate with no legible letters",
        ],
        "business_context": [
            "conference-room projector glow",
            "muted financial evidence board",
            "empty desks",
            "single desk lamp",
        ],
    }.get(world, ["evidence folder", "desk lamp", "empty office"])


def _visual_must_not(world: str) -> list[str]:
    base = [
        "no readable text",
        "no legible logos",
        "no random decorative vine leaves",
        "no unrelated music or record imagery",
    ]
    if world == "legal_fraud":
        return base + [
            "no smartphone",
            "no app interface",
            "no startup pitch room",
            "no founder glamour portrait",
        ]
    if world == "product_platform":
        return base + [
            "no courtroom",
            "no gavel",
            "no prison bars",
            "no bankruptcy paperwork as the main subject",
        ]
    if world == "name_confusion_bridge":
        return base + [
            "no single giant app logo",
            "no generic founder keynote",
        ]
    return base


def _movie_shot_prompt(beat: dict, scene: dict) -> str:
    """Rewrite a beat prompt as a movie shot tied to the scene spine."""
    script = (beat.get("script_text") or "").strip()
    desc = (beat.get("specific_visual_description") or "").strip()
    world = beat.get("story_world") or scene["story_world"]
    props = ", ".join(scene.get("recurring_props") or _recurring_props(world))
    must_not = ", ".join(scene.get("must_not_show") or _visual_must_not(world))
    intent = (beat.get("visual_intent") or "").strip()

    if intent == "chart_abstraction":
        desc = (
            "A grounded investigation-board shot instead of a floating "
            "abstract chart: folders, pinned blurred papers, colored string, "
            "and one simple unlabeled line or timeline shape in the scene."
        )
    elif intent == "montage_panel":
        desc = (
            "A physical montage wall inside the same investigation room, "
            "using pinned photos, folders, and blurred documents rather than "
            "free-floating symbolic shapes."
        )

    if world == "legal_fraud":
        camera = "tight investigative close-up or over-the-shoulder legal desk shot"
        purpose = (
            "make the viewer feel the fraud is being uncovered through court "
            "records and money trails"
        )
    elif world == "product_platform":
        camera = "cinematic startup-office product shot"
        purpose = (
            "make the viewer understand the fading short-form video product "
            "and Twitter-era strategy"
        )
    elif world == "name_confusion_bridge":
        camera = "split-focus investigative shot connecting two evidence folders"
        purpose = (
            "make the viewer understand that two legally different companies "
            "are being confused because they share a name"
        )
    elif world == "closing_resolution":
        camera = "quiet final-frame archive shot"
        purpose = "make the viewer feel the legal record outlived the product"
    else:
        camera = "restrained business-documentary scene shot"
        purpose = "support the current beat without adding unrelated story details"

    script_hint = re.sub(r"\s+", " ", script)[:260]
    return (
        f"Movie shot: {camera}. Scene: {scene['scene_title']}. "
        f"Narrative purpose: {purpose}. Recurring props in frame: {props}. "
        f"Beat narration context: {script_hint}. Specific composition: {desc}. "
        f"Must not show: {must_not}."
    )


def _apply_visual_continuity_plan(beats: list[dict]) -> list[dict]:
    """Group beats into scenes and rewrite FLUX prompts around a visual spine.

    The goal is for the video to read as one investigation movie, not 70
    independent symbolic panels. Scene IDs are deterministic and stored in
    beat_sheet.json for S09/S12 auditability.
    """
    scenes: list[dict] = []
    current: dict | None = None
    scene_idx = 0
    for i, beat in enumerate(beats):
        world = _story_world_for_beat(beat)
        if (
            current is None
            or current["story_world"] != world
            or len(current["beat_ids"]) >= 6
        ):
            scene_idx += 1
            current = {
                "scene_id": f"SCENE_{scene_idx:02d}",
                "scene_title": _scene_label(world),
                "story_world": world,
                "recurring_props": _recurring_props(world),
                "must_not_show": _visual_must_not(world),
                "beat_ids": [],
            }
            scenes.append(current)
        current["beat_ids"].append(beat.get("beat_id"))
        beat["scene_id"] = current["scene_id"]
        beat["scene_title"] = current["scene_title"]
        beat["story_world"] = world
        beat["recurring_props"] = current["recurring_props"]
        beat["visual_must_not"] = current["must_not_show"]
        beat["movie_shot_prompt"] = _movie_shot_prompt(beat, current)
        beat["flux_fallback_prompt"] = beat["movie_shot_prompt"]
        beat["visual_prompt_version"] = "storyboard_v1_2026-05-30"
    return scenes


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
    # Batch E 2026-05-27 performance hints (soft guidance).
    from ..performance_summary import summarise_for_prompt
    perf = summarise_for_prompt()
    prompt = template.format(
        visual_style_name=style_yaml["name"],
        visual_style_guidance=style_yaml.get("guidance_for_llm", ""),
        narrator_name=narrator["name"],
        script_with_beats=script,
        visual_intents_that_retained=perf["visual_intents_that_retained"],
        visual_intents_that_lost_viewers=perf["visual_intents_that_lost_viewers"],
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
    # Batch K 2026-05-29: log the parsed count unconditionally so the
    # operator can tell "S08 found 0 [CALLOUT: ...] markers" from
    # "S08 didn't run". final3.mp4 had no visible callout overlays
    # despite the script having 36 callout candidates — this log
    # makes the gap between S08's parse and S12's composite visible.
    if callouts_enabled:
        beats_with_callouts = sum(1 for b in beats if b.get("callouts"))
        logger.info(
            "S08 callouts: parsed %d markers across %d beats "
            "(cap %d per beat)",
            callout_total, beats_with_callouts, callouts_max,
        )

    for b in beats:
        b.setdefault("estimated_seconds", _estimate_seconds(b.get("script_text", "")))

    # ----- Ken Burns motion variety (Batch F 2026-05-27) -----
    # The writer LLM tends to default every beat to slow_zoom_in, which
    # makes 60-95 sequential beats feel monotonous (the Pets.com review
    # video had this — every panel zoomed in the same way for 14
    # minutes). Re-distribute deterministically over the 5 available
    # motions so the cadence varies but is reproducible across re-runs.
    _diversify_ken_burns_motion(beats, episode_seed=hash(episode["id"]) & 0xffff)

    # ----- Hook-beat visual-intent restriction (Batch F 2026-05-27) -----
    # First 3 beats decide whether the viewer keeps watching. They MUST
    # be high-contrast and contain a face / object the viewer can lock
    # onto. document_or_headline beats are flat, dark, and text-heavy
    # — exactly what tanked the Pets.com episode's 0:30 hook frame.
    _enforce_hook_beat_intents(beats)

    # ----- Era-aware per-beat visual style (Batch F 2026-05-27) -----
    # Decline stories should LOOK optimistic in the rise and bleak in
    # the fall. Override the episode-locked visual_style on a per-beat
    # basis for rise_and_fall / scandal_postmortem / founder_drama:
    # V1 for Acts 0-2 (the rise), V2 for Acts 3-5 (the fall).
    incident = episode.get("incident") or {}
    story_kind = (incident.get("story_kind") or "").strip().lower()
    _attach_act_and_style(beats, story_kind=story_kind,
                          locked_style=episode["visual_style"])

    # ----- Visual continuity / storyboard spine (Batch M 2026-05-30) -----
    # Rewrite FLUX-bound beat descriptions into scene-aware movie shots.
    # This keeps the image sequence aligned to the narration instead of
    # letting each beat become an isolated symbolic illustration.
    scene_plan = _apply_visual_continuity_plan(beats)
    logger.info(
        "S08 visual continuity: %d scenes across %d beats",
        len(scene_plan), len(beats),
    )

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
    REFERENCE_THRESHOLD = max(float(iqa.get("pd_reference_threshold", 0.20)), 0.52)
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
            if row[best] >= REFERENCE_THRESHOLD and _asset_reference_allowed(
                b, pd_assets[best], float(row[best])
            ):
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
    # Per-beat style lookup (Batch F 2026-05-27). Pre-load BOTH V1 and
    # V2 style YAMLs so we can pick whichever the beat's
    # `effective_visual_style` field requests. Beats with no
    # effective_visual_style fall back to the episode-locked style.
    style_cache: dict[str, dict] = {episode["visual_style"]: style_yaml}
    def _load_style(sid: str) -> dict:
        if sid in style_cache:
            return style_cache[sid]
        path = cfg.style_profiles_dir / f"{sid}.yaml"
        if not path.exists():
            logger.warning("S08: visual_style %s not found; falling back to %s",
                           sid, episode["visual_style"])
            style_cache[sid] = style_yaml
            return style_yaml
        sy = yaml.safe_load(path.read_text())
        style_cache[sid] = sy
        return sy

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

    # Iconic-asset preamble (Batch F 2026-05-27). Operator-curated list
    # of "must include" visual assets for THIS company — sock puppet
    # for Pets.com, 1999 web browsers, brown shipping boxes with logo,
    # etc. Loaded from 00_research/iconic_assets.json (emitted by S01
    # or hand-authored). Injected into every FLUX prompt so the
    # generated panels visually identify as belonging to the episode's
    # subject company rather than as generic business imagery. Batch M:
    # these cues are gated below by story_world/intent so legal scenes
    # don't turn into generic app-logo panels.
    iconic_preamble = ""
    iconic_path = ws / "00_research" / "iconic_assets.json"
    if iconic_path.exists():
        try:
            iconic_data = json.loads(iconic_path.read_text())
            assets = iconic_data.get("assets") or []
            if assets:
                iconic_preamble = (
                    "Iconic visual cues for this company (weave naturally "
                    "where the scene allows; not every beat must include "
                    "all): "
                    + "; ".join(a.get("description", "") for a in assets
                                if a.get("description"))
                )
        except Exception as e:
            logger.warning("S08: iconic_assets.json unreadable: %s", e)

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
        if iconography and _allow_hero_iconography(b, founder_tokens):
            beat_prompt = f"{iconography} {beat_prompt}"
            hero_inject_count += 1

        # Iconic-asset cues are intentionally gated. Batch M: injecting
        # them into every beat made legal/fraud scenes fill with app
        # phones and leaf logos. Product cues belong only in product or
        # name-confusion beats, and even there as subtle scene context.
        if iconic_preamble and _allow_iconic_cues(b):
            beat_prompt = (
                f"Subtle product-era context only; do not make the logo "
                f"or phone UI the main subject unless this beat explicitly "
                f"calls for it. {iconic_preamble}. {beat_prompt}"
            )

        # Path A — text grounding: weave PD reference description into prompt
        ref_desc = (b.get("reference_description") or "").strip()
        if ref_desc:
            beat_prompt = (
                f"Highly matched subject reference: {ref_desc}. "
                f"{beat_prompt}"
            )

        # Per-beat style lookup (Batch F 2026-05-27).
        beat_style_id = b.get("effective_visual_style") or episode["visual_style"]
        beat_style_yaml = _load_style(beat_style_id)
        bsp = (beat_style_yaml.get("prefix") or "").strip()
        bss = (beat_style_yaml.get("suffix") or "").strip()
        bsn = (beat_style_yaml.get("negative_prompt") or "").strip()

        composed = f"{bsp} {beat_prompt}, {bss}"
        negative = bsn
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
            "visual_continuity_plan": scene_plan,
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
}


def _allow_iconic_cues(beat: dict) -> bool:
    world = (beat.get("story_world") or "").strip()
    intent = (beat.get("visual_intent") or "").strip()
    if world == "legal_fraud":
        return False
    if intent in {"document_or_headline", "courtroom_scene"}:
        return False
    return world == "product_platform" and intent in {
        "product_reveal",
        "founder_portrait",
    }


def _allow_hero_iconography(beat: dict, founder_tokens: set[str]) -> bool:
    world = (beat.get("story_world") or "").strip()
    intent = (beat.get("visual_intent") or "").strip()
    if world == "legal_fraud":
        return False
    if intent != "founder_portrait":
        return False
    if world == "name_confusion_bridge":
        text = (beat.get("script_text") or "").lower()
        return bool(founder_tokens) and any(t in text for t in founder_tokens)
    return _beat_shows_hero(beat, founder_tokens)


def _asset_reference_allowed(beat: dict, asset: dict, score: float) -> bool:
    """Guard FLUX text references against weak/off-topic PD assets.

    Semantic similarity alone matched "Vine Inc." beats to vinyl records,
    P-Vine music logos, and sheet music because they shared surface words.
    References now need a high score plus intent/story-world sanity.
    """
    if score < 0.52:
        return False
    desc = " ".join([
        str(asset.get("caption") or ""),
        str(asset.get("description") or ""),
        str(asset.get("title") or ""),
    ]).lower()
    if not desc.strip():
        return False
    blocked = {
        "vinyl", "record", "records", "music", "song", "album",
        "sheet music", "piano", "label branding", "ornate header",
    }
    if any(term in desc for term in blocked):
        return False

    world = (beat.get("story_world") or "").strip()
    intent = (beat.get("visual_intent") or "").strip()
    if world == "legal_fraud":
        return any(
            term in desc
            for term in (
                "court", "legal", "filing", "document", "bankruptcy",
                "judge", "lawyer", "prison", "investor", "office",
            )
        )
    if intent == "founder_portrait":
        return any(
            term in desc
            for term in ("portrait", "person", "man", "woman", "founder", "ceo")
        )
    if world == "product_platform":
        return any(
            term in desc
            for term in ("phone", "app", "office", "startup", "video", "screen")
        )
    if world == "name_confusion_bridge":
        return any(
            term in desc
            for term in ("document", "filing", "office", "logo", "phone", "screen")
        )
    return score >= 0.68


def _beat_shows_hero(beat: dict, founder_tokens: set[str]) -> bool:
    """Decide whether to prepend the character iconography to this
    beat's FLUX prompt. True iff:
      - the beat's visual_intent is hero-centric, OR
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
