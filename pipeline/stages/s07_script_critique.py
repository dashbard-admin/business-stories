"""S07 — Script Critique & Rewrite.

Sends the script to Gemma-4 (critic) for an adversarial editorial
pass tuned for business-story retention + voice. Applies up to 3
substring rewrites in place per loop. Loops at most twice.

Critic responses include `original` strings that are *supposed* to be
verbatim substrings of the script. In practice the critic often
paraphrases lightly — smart vs straight quotes, em vs en dashes,
collapsed whitespace. We do a normalized-text match so those near-
misses still apply instead of being silently skipped.

After the rewrite loops, an independent brand-safety pass runs the
brand_safety_review.txt prompt to flag defamation risk on living
named people, unframed speculation, and intent attributions not
supported by the fact ledger. Output flags go to
02_script/brand_safety_flags.json. When cfg.brand_safety.enabled
is true AND any flag at gate_on_severity or above fires, S07
returns a needs_human reason so the operator can review before
the pipeline advances to S08. The operator clears the gate via
`python -m pipeline.hermes_orchestrator --approve <ep_id>`. Added
Batch B 2026-05-26.

Inputs:  02_script/script.txt
Outputs: 02_script/script.txt (modified)
         02_script/critique_history.json
         02_script/brand_safety_flags.json
"""

from __future__ import annotations

import json
import logging
import re

from ..config import load_config
from ..llm import LLM
from ..state import find_episode_workspace

logger = logging.getLogger("hermes.stage.s07")

MAX_LOOPS = 2

_NORMALIZE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-",
    "…": "...",
    " ": " ",
}

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    for src, dst in _NORMALIZE_MAP.items():
        text = text.replace(src, dst)
    return _WS_RE.sub(" ", text).strip()


def _apply_rewrite(script: str, original: str, replacement: str) -> tuple[str, str]:
    if original in script:
        return script.replace(original, replacement, 1), "exact"

    norm_orig = _normalize(original)
    if not norm_orig:
        return script, "miss"

    raw = script
    norm_chars: list[str] = []
    raw_idx_for_norm_idx: list[int] = []
    prev_was_space = False
    for i, ch in enumerate(raw):
        mapped = _NORMALIZE_MAP.get(ch, ch)
        if mapped.isspace() or ch.isspace():
            if prev_was_space:
                continue
            mapped = " "
            prev_was_space = True
        else:
            prev_was_space = False
        for sub_ch in mapped:
            norm_chars.append(sub_ch)
            raw_idx_for_norm_idx.append(i)

    norm_script = "".join(norm_chars).strip()
    if not norm_script:
        return script, "miss"

    lead = 0
    while lead < len(norm_chars) and norm_chars[lead] == " ":
        lead += 1
    norm_script_aligned = "".join(norm_chars[lead:])
    raw_idx_aligned = raw_idx_for_norm_idx[lead:]

    pos = norm_script_aligned.find(norm_orig)
    if pos < 0:
        return script, "miss"

    start_raw = raw_idx_aligned[pos]
    end_norm = pos + len(norm_orig) - 1
    if end_norm >= len(raw_idx_aligned):
        return script, "miss"
    end_raw = raw_idx_aligned[end_norm] + 1

    return raw[:start_raw] + replacement + raw[end_raw:], "normalized"


def run(episode: dict, queue: dict) -> str | None:
    cfg = load_config()
    critic = LLM(role="critic")
    ws = find_episode_workspace(episode["id"])
    if not ws:
        return "no episode workspace"

    script_path = ws / "02_script" / "script.txt"
    if not script_path.exists():
        return "no script.txt"
    script = script_path.read_text()

    template = (cfg.prompts_dir / "script_critique.txt").read_text()
    history: list[dict] = []
    incident = episode["incident"]
    narrator_id = episode["narrator"]
    narr = cfg.narrator_by_id(narrator_id)

    for loop in range(MAX_LOOPS):
        prompt = template.format(
            incident_name=incident["company_name"],
            hero=incident.get("hero", ""),
            conflict=incident.get("conflict", ""),
            narrator_name=narr["name"],
            narrator_tone=narr.get("tone", ""),
            script=script,
        )
        try:
            review = critic.complete_json(
                prompt, temperature=0.4, max_tokens=4000
            )
        except Exception as e:
            logger.warning("critique JSON parse failed: %s", e)
            history.append({"loop": loop, "error": str(e)})
            break

        verdict = (review.get("verdict") or "").lower()
        rewrites = review.get("rewrites") or []
        loop_record: dict = {
            "loop": loop,
            "verdict": verdict,
            "issues_summary": review.get("issues_summary", ""),
            "rewrites_proposed": len(rewrites),
            "rewrites": [],
        }

        if verdict == "pass" or not rewrites:
            logger.info("S07 critique pass on loop %d", loop)
            history.append(loop_record)
            break

        applied = 0
        for r in rewrites[:3]:
            original = (r.get("original") or "").strip()
            replacement = (r.get("replacement") or "").strip()
            reason = (r.get("reason") or "").strip()
            entry = {"original": original, "replacement": replacement, "reason": reason}

            if not original or not replacement:
                entry["status"] = "empty"
            else:
                script, status = _apply_rewrite(script, original, replacement)
                entry["status"] = status
                if status in ("exact", "normalized"):
                    applied += 1
                    logger.info("S07 applied rewrite (%s): %s", status, reason[:60])
                else:
                    logger.info("S07 rewrite missed: %s", original[:60])

            loop_record["rewrites"].append(entry)

        loop_record["rewrites_applied"] = applied
        history.append(loop_record)

        if applied == 0:
            logger.info("S07 no rewrites applied; treating as pass")
            break

        if verdict == "ship_blocker" and applied < len(rewrites):
            # Ship-blocker but rewrites partially landed — loop again.
            continue
        break

    script_path.write_text(script)
    (ws / "02_script" / "critique_history.json").write_text(
        json.dumps(history, indent=2)
    )

    # ---------- Brand-safety pass (Batch B 2026-05-26) ----------
    return _run_brand_safety_pass(
        episode, queue, cfg, script, ws, incident,
    )


def _run_brand_safety_pass(
    episode: dict,
    queue: dict,
    cfg,
    script: str,
    ws,
    incident: dict,
) -> str | None:
    """Run brand_safety_review.txt over the (post-critique) script.
    Emits 02_script/brand_safety_flags.json. Returns None on a clean
    pass; returns a needs_human reason when the flag count at or
    above the configured gate_on_severity is > 0 (operator clears
    via --approve)."""
    bs_cfg = cfg.brand_safety
    flag_file = ws / "02_script" / "brand_safety_flags.json"

    if not bs_cfg.get("enabled", True):
        # Logged-only mode — still write an empty-result file so
        # downstream inspection has something to look at.
        flag_file.write_text(json.dumps({
            "verdict": "skipped",
            "reason": "brand_safety.enabled=false",
            "flags": [],
            "high_severity_count": 0,
            "low_severity_count": 0,
        }, indent=2))
        return None

    # Lazy-load the fact ledger if it exists. The brand-safety prompt
    # uses it to decide what's safe to state as fact.
    fact_ledger_path = ws / "01_factcheck" / "verified_facts.json"
    fact_ledger_json = "[]"
    if fact_ledger_path.exists():
        try:
            fact_ledger_json = fact_ledger_path.read_text()
        except Exception:
            pass

    template_path = cfg.prompts_dir / "brand_safety_review.txt"
    if not template_path.exists():
        logger.warning("brand_safety_review.txt missing; skipping pass")
        return None
    template = template_path.read_text()

    critic = LLM(role="critic")
    prompt = template.format(
        incident_name=incident.get("company_name", ""),
        hero=incident.get("hero", ""),
        conflict=incident.get("conflict", ""),
        fact_ledger_json=fact_ledger_json,
        script=script,
    )
    try:
        result = critic.complete_json(prompt, temperature=0.2, max_tokens=4000)
    except Exception as e:
        logger.warning("brand-safety review JSON parse failed: %s", e)
        flag_file.write_text(json.dumps({
            "verdict": "error",
            "error": str(e)[:300],
            "flags": [],
            "high_severity_count": 0,
            "low_severity_count": 0,
        }, indent=2))
        return None

    flags = result.get("flags") or []
    high = sum(1 for f in flags if (f.get("severity") or "").lower() == "high")
    low = sum(1 for f in flags if (f.get("severity") or "").lower() == "low")
    result["high_severity_count"] = high
    result["low_severity_count"] = low

    # Always log a count summary, per Q-B2 (log on every S07 run).
    logger.info(
        "S07 brand-safety: verdict=%s flags=%dH/%dL",
        result.get("verdict", "?"), high, low,
    )
    if high > 0 or low > 0:
        # Log the first few flags so the operator can spot patterns
        # in the daily log without opening the JSON.
        for f in flags[:5]:
            logger.info(
                "  [%s] %s — %s",
                f.get("severity", "?"),
                (f.get("sentence") or "")[:120],
                (f.get("reasoning") or "")[:140],
            )

    flag_file.write_text(json.dumps(result, indent=2))

    # Record the counts on the episode so --status can show them.
    from ..state import update_episode
    update_episode(
        queue, episode["id"],
        safety_flags_count={"high": high, "low": low},
    )

    # Apply the gate.
    gate = (bs_cfg.get("gate_on_severity") or "high").lower()
    if gate == "off":
        return None
    if gate == "high" and high > 0:
        return (
            f"brand-safety: {high} high-severity flag(s) require review. "
            f"Inspect 02_script/brand_safety_flags.json then run "
            f"`--approve {episode['id']}` to clear."
        )
    if gate == "low" and (high > 0 or low > 0):
        return (
            f"brand-safety: {high}H + {low}L flag(s) require review "
            f"(gate_on_severity=low). Inspect 02_script/"
            f"brand_safety_flags.json then run `--approve "
            f"{episode['id']}` to clear."
        )
    return None
