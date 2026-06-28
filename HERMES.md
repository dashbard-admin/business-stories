# HERMES.md

Runbook for autonomous agents operating the Hermes business-story pipeline.

This document has three parts:

1. CLI arguments for `pipeline.hermes_orchestrator`, `run_full_auto_approve.sh`, `run_full_auto_correct_publish.sh`, and `run_orchestrator.sh`.
2. Human QA gates, when they trigger, and which artifacts they produce.
3. A complete representative log transcript for the whole pipeline, with success and blocker signatures.

Assume commands are run from the project root:

```bash
cd /Users/cantemir/Projects/business_success_stories
```

Use `python3`, not `python`, unless the environment explicitly provides a `python` alias.

---

## Part 1: Scripts And Arguments

### `pipeline.hermes_orchestrator`

Primary entry point:

```bash
python3 -m pipeline.hermes_orchestrator [OPTIONS]
```

The orchestrator acquires `state/locks/orchestrator.lock`, loads `state/episode_queue.json`, runs exactly one pending stage for one episode, saves queue state, and exits. Most long-running operation is inside the selected stage.

Default behavior with no flags:

```bash
python3 -m pipeline.hermes_orchestrator
```

Runs one pending stage for the next queue-head episode that is not blocked. If the queue is empty or fully blocked, it logs:

```text
queue empty or fully blocked; nothing to do
```

Global/common flags:

| Argument | Meaning | Agent action |
| --- | --- | --- |
| `-h`, `--help` | Print CLI help. | Safe, read-only. |
| `-v`, `--verbose` | Increase logging to DEBUG. Can be repeated. | Use when investigating unclear failures. |
| `--status` | Print queue status and exit. | Safe. Use before selecting an episode. |

Queue creation:

| Argument | Meaning | Example |
| --- | --- | --- |
| `--enqueue N` | Add `N` empty episode records. S01 will discover topics later. | `python3 -m pipeline.hermes_orchestrator --enqueue 1` |
| `--preview` | Modifier for `--enqueue` or `--inject-topic`. Marks new episode as preview mode. S06 generates Act 0 + Act 5 only, and S12 produces `final_preview.mp4`. | `python3 -m pipeline.hermes_orchestrator --enqueue 1 --preview` |
| `--narrator N_ID` | Modifier for `--enqueue` or `--inject-topic`. Pin narrator. Validated against `config.yaml`. | `--narrator N5` |
| `--archetype A_ID` | Modifier for `--enqueue` or `--inject-topic`. Pin archetype. | `--archetype A2` |
| `--visual-style V_ID` | Modifier for `--enqueue` or `--inject-topic`. Pin visual style. | `--visual-style V3` |

Manual topic injection:

```bash
python3 -m pipeline.hermes_orchestrator --inject-topic drafts/my_topic.json
```

Required JSON fields:

```json
{
  "company_name": "Example Co",
  "founder_or_protagonist": "Founder Name",
  "year_anchor": 2012,
  "story_kind": "rise_and_fall",
  "hq_country": "US",
  "hero": "The protagonist or company",
  "conflict": "The central conflict"
}
```

Optional JSON pins: `archetype`, `narrator`, `visual_style`.

| Argument | Meaning | Agent action |
| --- | --- | --- |
| `--inject-topic FILE` | Queue one episode from a manually authored incident JSON. | Validate JSON exists first. |
| `--no-validate` | With `--inject-topic`, skip SearXNG demand validation. | Use only when operator intentionally wants a niche topic. |
| `--preview` | With `--inject-topic`, create a short preview episode. | Useful for tone checks. |
| `--narrator`, `--archetype`, `--visual-style` | CLI pins override pins inside the JSON. | Use when operator requests a specific persona/style. |

Running a specific episode:

| Argument | Meaning | Example |
| --- | --- | --- |
| `--run-episode EP_ID` | Run one pending stage for a named episode, bypassing queue-head order. | `python3 -m pipeline.hermes_orchestrator --run-episode EP_004` |

If the named episode is blocked, the command logs that approval is required and exits without running a stage.

Approvals and review-gate helpers:

| Argument | Meaning | Safe for automation? |
| --- | --- | --- |
| `--approve EP_ID` | Clear any `needs_human` gates on the named episode. | Only after inspecting the blocker artifact. Do not use for build-stage failures unless operator explicitly accepts missing/bad artifacts. |
| `--auto-correct-s07 EP_ID` | Apply all `suggested_rewrite` entries from `02_script/brand_safety_flags.json` to `02_script/script.txt`, write `brand_safety_autocorrect.json`, and clear only S07. | Safe when S07 flags contain concrete rewrites and operator allows automatic text replacement. |
| `--auto-correct-s08 EP_ID` | Apply structured `original`/`replacement` suggestions found in `02_script/beat_sheet.json`, write `beat_sheet_autocorrect.json`, and clear only S08. | Safe only when suggestions are present and low risk. |
| `--auto-approve-s09 EP_ID` | Clear only the S09 visual brand-safety review gate. | Safe only when visual flags are acceptable or already manually reviewed. |

Reruns:

| Argument | Meaning | Example |
| --- | --- | --- |
| `--rerun-from EP_ID STAGE_ID` | Reset the named stage and all later stages to `pending`; set `current_stage` to `STAGE_ID`; clear blockers. Does not delete artifacts. | `python3 -m pipeline.hermes_orchestrator --rerun-from EP_004 S12` |

Stage names accepted by `--rerun-from`: `S1`, `S2`, `S3`, `S4`, `S5`, `S6`, `S7`, `S8`, `S9`, `S10`, `S11`, `S12`, `S13`. Numeric shorthand such as `12` is normalized to `S12`.

Common rerun choices:

| Goal | Command |
| --- | --- |
| Regenerate script and everything downstream | `--rerun-from EP_ID S6` |
| Regenerate beat sheet, images, audio, video, package | `--rerun-from EP_ID S8` |
| Regenerate missing/bad images from current beat sheet | `--rerun-from EP_ID S9` |
| Regenerate TTS/audio/video/package only | `--rerun-from EP_ID S10` |
| Regenerate final mix/video/package after music/SFX changes | `--rerun-from EP_ID S11` |
| Regenerate video from existing images/audio | `--rerun-from EP_ID S12` |
| Regenerate titles, thumbnails, Shorts, YouTube package inputs | `--rerun-from EP_ID S13` |

Single-image rerender:

```bash
python3 -m pipeline.hermes_orchestrator --rerender EP_004 BEAT_023
```

| Argument | Meaning |
| --- | --- |
| `--rerender EP_ID BEAT_ID` | Re-run S09 for one beat. Existing output is archived to `03_assets/quarantine/`. |
| `--from-edited-prompt` | With `--rerender`, re-read the prompt from `02_script/beat_sheet.json`. Use after manually editing the beat prompt. |
| `--force-grok` | With `--rerender`, bypass FLUX and generate directly with Grok. The output is promoted to `03_assets/flux/BEAT_ID.png`. |

Examples:

```bash
python3 -m pipeline.hermes_orchestrator --rerender EP_004 BEAT_023 --from-edited-prompt
python3 -m pipeline.hermes_orchestrator --rerender EP_004 BEAT_023 --force-grok
```

YouTube and performance:

| Argument | Meaning | Example |
| --- | --- | --- |
| `--authorize-youtube` | Run OAuth flow. Requires `YOUTUBE_OAUTH_CLIENT_ID` and `YOUTUBE_OAUTH_CLIENT_SECRET` in `.env`. Token goes to `state/youtube_oauth_token.json`. | `python3 -m pipeline.hermes_orchestrator --authorize-youtube` |
| `--set-video-id EP_ID YT_VIDEO_ID` | Bind an existing YouTube video ID to an episode for performance analysis. | `--set-video-id EP_004 abc123` |
| `--analyse-performance` | Run off-band S14 for episodes with `youtube_video_id`; writes performance JSON. | `python3 -m pipeline.hermes_orchestrator --analyse-performance` |
| `--build-youtube-package EP_ID` | Build local review package under `06_metadata/youtube_upload_package/`; no upload. | `--build-youtube-package EP_004` |
| `--upload-youtube-package EP_ID --approve-youtube-upload` | Upload an existing package. Approval flag is mandatory. | `--upload-youtube-package EP_004 --approve-youtube-upload` |
| `--backfill-youtube-captions EP_ID --approve-youtube-upload` | Upload packaged captions to existing uploaded video IDs without re-uploading videos. | `--backfill-youtube-captions EP_004 --approve-youtube-upload` |
| `--youtube-privacy private|unlisted|public` | Optional upload privacy override. | `--youtube-privacy unlisted` |
| `--youtube-publish-at TIMESTAMP` | Optional scheduled publish time, RFC3339. Forces private-until-publish behavior. | `--youtube-publish-at 2026-06-05T18:00:00Z` |
| `--youtube-caption-target all|long|shorts|short_NN` | Caption backfill target. | `--youtube-caption-target short_02` |
| `--youtube-caption-language LANG` | Caption backfill language. Repeat for multiple languages. Defaults to all packaged tracks. | `--youtube-caption-language es --youtube-caption-language fr` |

YouTube package outputs:

```text
episodes/EP_NNN_slug/06_metadata/youtube_upload_package/
  package_manifest.json
  PACKAGE_SUMMARY.md
  long_form/
  shorts/
  thumbnails/
```

Exit-code interpretation:

| Exit code | Meaning |
| --- | --- |
| `0` | Command completed, or no work was available. Check logs and queue state for actual stage outcome. |
| `1` | Partial auto-correction or non-fatal helper issue. Inspect command output. |
| `2` | CLI error, missing file, stage exception, lock contention in approval helper, or upload/package failure. Inspect stderr and logs. |

### `run_full_auto_approve.sh`

Foreground runner:

```bash
./run_full_auto_approve.sh [EP_ID]
```

Purpose: run a selected episode continuously until `05_video/final.mp4` exists. It repeatedly invokes:

```bash
python3 -m pipeline.hermes_orchestrator --run-episode EP_ID
```

It logs to:

```text
logs/full_auto.<UTC_TIMESTAMP>.log
```

Behavior with no argument:

1. Select first non-DONE episode from the queue.
2. If no queued episode exists, enqueue one episode.
3. Loop until final video exists, a non-review blocker appears, or `MAX_ITERATIONS` is reached.

Behavior with `EP_ID`:

1. Select exactly that episode.
2. Run it even if another older episode exists in the queue.
3. Stop if the selected episode is not found.

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PYTHON_BIN` | `python3` | Python executable used for orchestrator calls. |
| `MAX_ITERATIONS` | `80` | Maximum loop count before giving up. |
| `SLEEP_SECONDS` | `2` | Delay between loop iterations. |
| `AUTO_APPROVE_STAGES` | `S7 S8 S9` | Blocked stages that the script may auto-approve. |

Examples:

```bash
./run_full_auto_approve.sh
./run_full_auto_approve.sh EP_004
MAX_ITERATIONS=120 SLEEP_SECONDS=5 ./run_full_auto_approve.sh EP_004
AUTO_APPROVE_STAGES="S7 S8 S9" ./run_full_auto_approve.sh EP_004
```

Important safety behavior:

```text
Episode EP_004 is blocked at S12; not auto-approving build-stage failures.
Review the blocker in state/episode_queue.json and the latest log before rerunning.
```

This is correct. S10+ failures usually mean missing audio/video/package artifacts. Do not override with `--approve` unless the operator explicitly accepts the broken or missing artifact.

Success signature:

```text
final video ready: /.../episodes/EP_004_slug/05_video/final.mp4
```

Failure signatures:

```text
Episode EP_004 is not in the queue.
Episode EP_004 is DONE but final.mp4 was not found.
Stopped after MAX_ITERATIONS=80 without producing final.mp4 for EP_004.
orchestrator returned non-zero; checking queue state on next loop
```

### `run_full_auto_correct_publish.sh`

Foreground build-and-upload runner:

```bash
./run_full_auto_correct_publish.sh [EP_ID]
```

Purpose: run an episode continuously until `05_video/final.mp4` exists, auto-handle review gates with stage-specific actions, then build and upload the YouTube package.

Review-gate policy:

| Stage | Action |
| --- | --- |
| `S7` | Run `--auto-correct-s07 EP_ID`; if no rewrite applies, fall back to `--approve EP_ID` so review-stage automation keeps moving. |
| `S8` | Run `--auto-correct-s08 EP_ID`. |
| `S9` | Run `--auto-approve-s09 EP_ID`. |
| `S10+` | Stop. These are build failures, not review gates. |

After `final.mp4` exists, the script runs:

```bash
python3 -m pipeline.hermes_orchestrator --build-youtube-package EP_ID
python3 -m pipeline.hermes_orchestrator --upload-youtube-package EP_ID --approve-youtube-upload
```

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PYTHON_BIN` | `python3` | Python executable used for orchestrator calls. |
| `MAX_ITERATIONS` | `100` | Maximum loop count before giving up. |
| `SLEEP_SECONDS` | `2` | Delay between loop iterations. |
| `YOUTUBE_PRIVACY` | unset | Optional upload privacy override: `private`, `unlisted`, or `public`. |
| `YOUTUBE_PUBLISH_AT` | unset | Optional RFC3339 scheduled publish time passed to upload. |

Examples:

```bash
./run_full_auto_correct_publish.sh EP_007
YOUTUBE_PRIVACY=private ./run_full_auto_correct_publish.sh EP_007
YOUTUBE_PUBLISH_AT=2026-06-08T18:00:00Z ./run_full_auto_correct_publish.sh EP_007
```

Success signatures:

```text
final video ready: /.../05_video/final.mp4
building YouTube package for EP_007
uploading YouTube package for EP_007
completed build/package/upload for EP_007
```

### `run_orchestrator.sh`

Cron/launchd-friendly wrapper:

```bash
./run_orchestrator.sh [ANY pipeline.hermes_orchestrator OPTIONS]
```

It derives the project root from its own path, activates `.venv` if present, exports:

```bash
PIPELINE_CONFIG="${ROOT}/config.yaml"
```

Then detaches the Python process:

```bash
nohup python3 -m pipeline.hermes_orchestrator "$@" </dev/null >>"${LOGFILE}" 2>&1 &
```

It immediately prints:

```text
orchestrator pid=12345 -> /.../logs/run.2026-06-03T08-00-00Z.log
```

Use it for scheduler-style operation where the shell must not kill Python mid-stage.

Examples:

```bash
./run_orchestrator.sh
./run_orchestrator.sh --status
./run_orchestrator.sh --run-episode EP_004
./run_orchestrator.sh --build-youtube-package EP_004
```

Log files:

| Wrapper | Log path |
| --- | --- |
| `run_orchestrator.sh` | `logs/run.<timestamp>.log` plus daily orchestrator log `logs/orch.YYYY-MM-DD.log` |
| `run_full_auto_approve.sh` | `logs/full_auto.<timestamp>.log` plus daily orchestrator log |
| `run_full_auto_correct_publish.sh` | `logs/full_auto_correct_publish.<timestamp>.log` plus daily orchestrator log |
| Direct `python3 -m pipeline.hermes_orchestrator` | terminal output plus daily orchestrator log |

Lock behavior:

If a previous stage is still running, another wrapper invocation should log:

```text
lock contention: lock already held: /.../state/locks/orchestrator.lock
```

This is not a failure. The autonomous agent should wait and check again later.

---

## Part 2: Human QA Gates And Artifacts

### Gate Taxonomy

The orchestrator treats any stage return string as a `needs_human` blocker. There are two classes:

1. Review gates: expected QA holds where artifacts should be inspected and then approved or corrected.
2. Build/data failures: missing files, failed ffmpeg, failed TTS, timeline mismatch, upload failures, etc. These should usually stop the runner.

Default review gates:

| Stage | Gate | Default behavior | Approval helper |
| --- | --- | --- | --- |
| S07 | Script brand-safety | Enabled. Gates on high severity by default. | `--approve`, or `--auto-correct-s07` if suggested rewrites exist. |
| S08 | Beat-sheet review | Disabled by default via `orchestrator.gate_at_S08: false`. | `--approve`, or `--auto-correct-s08` if structured replacements exist. |
| S09 | Visual brand-safety | Enabled. Gates on high severity by default. | `--auto-approve-s09`, `--approve`, or rerender flagged beats first. |

Do not auto-approve build/data failures unless explicitly instructed. Examples: S10 no chunks, S11 mix failure, S12 timeline mismatch, missing image, failed final concat, S13 packaging exception.

### Queue State For Blockers

All stage blockers are recorded in:

```text
state/episode_queue.json
```

Look for:

```json
{
  "id": "EP_004",
  "current_stage": "S9",
  "blockers": [
    {
      "stage": "S9",
      "reason": "visual brand-safety: 1 high-severity flag(s) require review..."
    }
  ],
  "stages": {
    "S9": {
      "status": "needs_human"
    }
  }
}
```

The command:

```bash
python3 -m pipeline.hermes_orchestrator --status
```

prints compact status, including safety counts when present:

```text
EP_004    stage=S9    BLOCKED  Vine, Inc. (visual_safety=1H/2L)
```

### S07 Script Brand-Safety Gate

Stage:

```text
S7 (Script Critique)
```

Main artifacts:

```text
episodes/EP_NNN_slug/02_script/script.txt
episodes/EP_NNN_slug/02_script/critique_history.json
episodes/EP_NNN_slug/02_script/brand_safety_flags.json
episodes/EP_NNN_slug/02_script/brand_safety_autocorrect.json   # only after --auto-correct-s07
```

What S07 does:

1. Critiques script voice and structure.
2. Applies critique rewrites where possible.
3. Runs brand-safety review against the script and fact ledger.
4. Writes `brand_safety_flags.json` every time the brand-safety pass runs.

`brand_safety_flags.json` shape:

```json
{
  "verdict": "review_required",
  "flags": [
    {
      "severity": "high",
      "sentence": "Flagged sentence from script.",
      "reasoning": "Why this could be unsafe or insufficiently supported.",
      "suggested_rewrite": "Safer replacement sentence."
    }
  ],
  "high_severity_count": 1,
  "low_severity_count": 0
}
```

Clean or skipped shape:

```json
{
  "verdict": "clean",
  "flags": [],
  "high_severity_count": 0,
  "low_severity_count": 0
}
```

Gate trigger:

```text
stage S7 (Script Critique) needs_human for EP_004: brand-safety: 1 high-severity flag(s) require review. Inspect 02_script/brand_safety_flags.json then run `--approve EP_004` to clear.
```

Agent decision tree:

1. Open `02_script/brand_safety_flags.json`.
2. If every flag has a good `suggested_rewrite`, run:

   ```bash
   python3 -m pipeline.hermes_orchestrator --auto-correct-s07 EP_004
   ```

3. If flags require judgment, summarize them for the operator.
4. If the operator accepts the script as-is, run:

   ```bash
   python3 -m pipeline.hermes_orchestrator --approve EP_004
   ```

Do not silently delete flags. The artifact is the audit trail.

### S08 Beat-Sheet Review Gate

Stage:

```text
S8 (Beat Sheet)
```

Main artifacts:

```text
episodes/EP_NNN_slug/02_script/beat_sheet.json
episodes/EP_NNN_slug/02_script/beat_sheet_raw.txt              # only when raw invalid output is saved
episodes/EP_NNN_slug/02_script/beat_sheet_autocorrect.json     # only after --auto-correct-s08
```

What S08 does:

1. Converts the script into beats.
2. Groups beats into visual scenes.
3. Builds `visual_continuity_plan`.
4. Assigns visual intents, durations, prompts, callouts, SFX cues, and asset references.
5. Writes `beat_sheet.json`.

`beat_sheet.json` top-level shape:

```json
{
  "beats": [
    {
      "beat_id": "BEAT_01",
      "estimated_seconds": 9.3,
      "visual_intent": "founder_portrait",
      "scene_id": "SCENE_01",
      "description": "What the viewer sees.",
      "flux_render_request": {
        "prompt": "Movie-shot prompt for FLUX/Grok.",
        "negative_prompt": "Things to avoid."
      },
      "callouts": [
        {
          "text": "Key number",
          "start_seconds": 1.2,
          "duration_seconds": 2.5
        }
      ]
    }
  ],
  "visual_continuity_plan": {
    "scenes": []
  },
  "total_estimated_seconds": 620.4,
  "matched_pd_count": 0,
  "flux_needed_count": 68
}
```

Gate trigger only if:

```yaml
orchestrator:
  gate_at_S08: true
```

Gate log:

```text
stage S8 (Beat Sheet) needs_human for EP_004: S08 gate enabled: review 02_script/beat_sheet.json (68 beats, 0 PD, 68 FLUX; distribution: business_context=20, legal_fraud=12) then run `--approve EP_004` to advance to S09.
```

Agent decision tree:

1. Inspect `02_script/beat_sheet.json`.
2. Confirm beat count, durations, callouts, scene continuity, visual prompts, repeated props, and SFX cues.
3. If structured replacement suggestions exist inside the JSON, run:

   ```bash
   python3 -m pipeline.hermes_orchestrator --auto-correct-s08 EP_004
   ```

4. Otherwise ask the operator or approve:

   ```bash
   python3 -m pipeline.hermes_orchestrator --approve EP_004
   ```

### S09 Visual Brand-Safety Gate

Stage:

```text
S9 (FLUX Render)
```

Main artifacts:

```text
episodes/EP_NNN_slug/03_assets/flux/BEAT_NN.png
episodes/EP_NNN_slug/03_assets/grok/*                          # Grok archives/corrections/forced renders
episodes/EP_NNN_slug/03_assets/quarantine/                     # archived old renders
episodes/EP_NNN_slug/03_assets/asset_manifest.json
episodes/EP_NNN_slug/03_assets/visual_brand_safety_flags.json
episodes/EP_NNN_slug/03_assets/flux/title.png
episodes/EP_NNN_slug/03_assets/flux/credits.png                # only when closing card enabled
```

What S09 does:

1. Renders title image.
2. Renders beat images via configured backend:
   - `image_generation.backend: both`: FLUX first, Grok fallback when VLM flags issues.
   - `image_generation.backend: flux`: FLUX only.
   - `image_generation.backend: grok`: Grok only.
3. If Grok returns a moderation-style HTTP 400, retries once with a sanitized generic editorial prompt.
4. Uses VLM image QA.
5. Promotes chosen images to `03_assets/flux/`.
6. Runs visual brand-safety review.
7. Writes `visual_brand_safety_flags.json`.

`visual_brand_safety_flags.json` shape:

```json
{
  "verdict": "ship_blocker",
  "high_severity_count": 1,
  "low_severity_count": 2,
  "checked_count": 34,
  "flags": [
    {
      "beat_id": "BEAT_17",
      "image_path": "03_assets/flux/BEAT_17.png",
      "severity": "high",
      "category": "brand_misrepresentation",
      "issue": "The image shows a fake app logo that could misrepresent the company.",
      "suggestion": "Rerender with no fake UI text or generic logos."
    }
  ]
}
```

Gate trigger:

```text
stage S9 (FLUX Render) needs_human for EP_004: visual brand-safety: 1 high-severity flag(s) require review. Inspect 03_assets/visual_brand_safety_flags.json then run `--approve EP_004` to clear (or `--rerender` the specific beats first).
```

Agent decision tree:

1. Inspect `03_assets/visual_brand_safety_flags.json`.
2. For high severity flags on specific beats, prefer rerendering:

   ```bash
   python3 -m pipeline.hermes_orchestrator --rerender EP_004 BEAT_17 --force-grok
   ```

3. If the prompt was edited in `beat_sheet.json`, use:

   ```bash
   python3 -m pipeline.hermes_orchestrator --rerender EP_004 BEAT_17 --from-edited-prompt
   ```

4. After rerenders, either rerun S09 from the same stage or approve if the operator accepts:

   ```bash
   python3 -m pipeline.hermes_orchestrator --auto-approve-s09 EP_004
   ```

5. Do not approve images that misrepresent logos, people, product UIs, legal facts, or brands unless explicitly accepted by the operator.

### Build/Data Failure Blockers

These are not review gates, but they become `needs_human` because the stage returned a reason or raised an exception.

Common blockers:

| Stage | Example reason | Agent action |
| --- | --- | --- |
| S01 | `manual injection demand-gate failed...` | Use `--no-validate` only if operator wants the niche topic; otherwise let S01 find another topic. |
| S02 | too few sources / no tier-1 sources | Check SearXNG, topic quality, internet access, source inventory. |
| S03 | no source inventory | Rerun S02 or inspect `00_research/`. |
| S04 | not enough verified claims | Inspect `01_factcheck/fact_ledger.json`, rerun S2-S4 if source quality is poor. |
| S05 | no episode workspace | Queue/workspace corruption; inspect state. |
| S06 | forbidden phrase, bad word count, too few/many beats | Rerun S06; if repeated, inspect prompts/config and script logs. |
| S10 | `TTS produced no chunks` | Check the configured TTS backend and `04_audio/chunks/`. |
| S11 | `audio_post_mix failed...` | Check ffmpeg, music/SFX files, `04_audio/final_mix.wav`. |
| S12 | `video/audio timeline mismatch`, `final concat failed`, missing image | Stop. Inspect `05_video/clips/`, `voice_timing.json`, `final_mix.wav`, ffmpeg logs. |
| S13 | Shorts/thumbnail/package failures | Inspect `05_video/shorts/`, `06_metadata/`, ffmpeg logs. |

For build/data blockers, prefer fixing the root cause and using `--rerun-from`, not `--approve`.

---

## Part 3: Complete Mock Log Output

This is a representative transcript. Exact counts, durations, seeds, file names, and timestamps will vary.

### 0. Enqueue

Command:

```bash
python3 -m pipeline.hermes_orchestrator --enqueue 1
```

Output:

```text
enqueued 1: EP_004
```

Success signal: an episode ID is printed.

### 1. Full Auto Runner Start

Command:

```bash
./run_full_auto_approve.sh EP_004
```

Output:

```text
full-auto runner log: /Users/cantemir/Projects/business_success_stories/logs/full_auto.2026-06-03T08-00-00Z.log
selected episode: EP_004
iteration 1/80: EP_004 stage=S1 blocked=false final=false
```

Success signal: `selected episode: EP_004`.

### 2. S01 Topic Discovery

Stage-begin signature:

```text
2026-06-03 08:00:01,010 hermes.orchestrator INFO running S1 (Topic Discovery) for episode EP_004 (topic=<unset>)
```

Representative stage logs:

```text
2026-06-03 08:00:01,219 hermes.stage.s01 INFO S01 attempt 1/100: asking writer for candidate topics
2026-06-03 08:00:18,604 hermes.stage.s01 INFO S01 candidate: Vine, Inc. [US, 2012, rise_and_fall]
2026-06-03 08:00:18,799 hermes.trends INFO demand validation: youtube=1840 recent_news=8
2026-06-03 08:00:18,802 hermes.stage.s01 INFO S01 selected assignment: archetype=A3 narrator=N5 visual_style=V2
2026-06-03 08:00:23,412 hermes.stage.s01 INFO iconic_assets derived: 6 entries for Vine, Inc.
```

Stage-success signatures:

```text
2026-06-03 08:00:23,430 hermes.orchestrator INFO stage S1 (Topic Discovery) done for EP_004
iteration 2/80: EP_004 stage=S2 blocked=false final=false
```

Failure/blocker examples:

```text
2026-06-03 08:00:23,430 hermes.orchestrator WARNING stage S1 (Topic Discovery) needs_human for EP_004: manual injection demand-gate failed: youtube count below minimum...
```

Artifacts:

```text
episodes/EP_004_slug/00_research/incident.json
episodes/EP_004_slug/00_research/iconic_assets.json
state/used_topics.json
state/episode_queue.json
```

### 3. S02 Source Gathering

Stage-begin signature:

```text
2026-06-03 08:00:25,011 hermes.orchestrator INFO running S2 (Source Gathering) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 08:00:25,126 hermes.stage.s02 INFO S02: company='Vine, Inc.' year=2012 founder='Dom Hofmann, Rus Yusupov, Colin Kroll' story=rise_and_fall
2026-06-03 08:00:25,127 hermes.stage.s02 INFO S02: 12 recipes
2026-06-03 08:00:25,500 hermes.stage.s02 INFO [recipe 1/12 company_history] "Vine Inc history acquisition Twitter"
2026-06-03 08:00:28,345 hermes.stage.s02 INFO captured open_tier1 (tier=open_tier1, 1800 words) https://example.com/source-1
2026-06-03 08:00:31,902 hermes.stage.s02 INFO captured open_tier2 (tier=open_tier2, 1250 words) https://example.com/source-2
2026-06-03 08:00:42,115 hermes.stage.s02 INFO S02 complete: 18 sources (4 open_tier1)
```

Stage-success signature:

```text
2026-06-03 08:00:42,132 hermes.orchestrator INFO stage S2 (Source Gathering) done for EP_004
```

Blocker examples:

```text
2026-06-03 08:00:42,132 hermes.orchestrator WARNING stage S2 (Source Gathering) needs_human for EP_004: only 2 sources captured; need 6
```

Artifacts:

```text
00_research/source_inventory.json
00_research/raw/*.txt
00_research/extracted/*
```

### 4. S03 Fact Extraction

Stage-begin signature:

```text
2026-06-03 08:00:44,002 hermes.orchestrator INFO running S3 (Fact Extraction) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 08:00:44,120 hermes.stage.s03 INFO extracting from source_001.txt (3 chunks, tier=open_tier1)
2026-06-03 08:01:02,718 hermes.stage.s03 INFO extracting from source_002.txt (2 chunks, tier=open_tier2)
2026-06-03 08:01:40,225 hermes.stage.s03 INFO S03 complete: 142 raw facts from 18 sources
2026-06-03 08:01:48,441 hermes.stage.s03 INFO S03 HQ: New York, US, 2012 (conf=0.82, method=llm_consolidation)
```

Stage-success signature:

```text
2026-06-03 08:01:48,500 hermes.orchestrator INFO stage S3 (Fact Extraction) done for EP_004
```

Artifacts:

```text
01_factcheck/raw_facts.json
01_factcheck/company_hq.json
```

### 5. S04 Fact Verification

Stage-begin signature:

```text
2026-06-03 08:01:50,100 hermes.orchestrator INFO running S4 (Fact Verification) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 08:01:50,312 hermes.stage.s04 INFO S04 merge: 95 total claims from 5 batches
2026-06-03 08:02:12,884 hermes.stage.s04 INFO rejected by skeptic (unsupported): claim text...
2026-06-03 08:02:35,004 hermes.stage.s04 INFO S04 complete: 64 verified claims (need 5)
```

Stage-success signature:

```text
2026-06-03 08:02:35,041 hermes.orchestrator INFO stage S4 (Fact Verification) done for EP_004
```

Artifacts:

```text
01_factcheck/fact_ledger.json
```

### 6. S05 PD Asset Hunt

Stage-begin signature:

```text
2026-06-03 08:02:37,100 hermes.orchestrator INFO running S5 (PD Asset Hunt) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 08:02:37,412 hermes.stage.s05 INFO S05 company logo: https://example.com/logo.png -> /.../03_assets/pd/company_logo.png
2026-06-03 08:02:37,550 hermes.stage.s05 INFO S05: asset_hunt.enabled=false - skipping PD hunt; retaining title-logo discovery
2026-06-03 08:02:39,901 hermes.stage.s05 INFO S05 character profile: founder/protagonist iconography saved
2026-06-03 08:02:39,955 hermes.stage.s05 INFO S05 complete: 1 assets total (1 incident-specific, 0 stash)
```

Stage-success signature:

```text
2026-06-03 08:02:40,000 hermes.orchestrator INFO stage S5 (PD Asset Hunt) done for EP_004
```

Artifacts:

```text
03_assets/pd/company_logo.png
03_assets/title_logo.json
03_assets/asset_manifest.json
01_factcheck/character_profile.json
06_metadata/license_attributions.txt
```

### 7. S06 Script Generation

Stage-begin signature:

```text
2026-06-03 08:02:42,010 hermes.orchestrator INFO running S6 (Script Generation) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 08:02:42,121 hermes.stage.s06 INFO S06 staged generation attempt 1: creating blueprint
2026-06-03 08:03:05,992 hermes.stage.s06 INFO S06 staged generation attempt 1: ACT_0 target=57 words/2 beats
2026-06-03 08:04:10,730 hermes.stage.s06 INFO S06 staged attempt 1: 1588 words (dist=0, in_range=True, forbidden_hits=0)
2026-06-03 08:04:11,200 hermes.stage.s06 INFO S06 forbidden-phrase substitutions applied: 0
2026-06-03 08:04:11,550 hermes.stage.s06 INFO S06 complete: 1604 words, 44 beats
```

Stage-success signature:

```text
2026-06-03 08:04:11,602 hermes.orchestrator INFO stage S6 (Script Generation) done for EP_004
```

Blocker examples:

```text
2026-06-03 08:04:11,602 hermes.orchestrator WARNING stage S6 (Script Generation) needs_human for EP_004: script word count 1541 outside 2090-2510 (after retry)
2026-06-03 08:04:11,602 hermes.orchestrator WARNING stage S6 (Script Generation) needs_human for EP_004: forbidden phrase reintroduced by length retry: ['the lesson is clear']
```

Artifacts:

```text
02_script/script.txt
02_script/script_meta.json
02_script/script_blueprint.json
02_script/script_blueprint_prompt.txt
02_script/script_act_prompts.txt
```

### 8. S07 Script Critique And Brand-Safety

Stage-begin signature:

```text
2026-06-03 08:04:13,100 hermes.orchestrator INFO running S7 (Script Critique) for episode EP_004 (topic=Vine, Inc.)
```

Clean representative logs:

```text
2026-06-03 08:04:13,421 hermes.stage.s07 INFO S07 critique pass on loop 1
2026-06-03 08:04:31,210 hermes.stage.s07 INFO S07 applied rewrite (exact): sharpen opening hook
2026-06-03 08:04:48,771 hermes.stage.s07 INFO S07 brand-safety: verdict=clean flags=0H/0L
2026-06-03 08:04:48,800 hermes.orchestrator INFO stage S7 (Script Critique) done for EP_004
```

Gate representative logs:

```text
2026-06-03 08:04:48,771 hermes.stage.s07 INFO S07 brand-safety: verdict=review_required flags=1H/1L
2026-06-03 08:04:48,772 hermes.stage.s07 INFO   [high] The founder lied to investors... - Direct accusation needs attribution or softer wording.
2026-06-03 08:04:48,800 hermes.orchestrator WARNING stage S7 (Script Critique) needs_human for EP_004: brand-safety: 1 high-severity flag(s) require review. Inspect 02_script/brand_safety_flags.json then run `--approve EP_004` to clear.
iteration 8/80: EP_004 stage=S7 blocked=true final=false
auto-approving 1 blocker(s) for EP_004
approved EP_004: blockers cleared; current_stage=S8
```

Artifacts:

```text
02_script/script.txt
02_script/critique_history.json
02_script/brand_safety_flags.json
02_script/brand_safety_autocorrect.json
```

### 9. S08 Beat Sheet

Stage-begin signature:

```text
2026-06-03 08:04:50,020 hermes.orchestrator INFO running S8 (Beat Sheet) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 08:04:50,540 hermes.stage.s08 INFO S08 visual continuity: grouped 68 beats into 12 scenes
2026-06-03 08:05:10,880 hermes.stage.s08 INFO S08 narrow-PD filter: ['founder_portrait', 'product_reveal']
2026-06-03 08:05:11,102 hermes.stage.s08 INFO S08 iconography: injected into 18 hero-beat FLUX prompts
2026-06-03 08:05:11,210 hermes.stage.s08 INFO S08 complete: 68 beats, 0 direct PD, 68 FLUX
```

Stage-success signature:

```text
2026-06-03 08:05:11,245 hermes.orchestrator INFO stage S8 (Beat Sheet) done for EP_004
```

Optional S08 gate:

```text
2026-06-03 08:05:11,245 hermes.orchestrator WARNING stage S8 (Beat Sheet) needs_human for EP_004: S08 gate enabled: review 02_script/beat_sheet.json (68 beats, 0 PD, 68 FLUX; distribution: business_context=18, legal_fraud=9) then run `--approve EP_004` to advance to S09.
```

Artifacts:

```text
02_script/beat_sheet.json
02_script/beat_sheet_raw.txt
02_script/beat_sheet_autocorrect.json
```

### 10. S09 Image Rendering

Stage-begin signature:

```text
2026-06-03 08:05:13,100 hermes.orchestrator INFO running S9 (FLUX Render) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 08:05:13,210 hermes.stage.s09 INFO S09: image backend=both, image QA enabled, max_attempts_per_beat=1, strict_borderline=false
2026-06-03 08:05:13,300 hermes.stage.s09 INFO S09 title card rendered: /.../03_assets/flux/title.png
2026-06-03 08:05:13,522 hermes.flux INFO flux (BEAT_01 seed=168330011) 1920x1080 steps=24 -> BEAT_01_a0.png
2026-06-03 08:10:59,225 hermes.stage.s09 INFO S09 BEAT_01 a1 (seed=168330011): verdict=pass score=9 match=9 anatomy=ok
2026-06-03 08:10:59,226 hermes.stage.s09 INFO S09 BEAT_01 a1 (seed=168330011) reasoning: Strong topic match and clean comic-book composition.
2026-06-03 08:11:00,100 hermes.flux INFO flux (BEAT_02 seed=221992045) 1920x1080 steps=24 -> BEAT_02_a0.png
2026-06-03 08:16:44,332 hermes.stage.s09 WARNING S09 BEAT_02: all 1 attempts rejected; keeping best (seed=221992045)
2026-06-03 08:16:44,991 hermes.stage.s09 INFO S09 BEAT_02: VLM flagged issues - routing to Grok. triggers=text,anatomy
2026-06-03 08:16:58,345 hermes.stage.s09 INFO S09 BEAT_02: Grok corrected (triggers=text,anatomy)
2026-06-25 06:34:23,883 hermes.grok WARNING grok 400 for BEAT_03_a0.png; body: {"error":"Generated image rejected by content moderation."}
2026-06-25 06:34:23,884 hermes.stage.s09 WARNING S09 BEAT_03: Grok moderation rejected prompt; retrying sanitized prompt
...
2026-06-03 11:52:00,140 hermes.stage.s09 INFO S09 complete: 68 rendered (8 kept-from-rejected), 0 failed
2026-06-03 11:52:12,993 hermes.stage.s09 INFO S09 visual-safety [low]: BEAT_22 - Minor generic interface visible in background.
2026-06-03 11:52:50,445 hermes.stage.s09 INFO S09 visual brand-safety: 0 high, 1 low, 33 clean (of 34 checked)
```

Stage-success signature:

```text
2026-06-03 11:52:50,500 hermes.orchestrator INFO stage S9 (FLUX Render) done for EP_004
```

Visual gate logs:

```text
2026-06-03 11:52:12,993 hermes.stage.s09 INFO S09 visual-safety [high]: BEAT_17 - The image contains a fake app logo and misleading UI.
2026-06-03 11:52:50,445 hermes.stage.s09 INFO S09 visual brand-safety: 1 high, 2 low, 31 clean (of 34 checked)
2026-06-03 11:52:50,500 hermes.orchestrator WARNING stage S9 (FLUX Render) needs_human for EP_004: visual brand-safety: 1 high-severity flag(s) require review. Inspect 03_assets/visual_brand_safety_flags.json then run `--approve EP_004` to clear (or `--rerender` the specific beats first).
iteration 11/80: EP_004 stage=S9 blocked=true final=false
auto-approving 1 blocker(s) for EP_004
approved EP_004: blockers cleared; current_stage=S10
```

Artifacts:

```text
03_assets/flux/title.png
03_assets/flux/BEAT_01.png
03_assets/flux/*.png
03_assets/grok/*.png
03_assets/quarantine/*
03_assets/visual_brand_safety_flags.json
03_assets/asset_manifest.json
```

### 11. S10 TTS Render

Stage-begin signature:

```text
2026-06-03 11:52:52,100 hermes.orchestrator INFO running S10 (TTS Render) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 11:52:52,180 hermes.tts INFO TTS backend: Chatterbox (voice=bf_emma, model=chatterbox-turbo-4bit)
2026-06-03 11:52:52,220 hermes.chatterbox INFO chatterbox: 180 words, voice=bf_emma, model=chatterbox-turbo-4bit, speed=1.00 -> chunk_000.wav
2026-06-03 11:52:57,770 hermes.chatterbox INFO chatterbox: 174 words, voice=bf_emma, model=chatterbox-turbo-4bit, speed=1.00 -> chunk_001.wav
2026-06-03 11:53:03,120 hermes.chatterbox INFO chatterbox: 168 words, voice=bf_emma, model=chatterbox-turbo-4bit, speed=1.00 -> chunk_002.wav
2026-06-03 11:53:55,550 hermes.stage.s10 INFO voice_full.wav: 1342.3s
2026-06-03 11:53:55,640 hermes.stage.s10 INFO S10 complete: 68 beats, 1342.3s voice
```

Stage-success signature:

```text
2026-06-03 11:53:55,700 hermes.orchestrator INFO stage S10 (TTS Render) done for EP_004
```

Warning that does not necessarily block:

```text
2026-06-03 11:53:55,560 hermes.stage.s10 WARNING voice duration 617.3s outside target 1100+/-210
```

Blocker:

```text
2026-06-03 11:53:55,700 hermes.orchestrator WARNING stage S10 (TTS Render) needs_human for EP_004: TTS produced no chunks
```

Artifacts:

```text
04_audio/chunks/chunk_000.wav
04_audio/voice_full.wav
04_audio/voice_timing.json
```

### 12. S11 Audio Post

Stage-begin signature:

```text
2026-06-03 11:53:57,100 hermes.orchestrator INFO running S11 (Audio Post) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 11:53:57,245 hermes.stage.s11 INFO voice duration: 1342.3s
2026-06-03 11:53:57,300 hermes.stage.s11 INFO S11 voice padded: +1.0s head / +5.0s tail (total 1348.3s for title+voice+closing cover)
2026-06-03 11:53:57,410 hermes.music_library INFO music_library: loaded 13 tracks
2026-06-03 11:53:57,550 hermes.music_library INFO music_library: picked 10 tracks (target=1384s, picked=1422s)
2026-06-03 11:53:59,420 hermes.stage.s11 INFO S11 SFX: 12 cues placed in sfx_track.wav
2026-06-03 11:54:36,900 hermes.stage.s11 INFO S11 music bed assembled: 10 tracks
2026-06-03 11:55:20,012 hermes.stage.s11 INFO S11 complete: final_mix.wav (10 music tracks, 12 SFX cues, voice-only=False)
```

Stage-success signature:

```text
2026-06-03 11:55:20,060 hermes.orchestrator INFO stage S11 (Audio Post) done for EP_004
```

Non-blocking warning:

```text
2026-06-03 11:53:57,431 hermes.music_library WARNING music_library: track file missing on disk: /.../assets/music_library/file.mp3
```

Blocker:

```text
2026-06-03 11:55:20,060 hermes.orchestrator WARNING stage S11 (Audio Post) needs_human for EP_004: audio_post_mix failed: ffmpeg returned non-zero
```

Artifacts:

```text
04_audio/voice_padded.wav
04_audio/music_bed.wav
04_audio/sfx_track.wav
04_audio/final_mix.wav
04_audio/mix_manifest.json
```

### 13. S12 Video Assembly

Stage-begin signature:

```text
2026-06-03 11:55:22,100 hermes.orchestrator INFO running S12 (Video Assembly) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 11:55:22,140 hermes.stage.s12 INFO S12 ASR alignment: transcribed 412 segments -> voice_asr_segments.json
2026-06-03 11:55:22,220 hermes.stage.s12 INFO S12 L1: purged 2 cached _callout.mp4 clips
2026-06-03 11:55:22,350 hermes.stage.s12 INFO S12 opening title card: 1.0s
2026-06-03 11:56:20,100 hermes.ffmpeg INFO render_ken_burns: BEAT_01.mp4 <- BEAT_01.png
2026-06-03 11:57:08,600 hermes.ffmpeg INFO composite_callouts: BEAT_12_callout.mp4 <- 1 overlays via BEAT_12.mp4 (fonts=/System/Library/Fonts/Supplemental/Impact.ttf @ 108 px, variants=corner_ribbon)
2026-06-03 12:04:20,700 hermes.stage.s12 INFO S12 conclusion music tail: 5.0s using final beat image
2026-06-03 12:04:21,100 hermes.stage.s12 INFO S12 closing card: disabled (production.closing_card_enabled=false or seconds=0)
2026-06-03 12:04:22,500 hermes.stage.s12 INFO S12 like/subscribe outro: disabled (production.append_like_subscribe_clip=false)
2026-06-03 12:04:44,340 hermes.ffmpeg INFO concat_clips: final_video.mp4 from 71 clips
2026-06-03 12:05:01,800 hermes.ffmpeg INFO mux_audio_video: final.mp4
2026-06-03 12:05:02,410 hermes.stage.s12 INFO S12 complete: final.mp4 (1355.3s)
```

ASR fallback signatures:

```text
hermes.asr WARNING ASR: whisper-cli not on PATH; falling back to estimated subtitle/callout timings.
hermes.asr WARNING ASR: input audio not found: /.../04_audio/voice_full.wav
hermes.asr WARNING ASR: whisper.cpp model not found: /.../models/whisper/ggml-base.en.bin
```

Direct whisper.cpp smoke test:

```bash
VOICE=$(find episodes -path '*/04_audio/voice_full.wav' | head -1)
whisper-cli -m models/whisper/ggml-base.en.bin -f "$VOICE" \
  --output-json-full --output-file /tmp/whisper_test --no-prints
```

Stage-success signature:

```text
2026-06-03 12:05:02,500 hermes.orchestrator INFO stage S12 (Video Assembly) done for EP_004
```

Important blocker example:

```text
2026-06-03 12:04:38,196 hermes.orchestrator WARNING stage S12 (Video Assembly) needs_human for EP_004: video/audio timeline mismatch before mux: video=627.8s, audio=618.3s, drift=9.5s
iteration 12/80: EP_004 stage=S12 blocked=true final=false
Episode EP_004 is blocked at S12; not auto-approving build-stage failures.
Review the blocker in state/episode_queue.json and the latest log before rerunning.
```

Agent action: stop. Inspect timeline and artifacts; do not run `--approve` automatically.

Artifacts:

```text
05_video/title_card.png
05_video/title_card_meta.json
05_video/clips/*.mp4
05_video/concat.txt
05_video/captions.srt
05_video/captions.vtt
05_video/final.mp4
06_metadata/license_attributions.txt
```

### 14. S13 Packaging

Stage-begin signature:

```text
2026-06-03 12:05:04,100 hermes.orchestrator INFO running S13 (Packaging) for episode EP_004 (topic=Vine, Inc.)
```

Representative stage logs:

```text
2026-06-03 12:05:24,666 hermes.stage.s13 INFO S13 titles: 10 variants -> titles.json
2026-06-03 12:05:24,805 hermes.thumbnails INFO thumbnails: generated 6 variants in /.../05_video/thumbnails
2026-06-03 12:05:24,805 hermes.stage.s13 INFO S13 thumbnails: 6 variants -> thumbnails
2026-06-03 12:05:38,565 hermes.shorts INFO shorts teaser TTS: 115 words, target 200 wpm, speed 1.00, enforce_wpm=True -> 34.5s
2026-06-03 12:06:10,960 hermes.stage.s13 INFO S13 short 1: short_01.mp4 (33.8s teaser)
2026-06-03 12:06:25,214 hermes.shorts INFO shorts teaser TTS: 108 words, target 200 wpm, speed 1.00, enforce_wpm=True -> 32.4s
2026-06-03 12:06:56,537 hermes.stage.s13 INFO S13 short 2: short_02.mp4 (31.7s teaser)
2026-06-03 12:06:56,873 hermes.stage.s13 INFO S13 complete: 10 titles, 2 shorts
```

Stage-success signature:

```text
2026-06-03 12:07:15,420 hermes.orchestrator INFO stage S13 (Packaging) done for EP_004
iteration 15/80: EP_004 stage=DONE blocked=false final=true
final video ready: /Users/cantemir/Projects/business_success_stories/episodes/EP_004_vine-inc/05_video/final.mp4
```

Warning example:

```text
2026-06-03 12:06:10,960 hermes.stage.s13 WARNING S13 short 1 FAILED for teaser build
```

Artifacts:

```text
06_metadata/titles.json
05_video/thumbnails/*.jpg
05_video/shorts/short_01.mp4
05_video/shorts/short_01.srt
05_video/shorts/short_01_title_card.png
05_video/shorts/teaser_script_01.txt
05_video/shorts/teaser_script_02.txt
05_video/shorts/manifest.json
05_video/shorts/teaser_script.txt
```

### 15. Build YouTube Package

Command:

```bash
python3 -m pipeline.hermes_orchestrator --build-youtube-package EP_004
```

Representative output:

```text
youtube package: /Users/cantemir/Projects/business_success_stories/episodes/EP_004_vine-inc/06_metadata/youtube_upload_package
manifest: /Users/cantemir/Projects/business_success_stories/episodes/EP_004_vine-inc/06_metadata/youtube_upload_package/package_manifest.json
contents: long_form=True, shorts=2, subtitle_tracks=33
```

Success signal: package path, manifest path, and nonzero expected content counts.

Artifacts:

```text
06_metadata/youtube_upload_package/package_manifest.json
06_metadata/youtube_upload_package/PACKAGE_SUMMARY.md
06_metadata/youtube_upload_package/long_form/
06_metadata/youtube_upload_package/shorts/
06_metadata/youtube_upload_package/thumbnails/
```

### 16. Upload YouTube Package

Command:

```bash
python3 -m pipeline.hermes_orchestrator \
  --upload-youtube-package EP_004 \
  --approve-youtube-upload \
  --youtube-privacy unlisted
```

Representative logs:

```text
2026-06-03 13:00:20,309 hermes.youtube_upload INFO YouTube upload complete: final.mp4 -> -_hTSZIDUv8
2026-06-03 13:00:22,101 hermes.youtube_upload INFO thumbnail uploaded: thumb_title_card_logo.jpg
2026-06-03 13:00:24,700 hermes.youtube_upload INFO caption uploaded: long_form_caption_en
2026-06-03 13:00:27,330 hermes.youtube_upload INFO caption uploaded: long_form_caption_es
2026-06-03 13:01:04,201 hermes.youtube_upload INFO YouTube upload complete: short_01.mp4 -> abcShort001
```

Representative output:

```text
uploaded long-form: https://www.youtube.com/watch?v=-_hTSZIDUv8
uploaded shorts: 2
- https://www.youtube.com/watch?v=abcShort001
- https://www.youtube.com/watch?v=abcShort002
```

Quota/scopes warning:

```text
googleapiclient.http WARNING Encountered 403 Forbidden with reason "quotaExceeded"
hermes.youtube_upload WARNING caption upload failed: short_01_caption_fr: quotaExceeded
```

If captions fail after the video upload succeeds, use caption backfill later.

### 17. Caption Backfill

Command:

```bash
python3 -m pipeline.hermes_orchestrator \
  --backfill-youtube-captions EP_004 \
  --approve-youtube-upload
```

Representative output:

```text
2026-06-03 19:30:25,208 hermes.youtube_upload INFO caption backfill skipped existing: long_form_caption_en
2026-06-03 19:30:25,496 hermes.youtube_upload INFO caption backfill skipped existing: short_01_caption_en
2026-06-03 19:30:26,100 hermes.youtube_upload INFO caption backfill uploaded: short_01_caption_fr
caption backfill package: /Users/cantemir/Projects/business_success_stories/episodes/EP_004_vine-inc/06_metadata/youtube_upload_package
caption backfill: attempted=1, uploaded=1, skipped_existing=11, warnings=0
```

Quota stop:

```text
2026-06-03 19:30:25,833 googleapiclient.http WARNING Encountered 403 Forbidden with reason "quotaExceeded"
2026-06-03 19:30:25,834 hermes.youtube_upload WARNING caption backfill stopping: quotaExceeded
caption backfill: attempted=1, uploaded=0, skipped_existing=11, warnings=1
```

Agent action: stop and retry after quota reset.

---

## Agent Operating Rules

1. Always check queue state before acting:

   ```bash
   python3 -m pipeline.hermes_orchestrator --status
   ```

2. A successful stage begins with:

   ```text
   hermes.orchestrator INFO running SNN (Stage Name) for episode EP_ID
   ```

3. A successful stage ends with:

   ```text
   hermes.orchestrator INFO stage SNN (Stage Name) done for EP_ID
   ```

4. A review/build hold is always:

   ```text
   hermes.orchestrator WARNING stage SNN (Stage Name) needs_human for EP_ID: REASON
   ```

5. Only auto-approve configured review gates: S07, optional S08, S09.

6. Do not auto-approve S10, S11, S12, or S13 blockers unless the operator explicitly says to accept the broken state.

7. Prefer targeted fixes:

   - S07 text safety: `--auto-correct-s07` or edit script, then `--rerun-from S7`.
   - S08 beat/prompt issue: edit `beat_sheet.json`, then `--rerun-from S9` or targeted `--rerender`.
   - S09 image issue: `--rerender EP_ID BEAT_ID --force-grok` or `--from-edited-prompt`.
   - S11 music/SFX issue: fix assets/manifests, then `--rerun-from S11`.
   - S12 video issue: fix timeline/assets, then `--rerun-from S12`.
   - S13 Shorts/package issue: fix packaging/shorts inputs, then `--rerun-from S13`.

8. If `run_orchestrator.sh` reports lock contention, wait. Do not delete the lock unless you have confirmed no Python orchestrator is running.

9. If YouTube upload partially succeeds, inspect:

   ```text
   06_metadata/youtube_upload_package/package_manifest.json
   06_metadata/youtube_upload_package/upload_results.json
   ```

   Then use caption backfill rather than re-uploading completed videos.

10. After code/config changes, regenerate only from the earliest affected stage.
