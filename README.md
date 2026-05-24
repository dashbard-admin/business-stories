# Business & Brand Origin Stories — 12-Stage Pipeline

A faceless YouTube long-form pipeline that produces 10–15 minute comic-book-styled videos about business origin / rise-and-fall / disruption / underdog stories. Built on the same one-stage-per-cron-invocation pattern as the maritime pipeline at `/Users/cantemir/Projects/maritime/`, but with a different domain, visual style, and audio model:

**Stack**: Qwen3.6 (writer + extractor) + Gemma-4 (critic) + Kokoro (TTS) + FLUX via local CLI (images) + ffmpeg (assembly) + curated local music library (no audio generation).

## 12 stages

| ID  | Module                                  | Purpose                                                                                    |
|-----|------------------------------------------|--------------------------------------------------------------------------------------------|
| S01 | `pipeline/stages/s01_topic_discovery`    | Writer LLM picks the next business story (company, founder, hero, conflict).               |
| S02 | `pipeline/stages/s02_source_gathering`   | Paywall-aware SearXNG recipes + Wayback fallback for gated outlets.                        |
| S03 | `pipeline/stages/s03_fact_extraction`    | Per-source extraction into a business-domain fact_type enum; HQ consolidation.             |
| S04 | `pipeline/stages/s04_fact_verification`  | Critic merges facts into claims; writer-as-skeptic verifies adversarially.                 |
| S05 | `pipeline/stages/s05_asset_hunt`         | PD asset hunt (Wikimedia, Smithsonian, LoC, archive.org, etc). No map renderer.            |
| S06 | `pipeline/stages/s06_script_generation`  | 2000-word hero/conflict business narrative, multi-pass length adjustment.                  |
| S07 | `pipeline/stages/s07_script_critique`    | Retention + voice audit; fuzzy-replace rewrites.                                           |
| S08 | `pipeline/stages/s08_beat_sheet`         | Per-beat visual + sfx hints; PD-vs-FLUX semantic routing.                                  |
| S09 | `pipeline/stages/s09_flux_render`        | FLUX rendering via local CLI subprocess + VLM-judged QA retry loop.                        |
| S10 | `pipeline/stages/s10_kokoro_render`      | Kokoro TTS with pronunciation overrides; per-beat timing.                                  |
| S11 | `pipeline/stages/s11_audio_post`         | Music bed from local library, sidechain duck, loudnorm. **No SFX, no MusicGen.**           |
| S12 | `pipeline/stages/s12_video_assembly`     | Per-beat Ken Burns clips, FLUX title + credits cards, concat + mux, SRT/VTT.               |

## Layout

```
business_success_stories/
├── pipeline/                       # all code
│   ├── hermes_orchestrator.py      # cron entry point
│   ├── config.py / state.py        # configuration + state plumbing
│   ├── llm.py / tts.py / vlm.py    # LLM / Kokoro / VLM adapters
│   ├── flux.py                     # FLUX CLI subprocess adapter
│   ├── browser.py                  # SearXNG + paywall-aware fetch
│   ├── music_library.py            # local music-bed matcher
│   ├── ffmpeg_builder.py           # ffmpeg wrapper
│   ├── generic_stash.py            # operator-curated PD stash
│   ├── constraints.py              # rolling-window anti-template engine
│   ├── stages/                     # 12 stage modules (s01..s12)
│   ├── prompts/                    # all LLM prompts (operator-editable)
│   ├── style_profiles/             # V1.yaml + V2.yaml + archetypes + narrators
│   ├── lint/forbidden_phrases.txt
│   ├── lexicon/pronunciation_overrides.yaml
│   └── sources/                    # SearXNG recipe modules
├── assets/
│   ├── generic_stock/              # operator-curated PD stash + manifest
│   └── music_library/              # documentary music + manifest
├── state/                          # episode_queue.json, locks, used_topics
├── episodes/                       # per-episode workspaces (created by S01)
├── logs/
├── config.yaml                     # operator-facing config
├── pyproject.toml
├── run_orchestrator.sh
└── README.md
```

## Running

```bash
# One-shot (cron target)
./run_orchestrator.sh

# Or directly
python -m pipeline.hermes_orchestrator
```

Seed the queue:

```bash
python -m pipeline.hermes_orchestrator --enqueue 5
```

Inspect:

```bash
python -m pipeline.hermes_orchestrator --status
```

Cron suggestion:

```
0 */3 * * * /Users/cantemir/Projects/business_success_stories/run_orchestrator.sh
```

## External requirements

- **FLUX CLI** on `$PATH`, invokable as `flux "<prompt>" --height 1080 --width 1920 --steps 24 --seed N --output <path.png>`.
- **Kokoro TTS server** running on `127.0.0.1:8001` (the maritime stack's existing server is fine).
- **LLM gateway** at `10.0.4.250:9000` (oMLX OpenAI-compatible) with Qwen3.6, Gemma-4, and Qwen3-VL loaded.
- **SearXNG** at `10.0.4.252:8080` with JSON output enabled.
- **ffmpeg + ffprobe** on `$PATH`.
- **Music library** populated at `assets/music_library/` with `manifest.json` describing each track (mood, tempo, instruments, duration).

## Mock-mode smoke test

Set `models.mock_mode: true` in `config.yaml`, then:

```bash
python -m pipeline.hermes_orchestrator --enqueue 1
for _ in {1..12}; do python -m pipeline.hermes_orchestrator -v; done
python -m pipeline.hermes_orchestrator --status
```

Each invocation should advance the episode by one stage; the final episode workspace at `episodes/EP_001*/` will contain mock blank PNGs, a silent voice track, and a stub final.mp4.
