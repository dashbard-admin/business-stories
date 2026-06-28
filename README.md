# Business & Brand Origin Stories — 13-Stage Pipeline

A faceless YouTube long-form pipeline that produces 9–11 minute comic-book-styled videos about business origin / rise-and-fall / disruption / underdog stories. Built on the same one-stage-per-cron-invocation pattern as the maritime pipeline at `/Users/cantemir/Projects/maritime/`, but with a different domain, visual style, and audio model:

**Stack**: Qwen3.6 (writer + extractor) + Gemma-4 (critic) + Chatterbox TTS by default (Kokoro fallback, ElevenLabs optional) + FLUX via local CLI and/or Grok via xAI (images) + ffmpeg (assembly) + curated local music library.

> 📖 **For the full engineering reference — module-by-module description, prompt catalog, architecture diagram, state schemas, and the maintenance contract — see [`AGENTS.md`](./AGENTS.md).** That file is the canonical source of truth for how this project is built; every code change must update it in the same commit.

## 13 stages

| ID  | Module                                  | Purpose                                                                                    |
|-----|------------------------------------------|--------------------------------------------------------------------------------------------|
| S01 | `pipeline/stages/s01_topic_discovery`    | Writer LLM picks the next business story (company, founder, hero, conflict).               |
| S02 | `pipeline/stages/s02_source_gathering`   | Paywall-aware SearXNG recipes + Wayback fallback for gated outlets.                        |
| S03 | `pipeline/stages/s03_fact_extraction`    | Per-source extraction into a business-domain fact_type enum; HQ consolidation.             |
| S04 | `pipeline/stages/s04_fact_verification`  | Critic merges facts into claims; writer-as-skeptic verifies adversarially.                 |
| S05 | `pipeline/stages/s05_asset_hunt`         | PD asset hunt (Wikimedia, Smithsonian, LoC, archive.org, etc). No map renderer.            |
| S06 | `pipeline/stages/s06_script_generation`  | ~1550-word hero/conflict business narrative, multi-pass length adjustment.                 |
| S07 | `pipeline/stages/s07_script_critique`    | Retention + voice audit; fuzzy-replace rewrites.                                           |
| S08 | `pipeline/stages/s08_beat_sheet`         | Per-beat visual + sfx hints; PD-vs-FLUX semantic routing.                                  |
| S09 | `pipeline/stages/s09_flux_render`        | Image rendering via FLUX, Grok, or both; VLM-judged QA and Grok moderation retry.          |
| S10 | `pipeline/stages/s10_kokoro_render`      | TTS render with pronunciation overrides; Chatterbox default, Kokoro/ElevenLabs optional; per-beat timing. |
| S11 | `pipeline/stages/s11_audio_post`         | Music bed from local library, sidechain duck, loudnorm. **No SFX, no MusicGen.**           |
| S12 | `pipeline/stages/s12_video_assembly`     | Per-beat Ken Burns clips, generated title/credits cards, concat + mux, SRT/VTT.            |
| S13 | `pipeline/stages/s13_packaging`          | Titles, thumbnails, Shorts, and YouTube package inputs.                                    |

## Layout

```
business_success_stories/
├── pipeline/                       # all code
│   ├── hermes_orchestrator.py      # cron entry point
│   ├── config.py / state.py        # configuration + state plumbing
│   ├── llm.py / tts.py / vlm.py    # LLM / TTS dispatcher / VLM adapters
│   ├── flux.py                     # FLUX CLI subprocess adapter
│   ├── browser.py                  # SearXNG + paywall-aware fetch
│   ├── music_library.py            # local music-bed matcher
│   ├── ffmpeg_builder.py           # ffmpeg wrapper
│   ├── generic_stash.py            # operator-curated PD stash
│   ├── constraints.py              # rolling-window anti-template engine
│   ├── stages/                     # stage modules (s01..s13)
│   ├── prompts/                    # all LLM prompts (operator-editable)
│   ├── style_profiles/             # visual styles + archetypes + narrators
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
- **xAI Grok API key** in `.env` as `XAI_API_KEY` when `image_generation.backend` is `grok` or `both`.
- **TTS backend**: Chatterbox via the oMLX gateway at `10.0.4.250:9000` by default. Kokoro remains available with `tts.backend: kokoro`.
- **LLM gateway** at `10.0.4.250:9000` (oMLX OpenAI-compatible) with Qwen3.6, Gemma-4, and Qwen3-VL loaded.
- **SearXNG** at `10.0.4.252:8080` with JSON output enabled.
- **ffmpeg + ffprobe** on `$PATH`.
- **Music library** populated at `assets/music_library/` with `manifest.json` describing each track (mood, tempo, instruments, duration).

## Mock-mode smoke test

Set `models.mock_mode: true` in `config.yaml`, then:

```bash
python -m pipeline.hermes_orchestrator --enqueue 1
for _ in {1..13}; do python -m pipeline.hermes_orchestrator -v; done
python -m pipeline.hermes_orchestrator --status
```

Each invocation should advance the episode by one stage; the final episode workspace at `episodes/EP_001*/` will contain mock blank PNGs, a silent voice track, and a stub final.mp4.
