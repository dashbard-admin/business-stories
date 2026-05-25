"""Vision-Language Model adapter for image QA + captioning.

Sends rendered images to a VLM (Qwen3-VL via the local oMLX gateway).
Returns a structured verdict — score, anatomy flag, artifact list, and
a pass/borderline/reject decision. Used by S9 to trigger re-renders
when FLUX produces obvious artifacts.

Returns None on any infrastructure failure so a missing VLM never
blocks production — the caller treats None as "accept without judgment".

Both critique and caption prompts are tuned for comic-book panels
depicting business stories (NOT photo-realistic documentary content).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from .config import load_config

logger = logging.getLogger("hermes.vlm")

VLM_BASE = "http://10.0.4.250:9000/v1"
VLM_API_KEY = "pass123"

_VERDICTS = {"pass", "borderline", "reject"}


@dataclass
class ImageVerdict:
    score: int
    prompt_match: int
    anatomy_ok: bool
    artifacts: list[str]
    verdict: str          # one of: pass, borderline, reject
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_CRITIQUE_PROMPT = """\
You are reviewing an AI-generated COMIC BOOK PANEL intended for a
documentary about a real business or brand origin story. The image was
produced by a diffusion model from this text prompt:

PROMPT:
{prompt}

The image is NOT supposed to be photo-realistic. It IS supposed to look
like a polished comic-book panel: bold ink linework, flat or cel-shaded
colors, painterly textures, cinematic framing. Judge it as a comic
illustration, not a photograph.

Examine carefully for:
1. Subject match — does the panel depict what the prompt describes
   (the right person, scene, era, mood)?
2. Anatomy + composition — are visible people drawn with normal
   anatomy (no melted faces, fused fingers, extra limbs)? Are
   objects (phones, computers, office furniture, vehicles, signage)
   geometrically coherent?
3. Era and brand accuracy — does period-appropriate clothing,
   technology, and architecture appear correct for the era specified?
   Are any visible brand logos plausibly placed (or absent, since
   FLUX usually garbles real logos)?
4. Text legibility — diffusion models render text as illegible
   glyph-scribble. If the prompt asked for "no text" but text-like
   shapes appear on signage, headlines, screens, books — flag it.
5. AI artifacts — extra limbs, surreal distortions, glitchy
   textures, garbled text, watermarks, double exposure.

Output ONLY a JSON object, no prose, no code fences:
{{
  "score": <1-10 overall quality for publication>,
  "prompt_match": <1-10 subject accuracy>,
  "anatomy_ok": <true if all visible humans/objects are drawn
                 anatomically and structurally correct>,
  "artifacts": [<short list of specific issues spotted; empty if none>],
  "verdict": <"pass" | "borderline" | "reject">,
  "reasoning": <one short sentence on the verdict>
}}

Verdict rules:
- "pass": no notable artifacts, anatomy/objects correct, on-topic,
  publication-ready as a comic-book documentary frame.
- "borderline": minor flaws but acceptable for a comic frame.
- "reject": clear AI tells (deformed anatomy, surreal distortions,
  wrong era/subject, illegible required text, melting features) —
  must re-render.
"""


_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _clean(text: str) -> str:
    """Strip Qwen-style reasoning blocks + code fences."""
    text = _THINK_RE.sub("", text or "").strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _coerce(data: dict) -> ImageVerdict | None:
    try:
        verdict_str = str(data.get("verdict", "")).lower().strip()
        if verdict_str not in _VERDICTS:
            return None
        return ImageVerdict(
            score=int(data.get("score", 5)),
            prompt_match=int(data.get("prompt_match", 5)),
            anatomy_ok=bool(data.get("anatomy_ok", True)),
            artifacts=[str(a) for a in (data.get("artifacts") or [])][:10],
            verdict=verdict_str,
            reasoning=str(data.get("reasoning", ""))[:300],
        )
    except (ValueError, TypeError):
        return None


_ICONOGRAPHY_PROMPT = """\
You are briefing a comic-book illustrator about the iconic visual
identity of {person_name}. Look at the reference photograph and
describe what makes them visually distinct — the markers a reader
would recognise them by at a glance.

Focus on:
- Hair: shape, length, colour, hairline if notable
- Facial hair: kind, density, style
- Signature glasses, hats, or other facial accessories
- Signature clothing: a specific garment they are known for
  (black turtleneck, hoodie, suit-no-tie, bohemian linen, etc.)
- Signature posture or stance: how they typically hold themselves
- Signature props or environment: a specific phone, a podium,
  a stage backdrop, a recurring object

Do NOT describe individual facial proportions, exact age in years,
skin texture, or anything that would require photorealistic likeness.
The goal is COMIC-BOOK ICONOGRAPHY — the iconic markers, not the
face. A reader should be able to recognise this person from a
silhouette + the markers, without needing a photoreal portrait.

Write 60-100 words as a single paragraph the illustrator can paste
directly into a prompt. Do NOT use second person ("you see..."),
do NOT preamble, do NOT use the person's name in the paragraph.
Start with the most distinctive marker.

Output ONLY the paragraph. No JSON, no markdown, no preamble.
"""


_CAPTION_PROMPT = """\
You are captioning an archival image for a business-story documentary
pipeline. The caption will be used by a semantic-search system to
match this image to beats in a script, so it must be specific and
content-rich.

{context}\
Describe what is visible in 25-40 words. Be specific:
- Subject: name the person, building, product, or scene visible
  (e.g. "a founder posing at the entrance of a corporate headquarters",
  "a 1990s desktop computer and CRT monitor on a cluttered desk",
  "a 1970s assembly line in a brightly lit factory", "an empty
  boardroom with a long lacquered table", "a newspaper front-page
  headline announcing a corporate bankruptcy")
- Composition: close-up / wide / aerial / interior / portrait / etc.
- Era markers visible: clothing, technology, architecture, lighting
- Mood where salient (celebratory, tense, abandoned, busy)

Hard constraints:
- DO NOT speculate about events, financial outcomes, or editorialize.
- DO NOT use phrases like "appears to be" or "could be".
- DO NOT mention what is NOT in the image.

Output ONLY the caption sentence(s). No JSON, no markdown, no preamble.
"""


class VLM:
    """Adapter for the local Vision-Language model."""

    def __init__(self):
        cfg = load_config()
        self.model_name: str = cfg.models.get(
            "vlm", "Qwen3-VL-4B-Instruct-MLX-8bit"
        )
        self.timeout: int = int(cfg.image_qa.get("timeout_seconds", 30))

    # ------------------------------------------------------------------

    def caption_image(self, image_path: Path, incident_name: str = "") -> str | None:
        """Generate a content-rich caption for an archival image.

        Used by S5 to enrich PD asset descriptions so downstream
        semantic matching (S8) has more signal than the source filename.
        Returns None on any failure path.
        """
        try:
            if not image_path.exists() or image_path.stat().st_size < 1000:
                return None
            b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as e:
            logger.warning("VLM cannot read %s: %s", image_path, e)
            return None

        context = (f'Context: this image is being assembled for a '
                   f'documentary about "{incident_name}".\n\n') if incident_name else ""

        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": _CAPTION_PROMPT.format(context=context)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "temperature": 0.3,
            "max_tokens": 150,
        }

        try:
            r = requests.post(
                f"{VLM_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {VLM_API_KEY}"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("VLM caption failed for %s: %s", image_path.name, e)
            return None

        caption = _clean(text).strip()
        if len(caption) < 15 or len(caption) > 500:
            return None
        caption = caption.strip('"\'').strip()
        return caption or None

    def describe_iconography(self, image_path: Path, person_name: str) -> str | None:
        """Brief a comic-book illustrator on a person's iconic visual
        markers (hair, signature glasses/clothing/props, stance).

        Used by S05's character-profile sub-step. Returns a 60-100
        word paragraph the illustrator can paste directly into a
        FLUX prompt, or None on any infrastructure failure.

        Critically, the prompt explicitly tells the VLM to AVOID
        face-proportion specifics, because base FLUX cannot reliably
        reproduce facial likeness from text. We aim for COMIC-BOOK
        ICONOGRAPHY (signature markers) rather than portraiture.
        """
        if not person_name:
            return None
        try:
            if not image_path.exists() or image_path.stat().st_size < 1000:
                return None
            b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as e:
            logger.warning("VLM cannot read %s: %s", image_path, e)
            return None

        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": _ICONOGRAPHY_PROMPT.format(person_name=person_name)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "temperature": 0.35,
            "max_tokens": 250,
        }

        try:
            r = requests.post(
                f"{VLM_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {VLM_API_KEY}"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("VLM iconography failed for %s: %s",
                           image_path.name, e)
            return None

        cleaned = _clean(text).strip().strip('"\'').strip()
        # Reject pathological outputs (single word, refusal, etc.).
        if len(cleaned) < 40 or len(cleaned) > 1500:
            return None
        return cleaned

    def critique_image(self, image_path: Path, prompt: str) -> ImageVerdict | None:
        """Critique `image_path` against `prompt` for comic-panel artifacts.

        Returns the structured verdict on success, None on any failure
        path (so VLM outages are non-blocking).
        """
        try:
            if not image_path.exists() or image_path.stat().st_size < 1000:
                return None
            b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as e:
            logger.warning("VLM cannot read %s: %s", image_path, e)
            return None

        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": _CRITIQUE_PROMPT.format(prompt=prompt)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "temperature": 0.2,
            "max_tokens": 800,
        }

        try:
            r = requests.post(
                f"{VLM_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {VLM_API_KEY}"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("VLM request failed for %s: %s", image_path.name, e)
            return None

        text = _clean(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                logger.warning("VLM returned no JSON for %s: %r",
                               image_path.name, text[:200])
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                logger.warning("VLM JSON parse failed for %s: %s",
                               image_path.name, e)
                return None

        verdict = _coerce(data)
        if verdict is None:
            logger.warning("VLM verdict coercion failed for %s: %s",
                           image_path.name, str(data)[:200])
        return verdict
