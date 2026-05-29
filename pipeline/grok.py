"""xAI Grok image-regeneration adapter.

S09 uses this to repair FLUX renders whose VLM verdict flagged
malformed / illegible text. We POST the original FLUX prompt
(wrapped in the operator-tunable grok_text_correction.txt template)
to xAI's image-generation endpoint and receive a corrected image
back, then write it to disk.

The endpoint shape defaults to OpenAI-compatible JSON
(POST /v1/images/generations with `model`, `prompt`, `resolution`,
`aspect_ratio`, `n`). Both base64-JSON and URL response shapes are
handled. If xAI ships a slightly different shape later, the
endpoint_path can be overridden in config.yaml without code changes.

Auth: bearer token from config.yaml > grok.api_key. While the key
is empty or matches the placeholder, the adapter reports
unavailable and callers skip the correction pass.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import requests

from .config import load_config

logger = logging.getLogger("hermes.grok")

PLACEHOLDER_KEY_TOKENS = ("replace_with", "your_key", "todo", "xxx")

# Environment variables checked, in priority order, for the xAI API
# key. XAI_API_KEY is the canonical name (matches xAI's own docs);
# GROK_API_KEY is accepted as an alias for ergonomics.
API_KEY_ENV_VARS = ("XAI_API_KEY", "GROK_API_KEY")


class Grok:
    """xAI Grok image-regeneration client.

    Public surface:
        .available — bool, True iff config is good enough to call.
        .regenerate_from_prompt(prompt, out_path) -> Path | None
    """

    def __init__(self):
        cfg = load_config()
        gc = cfg.grok
        self.enabled: bool = bool(gc.get("enabled", False))
        # API key resolution order:
        #   1. XAI_API_KEY in os.environ
        #   2. GROK_API_KEY in os.environ
        #   3. config.yaml > grok.api_key (back-compat only; not
        #      recommended — GitHub secret-scanning rejects pushes
        #      containing real keys. Operator workflow: keep keys in
        #      .env at the project root; .env is gitignored.)
        # pipeline/__init__.py loads .env into os.environ at import
        # time, so adapters never have to touch the file directly.
        self.api_key: str = _resolve_api_key(gc.get("api_key", ""))
        self.model: str = gc.get("model", "grok-imagine-image")
        self.base_url: str = (gc.get("base_url") or "").rstrip("/")
        self.endpoint_path: str = gc.get("endpoint_path", "/images/generations")
        self.timeout: int = int(gc.get("timeout_seconds", 180))
        self.resolution: str = gc.get("resolution", "2k")
        self.aspect_ratio: str = gc.get("aspect_ratio", "16:9")
        self._mock = cfg.mock_mode

    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True iff a real API call could plausibly succeed (or we are
        in mock_mode and will fabricate a corrected image)."""
        if self._mock:
            return True
        if not self.enabled:
            return False
        if not self.api_key:
            return False
        low = self.api_key.lower()
        if any(tok in low for tok in PLACEHOLDER_KEY_TOKENS):
            return False
        if not self.base_url:
            return False
        return True

    def unavailability_reason(self) -> str:
        """Single short reason this adapter cannot run, for logging."""
        if not self.enabled:
            return "grok.enabled=false"
        if not self.api_key:
            return "grok.api_key not set"
        if any(tok in self.api_key.lower() for tok in PLACEHOLDER_KEY_TOKENS):
            return "grok.api_key still holds the placeholder"
        if not self.base_url:
            return "grok.base_url empty"
        return "unknown"

    # ------------------------------------------------------------------

    def regenerate_from_prompt(
        self,
        prompt: str,
        out_path: Path,
    ) -> Path | None:
        """Ask Grok to generate an image from `prompt` and write the
        result to `out_path`. Returns `out_path` on success, `None`
        on any failure.

        We send only the prompt — no reference image, no editing
        directives. Grok produces a fresh image based on the same
        FLUX-generated prompt that the orchestrator originally
        rendered. The expectation is that Grok renders text more
        reliably than FLUX, so beats with malformed text in the
        FLUX output get a clean regeneration.

        Request shape per xAI docs:
            POST /v1/images/generations
            {
              "model": "<grok-imagine-image | -quality>",
              "prompt": "...",
              "resolution": "2k",
              "aspect_ratio": "16:9",
              "n": 1
            }

        Response:
            { "data": [ { "url": "https://..." }, ... ] }

        _write_response_image handles both the URL response (xAI's
        actual return shape) and an OpenAI-style base64 response
        (in case xAI ever adds it as an option).
        """
        if self._mock:
            return self._mock_correct(out_path)
        if not self.available:
            logger.warning("grok unavailable (%s); skipping regeneration",
                           self.unavailability_reason())
            return None

        url = f"{self.base_url}{self.endpoint_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "prompt": prompt,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "n": 1,
        }

        try:
            r = requests.post(url, headers=headers, json=payload,
                              timeout=self.timeout)
        except requests.RequestException as e:
            logger.warning("grok HTTP error -> %s: %s", out_path.name, e)
            return None

        if r.status_code == 200:
            if self._write_response_image(r, out_path):
                logger.info("grok regenerated -> %s "
                            "(model=%s, resolution=%s, aspect=%s)",
                            out_path.name, self.model,
                            self.resolution, self.aspect_ratio)
                return out_path
            logger.warning("grok 200 but response body unrecognized "
                           "for %s (first 300 chars: %s)",
                           out_path.name, (r.text or "")[:300])
            return None

        logger.warning("grok %d for %s; body: %s",
                       r.status_code, out_path.name,
                       (r.text or "")[:500])
        return None

    # ------------------------------------------------------------------

    def _write_response_image(self, r: requests.Response, out: Path) -> bool:
        """Extract an image from an OpenAI-compatible image-generation
        response. Tries base64 (`data[0].b64_json`) first, falls back
        to a URL (`data[0].url`). Returns True on success."""
        try:
            body = r.json()
        except ValueError:
            return False
        items = (body or {}).get("data") or []
        if not items:
            return False
        item = items[0]
        b64 = item.get("b64_json")
        if b64:
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(base64.b64decode(b64))
                return out.exists() and out.stat().st_size > 1000
            except Exception as e:
                logger.warning("grok base64 decode failed: %s", e)
                return False
        url = item.get("url")
        if url:
            try:
                img_r = requests.get(url, timeout=60)
                img_r.raise_for_status()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(img_r.content)
                return out.exists() and out.stat().st_size > 1000
            except Exception as e:
                logger.warning("grok image download failed: %s", e)
                return False
        return False

    def _mock_correct(self, out: Path) -> Path | None:
        """Mock-mode regeneration — render a solid colour panel with a
        green border so a human inspecting the output can tell which
        file went through the (mock) Grok pass.
        """
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (1920, 1080), (30, 60, 120))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, img.width - 1, img.height - 1],
                        outline=(40, 200, 80), width=6)
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, "PNG")
            return out
        except Exception as e:
            logger.warning("grok mock regeneration failed: %s", e)
            return None


def _resolve_api_key(config_value: str) -> str:
    """Pick the xAI API key from the highest-priority source available.

    Priority:
      1. XAI_API_KEY env var (xAI's canonical name)
      2. GROK_API_KEY env var (alias for ergonomics)
      3. config.yaml > grok.api_key (back-compat only — discouraged
         because committing a real key to config.yaml will trip
         GitHub's secret scanner and reject the push)

    .env files at the project root are loaded into os.environ at
    package import time by pipeline/__init__.py, so any of these
    paths work transparently.
    """
    for var in API_KEY_ENV_VARS:
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    return (config_value or "").strip()
