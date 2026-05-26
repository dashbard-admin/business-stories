"""xAI Grok image-edit adapter.

S09 uses this to repair FLUX renders whose VLM verdict flagged
malformed / illegible text. We POST the offending image plus the
original FLUX prompt (wrapped in the operator-tunable
grok_text_correction.txt template) to xAI's image-edit endpoint
and receive a corrected image back, then write it to disk.

The endpoint shape defaults to OpenAI-compatible
multipart/form-data (POST /v1/images/edits with `image`, `prompt`,
`model`, `n`, `size`). Both base64-JSON and URL response shapes
are handled. If xAI ships a slightly different shape later, the
endpoint_path can be overridden in config.yaml without code
changes.

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
    """xAI Grok image-edit client.

    Public surface:
        .available — bool, True iff config is good enough to call.
        .correct_image(image_path, prompt, out_path) -> Path | None
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
        self.endpoint_path: str = gc.get("endpoint_path", "/images/edits")
        self.timeout: int = int(gc.get("timeout_seconds", 180))
        self.size_primary: str = gc.get("size_primary", "1920x1080")
        self.size_fallback: str = gc.get("size_fallback", "2048x1152")
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

    def correct_image(
        self,
        image_path: Path,
        prompt: str,
        out_path: Path,
    ) -> Path | None:
        """Send `image_path` + `prompt` to Grok, write the corrected
        image to `out_path`. Returns out_path on success, None on
        any failure.

        Tries the primary size first; on 4xx that mentions size,
        retries once with the fallback size.
        """
        if self._mock:
            return self._mock_correct(image_path, out_path)
        if not self.available:
            logger.warning("grok unavailable (%s); skipping correction",
                           self.unavailability_reason())
            return None

        url = f"{self.base_url}{self.endpoint_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Encode the image once (base64) — the request body is JSON,
        # not multipart, per xAI's 415 reply: "Expected request with
        # Content-Type: application/json".
        try:
            with image_path.open("rb") as fh:
                image_b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError as e:
            logger.warning("grok: cannot read %s: %s", image_path, e)
            return None
        image_data_uri = f"data:image/png;base64,{image_b64}"

        for size in (self.size_primary, self.size_fallback):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "image": image_data_uri,
                "n": 1,
                "size": size,
                "response_format": "b64_json",
            }
            try:
                r = requests.post(url, headers=headers, json=payload,
                                  timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning("grok HTTP error (size=%s) for %s: %s",
                               size, image_path.name, e)
                return None

            if r.status_code == 200:
                if self._write_response_image(r, out_path):
                    logger.info("grok corrected %s -> %s (size=%s)",
                                image_path.name, out_path.name, size)
                    return out_path
                logger.warning("grok 200 but response body unrecognized "
                               "for %s (first 200 chars: %s)",
                               image_path.name, (r.text or "")[:200])
                return None

            # On 4xx, peek at the body — if it complains about size,
            # the fallback size loop continues. If it complains about
            # the `image` field shape, retry once with the raw-base64
            # form instead of the data-URI form. Otherwise bail.
            if 400 <= r.status_code < 500:
                body_low = (r.text or "").lower()
                if "size" in body_low or "resolution" in body_low or "dimension" in body_low:
                    logger.warning("grok %d (size=%s) — trying fallback. "
                                   "body: %s", r.status_code, size,
                                   (r.text or "")[:200])
                    continue
                # Some endpoints reject the data: URI prefix and want
                # the raw base64 only. Retry once with the bare b64.
                if ("image" in body_low and (
                        "format" in body_low or "invalid" in body_low
                        or "data" in body_low or "base64" in body_low)
                        and payload["image"] == image_data_uri):
                    logger.warning("grok %d — body complains about "
                                   "image field; retrying with raw "
                                   "base64 (no data: prefix). body: %s",
                                   r.status_code, (r.text or "")[:200])
                    payload["image"] = image_b64
                    try:
                        r = requests.post(url, headers=headers,
                                          json=payload,
                                          timeout=self.timeout)
                    except requests.RequestException as e:
                        logger.warning("grok HTTP error on bare-b64 "
                                       "retry: %s", e)
                        return None
                    if r.status_code == 200:
                        if self._write_response_image(r, out_path):
                            logger.info("grok corrected %s -> %s "
                                        "(bare-b64, size=%s)",
                                        image_path.name, out_path.name,
                                        size)
                            return out_path
                    # If the bare-b64 retry also fails, fall through
                    # to the generic 4xx warning below.
                logger.warning("grok %d for %s; body: %s",
                               r.status_code, image_path.name,
                               (r.text or "")[:300])
                return None

            # 5xx or weird code — bail.
            logger.warning("grok %d for %s; body: %s",
                           r.status_code, image_path.name,
                           (r.text or "")[:300])
            return None

        return None

    # ------------------------------------------------------------------

    def _write_response_image(self, r: requests.Response, out: Path) -> bool:
        """Extract an image from an OpenAI-compatible /images/edits
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

    def _mock_correct(self, src: Path, out: Path) -> Path | None:
        """Mock-mode 'correction' — copy the source pixels into out
        and tint the borders slightly so a human inspecting the
        output can tell which file went through the (mock) Grok pass.
        """
        try:
            from PIL import Image, ImageDraw
            with Image.open(src) as bg:
                img = bg.convert("RGB").copy()
            d = ImageDraw.Draw(img)
            # 6-px green border to flag mock-Grok output visually
            d.rectangle([0, 0, img.width - 1, img.height - 1],
                        outline=(40, 200, 80), width=6)
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, "PNG")
            return out
        except Exception as e:
            logger.warning("grok mock correction failed: %s", e)
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
