"""FLUX image generation adapter — CLI subprocess.

Unlike the maritime pipeline (HTTP server), this project drives the
FLUX renderer via a `flux` CLI on PATH:

    flux "<prompt>" --height 1080 --width 1920 --steps 24 \\
                    --seed <N> --output <abs_path.png>

The CLI has no img2img / reference-image flag, so the public surface
keeps the `reference_image_path` field on FluxRequest for compatibility
with maritime-shaped beat sheets, but the adapter SILENTLY IGNORES it
when invoking the CLI. S8 should set `pd_image_reference_enabled:
false` in config.yaml so the caller folds the reference caption into
the text prompt instead.

S9's contract is preserved:
  - class Flux()
  - class FluxRequest(...)
  - method render_batch_with_retry(req, num_candidates, seed_offset)
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import load_config

logger = logging.getLogger("hermes.flux")


def compute_seed(prompt: str, *, seed_offset: int = 0) -> int:
    """Derive a deterministic 31-bit seed from a prompt."""
    return (
        int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) + seed_offset
    ) & 0x7FFFFFFF


@dataclass
class FluxRequest:
    beat_id: str
    prompt: str
    negative_prompt: str = ""
    out_path: Path | None = None
    # img2img reference — ignored by the CLI adapter. Kept for shape
    # compatibility with the maritime FluxRequest and so S8 doesn't
    # need to branch on adapter type.
    reference_image_path: str | Path | None = None
    reference_strength: float = 0.5
    # Per-request overrides; default to config values.
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class FluxImage:
    filepath: Path
    prompt: str
    seed: int


class Flux:
    """CLI-driven adapter around the global `flux` binary."""

    def __init__(self):
        cfg = load_config()
        self._mock: bool = cfg.mock_mode
        cli = cfg.flux_cli
        self.binary: str = cli.get("binary", "flux")
        self.timeout: int = int(cli.get("timeout_seconds", 600))
        self.default_steps: int = int(cli.get("steps", 24))
        self.default_width: int = int(cli.get("width", 1920))
        self.default_height: int = int(cli.get("height", 1080))
        self.fold_negative: bool = bool(cli.get("fold_negative_into_prompt", True))

    # ------------------ public API ------------------

    def render_batch_with_retry(
        self,
        req: FluxRequest,
        *,
        num_candidates: int = 4,
        seed_offset: int = 0,
    ) -> Path | None:
        """Render `num_candidates` images, keep the largest by file size,
        delete the rest. `seed_offset` is for higher-level retries (S9
        VLM reject → new seed)."""
        if not req.out_path:
            raise ValueError("FluxRequest.out_path is required")

        base = Path(req.out_path)
        base.parent.mkdir(parents=True, exist_ok=True)

        seed_base = compute_seed(req.prompt, seed_offset=seed_offset)
        candidates: list[Path] = []

        for i in range(num_candidates):
            seed = (seed_base + i) & 0x7FFFFFFF
            target = (
                base
                if num_candidates == 1
                else base.with_name(f"{base.stem}_c{i}{base.suffix}")
            )
            produced = self._render_one(req, target, seed)
            if produced and produced.exists() and produced.stat().st_size > 1000:
                candidates.append(produced)
            else:
                logger.warning("FLUX candidate %d failed for %s", i, req.beat_id)

        if not candidates:
            return None

        chosen = max(candidates, key=lambda p: p.stat().st_size)
        if chosen != base:
            shutil.copy(str(chosen), str(base))
        for c in candidates:
            if c != base:
                try:
                    c.unlink(missing_ok=True)
                except Exception:
                    pass
        return base

    # ------------------ internals ------------------

    def _render_one(self, req: FluxRequest, out_path: Path, seed: int) -> Path | None:
        if self._mock:
            return self._mock_render(out_path)

        prompt = req.prompt
        if self.fold_negative and req.negative_prompt:
            # The CLI has no --negative flag. Fold the negative-prompt
            # content into the positive prompt as an "avoid:" tail. FLUX
            # respects this convention surprisingly well at high CFG.
            prompt = f"{prompt} -- avoid: {req.negative_prompt}"

        if req.reference_image_path:
            logger.info(
                "flux CLI has no img2img; ignoring reference_image_path "
                "for %s (set image_qa.pd_image_reference_enabled=false to silence)",
                req.beat_id,
            )

        width = req.width or self.default_width
        height = req.height or self.default_height
        steps = req.steps or self.default_steps

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.binary,
            prompt,
            "--height", str(height),
            "--width", str(width),
            "--steps", str(steps),
            "--seed", str(int(seed)),
            "--output", str(out_path.resolve()),
        ]
        if req.extra_args:
            cmd.extend(req.extra_args)

        logger.info("flux (%s seed=%d) %dx%d steps=%d → %s",
                    req.beat_id, seed, width, height, steps, out_path.name)
        logger.debug("flux cmd: %s", " ".join(repr(c) for c in cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("flux timed out after %ds for %s seed=%d",
                           self.timeout, req.beat_id, seed)
            return None
        except FileNotFoundError:
            logger.error("flux binary not found on PATH (binary=%r). "
                         "Set flux_cli.binary in config.yaml or enable mock_mode.",
                         self.binary)
            return None

        if proc.returncode != 0:
            logger.warning("flux exit=%d for %s seed=%d: %s",
                           proc.returncode, req.beat_id, seed,
                           (proc.stderr or "")[-400:])
            return None

        if not out_path.exists() or out_path.stat().st_size < 1000:
            logger.warning("flux produced no/empty file for %s seed=%d",
                           req.beat_id, seed)
            return None

        return out_path

    def _mock_render(self, out_path: Path) -> Path:
        from PIL import Image
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (self.default_width, self.default_height),
                  color=(60, 35, 80)).save(out_path)
        return out_path
