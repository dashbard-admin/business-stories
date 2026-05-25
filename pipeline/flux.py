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
import time
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

        # Run the CLI from a CLEAN temp directory as CWD. Some FLUX
        # CLI versions have been observed to (a) write to CWD instead
        # of honoring an absolute --output path, or (b) write to a
        # default basename in CWD when their post-generation move
        # silently fails — producing the exact "Generation completed
        # but output file not found" symptom. By isolating CWD we can
        # scan the tempdir afterwards to recover whatever PNG the CLI
        # actually produced and promote it to the requested path.
        import tempfile
        scan_start = time.time()
        with tempfile.TemporaryDirectory(prefix="flux_") as tmpdir:
            tmpdir_path = Path(tmpdir)

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
            logger.debug("flux cmd: %s (cwd=%s)",
                         " ".join(repr(c) for c in cmd), tmpdir_path)

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(tmpdir_path),
                    # Explicit /dev/null stdin. Without this the child
                    # inherits the orchestrator's fd 0, which in cron /
                    # daemon contexts (and sometimes after many prior
                    # subprocess calls accumulate state) can be closed
                    # or invalid. The FLUX CLI's Python interpreter
                    # then crashes during startup with
                    #   "Fatal Python error: init_sys_streams:
                    #    can't initialize sys standard streams"
                    #   "OSError: [Errno 9] Bad file descriptor"
                    # in <20 ms — before any FLUX code runs. Passing
                    # DEVNULL gives the child a guaranteed-valid
                    # read-only empty stdin.
                    stdin=subprocess.DEVNULL,
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

            # Path 1: the CLI honored --output. Cheapest case.
            if out_path.exists() and out_path.stat().st_size >= 1000:
                return out_path

            # Path 2: the CLI wrote SOMEWHERE in our clean tempdir
            # (default basename, weird relative resolution, etc.).
            # Recover the largest PNG and promote it.
            recovered = _find_recovered_png(tmpdir_path, scan_start)
            if recovered is not None:
                logger.warning(
                    "flux %s seed=%d: CLI wrote %s in cwd "
                    "(not at --output); promoting to %s",
                    req.beat_id, seed, recovered.name, out_path,
                )
                shutil.copy(str(recovered), str(out_path))
                return out_path

            # Path 3: the CLI wrote near the requested --output path
            # (e.g. ignored our path and used a default filename in
            # the parent dir).
            recovered_parent = _find_recovered_png(out_path.parent, scan_start)
            if (recovered_parent is not None
                    and recovered_parent.resolve() != out_path.resolve()):
                logger.warning(
                    "flux %s seed=%d: CLI wrote %s in target dir "
                    "(not at --output); promoting to %s",
                    req.beat_id, seed, recovered_parent.name, out_path,
                )
                shutil.copy(str(recovered_parent), str(out_path))
                return out_path

        # Nothing recoverable. Surface the full stderr so the operator
        # can diagnose the CLI behaviour.
        if proc.returncode != 0:
            logger.warning(
                "flux exit=%d for %s seed=%d; no output recovered. "
                "stderr (last 2000 chars):\n%s",
                proc.returncode, req.beat_id, seed,
                (proc.stderr or "")[-2000:],
            )
        else:
            logger.warning(
                "flux exit=0 for %s seed=%d but no PNG produced anywhere. "
                "stderr (last 2000 chars):\n%s",
                req.beat_id, seed, (proc.stderr or "")[-2000:],
            )
        return None

    def _mock_render(self, out_path: Path) -> Path:
        from PIL import Image
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (self.default_width, self.default_height),
                  color=(60, 35, 80)).save(out_path)
        return out_path


def _find_recovered_png(dir_path: Path, since: float,
                        min_bytes: int = 1000) -> Path | None:
    """Find the largest PNG/JPEG file in `dir_path` created since
    `since`. Used to recover the actual output of FLUX CLI versions
    that write to CWD or a default location instead of honoring the
    --output flag we passed.

    Returns the file path, or None if nothing eligible.
    """
    try:
        if not dir_path.exists() or not dir_path.is_dir():
            return None
    except OSError:
        return None

    candidates: list[Path] = []
    for f in dir_path.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if st.st_size < min_bytes:
            continue
        # Accept files mtime >= since-1s for clock-skew slack.
        if st.st_mtime + 1.0 < since:
            continue
        candidates.append(f)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


