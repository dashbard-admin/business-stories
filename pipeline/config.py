"""Configuration loader.

Reads config.yaml from the package root (or path in PIPELINE_CONFIG env
var) and exposes a singleton `Config` object. Resolves ${root} and
${path.subkey} substitution tokens.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    # ----- convenience accessors -----

    @property
    def state_dir(self) -> Path:
        return Path(self._resolve(self.raw["paths"]["state"]))

    @property
    def episodes_dir(self) -> Path:
        return Path(self._resolve(self.raw["paths"]["episodes"]))

    @property
    def assets_dir(self) -> Path:
        return Path(self._resolve(self.raw["paths"]["assets"]))

    @property
    def logs_dir(self) -> Path:
        return Path(self._resolve(self.raw["paths"]["logs"]))

    @property
    def prompts_dir(self) -> Path:
        return Path(self._resolve(self.raw["paths"]["prompts"]))

    @property
    def style_profiles_dir(self) -> Path:
        return Path(self._resolve(self.raw["paths"]["style_profiles"]))

    @property
    def mock_mode(self) -> bool:
        return bool(self.raw["models"].get("mock_mode", False))

    @property
    def channel(self) -> dict[str, Any]:
        return self.raw["channel"]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def production(self) -> dict[str, Any]:
        return self.raw["production"]

    @property
    def quality_gates(self) -> dict[str, Any]:
        return self.raw["quality_gates"]

    @property
    def constraints(self) -> dict[str, Any]:
        return self.raw["constraints"]

    @property
    def orchestrator(self) -> dict[str, Any]:
        return self.raw["orchestrator"]

    @property
    def upload(self) -> dict[str, Any]:
        return self.raw.get("upload", {"mode": "manual"})

    @property
    def search(self) -> dict[str, Any]:
        defaults = {
            "backend": "searxng",
            "searxng_url": "http://127.0.0.1:8080",
            "engines": "",
            "safesearch": 0,
            "results_per_query": 30,
            "user_agent": "BusinessStoriesPipeline/0.1 (+research)",
            "request_timeout_seconds": 30,
        }
        return {**defaults, **(self.raw.get("search") or {})}

    @property
    def music_library(self) -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "path": str(self.assets_dir / "music_library"),
            "manifest": str(self.assets_dir / "music_library" / "manifest.json"),
            "crossfade_seconds": 4,
            "music_gain_db": -28.0,
            "music_start_offset_seconds": 20.0,
            "voice_dynaudnorm_enabled": True,
            "voice_dynaudnorm_framelen_ms": 200,
            "voice_dynaudnorm_gauss": 11,
            "voice_dynaudnorm_max_gain": 15.0,
        }
        raw = self.raw.get("music_library") or {}
        merged = {**defaults, **raw}
        # Resolve path tokens
        merged["path"] = self._resolve(str(merged["path"]))
        merged["manifest"] = self._resolve(str(merged["manifest"]))
        return merged

    @property
    def flux_cli(self) -> dict[str, Any]:
        defaults = {
            "binary": "flux",
            "width": 1920,
            "height": 1080,
            "steps": 24,
            "timeout_seconds": 600,
            "fold_negative_into_prompt": True,
        }
        return {**defaults, **(self.raw.get("flux_cli") or {})}

    @property
    def image_qa(self) -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "max_attempts_per_beat": 2,
            "timeout_seconds": 30,
            "caption_pd_assets": True,
            "strict_borderline": True,
            "pd_direct_use_threshold": 0.20,
            "pd_reference_threshold": 0.20,
            "pd_max_reuses_per_asset": 30,
            "pd_reference_strength": 0.1,
            "pd_image_reference_enabled": False,
            "flux_force_no_text": True,
        }
        return {**defaults, **(self.raw.get("image_qa") or {})}

    @property
    def generic_stash(self) -> dict[str, Any]:
        defaults = {"enabled": True, "threshold": 0.18, "max_reuses_per_asset": 5}
        return {**defaults, **(self.raw.get("generic_stash") or {})}

    @property
    def stock_sources(self) -> dict[str, Any]:
        return self.raw.get("stock_sources") or {"enabled": False}

    @property
    def pd_upscale(self) -> dict[str, Any]:
        return self.raw.get("pd_upscale") or {"enabled": False}

    @property
    def archetypes(self) -> list[dict[str, Any]]:
        return self.raw["archetypes"]

    @property
    def narrators(self) -> list[dict[str, Any]]:
        return self.raw["narrators"]

    @property
    def visual_styles(self) -> list[dict[str, Any]]:
        return self.raw["visual_styles"]

    # ----- helpers -----

    def _resolve(self, s: str) -> str:
        """Expand ${root} and any other ${path.subkey} tokens."""
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key == "root":
                return str(self.root)
            parts = key.split(".")
            cur: Any = self.raw
            for p in parts:
                cur = cur[p]
            return self._resolve(str(cur)) if isinstance(cur, str) else str(cur)
        return re.sub(r"\$\{([^}]+)\}", replace, s)

    def archetype_by_id(self, aid: str) -> dict[str, Any]:
        for a in self.archetypes:
            if a["id"] == aid:
                return a
        raise KeyError(aid)

    def narrator_by_id(self, nid: str) -> dict[str, Any]:
        for n in self.narrators:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    def visual_style_by_id(self, vid: str) -> dict[str, Any]:
        for v in self.visual_styles:
            if v["id"] == vid:
                return v
        raise KeyError(vid)


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load and cache the global config object.

    The project root is derived from the location of config.yaml on
    disk — config.yaml is always at the project root, so this works
    regardless of where the repository is cloned. The `paths.root`
    field in the YAML is treated as an optional override: if it's
    present and resolves to an existing directory, we use it
    (so operators can host the workspace on a different drive); if
    it's stale or missing, we silently fall back to the file-derived
    root.
    """
    import logging
    config_path = os.environ.get("PIPELINE_CONFIG")
    if config_path:
        path = Path(config_path).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent.parent / "config.yaml"

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    # Canonical root: the directory containing config.yaml.
    file_root = path.parent.resolve()

    # Configured override (optional).
    configured = (raw.get("paths") or {}).get("root")
    if configured:
        configured_root = Path(configured).expanduser().resolve()
    else:
        configured_root = file_root

    # Prefer the file-derived root when the configured one is stale
    # (e.g. a different developer's checkout path baked into the YAML).
    if configured_root != file_root and not configured_root.exists():
        logging.getLogger("hermes.config").info(
            "paths.root in %s (%s) does not exist; using config-file location %s instead",
            path, configured_root, file_root,
        )
        root = file_root
    else:
        # Either the YAML's root exists (operator deliberately pointed
        # elsewhere) or it matches the file-derived root. Use it.
        root = configured_root

    # Reflect the resolved root back into the raw dict so the
    # ${root} token in nested path entries (state, episodes, assets…)
    # resolves correctly without needing operator edits.
    raw.setdefault("paths", {})["root"] = str(root)

    cfg = Config(raw=raw, root=root)

    for d in (cfg.state_dir, cfg.episodes_dir, cfg.assets_dir, cfg.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "locks").mkdir(parents=True, exist_ok=True)

    return cfg
