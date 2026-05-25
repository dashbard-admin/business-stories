"""Business & Brand Origin Stories faceless YouTube pipeline."""

# SSL/cert verification defense: many .gov, .edu, and archive.org chains
# rely on roots that the python.org installer doesn't ship by default.
# truststore patches stdlib ssl to use the OS-native keychain on macOS;
# certifi is the fallback for commercial roots.
import os as _os
from pathlib import Path as _Path

try:
    import truststore as _truststore
    _truststore.inject_into_ssl()
    del _truststore
except ImportError:
    pass

try:
    import certifi as _certifi
    _bundle = _certifi.where()
    _os.environ.setdefault("SSL_CERT_FILE", _bundle)
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _bundle)
    del _certifi, _bundle
except ImportError:
    pass


def _load_dotenv() -> None:
    """Load secrets from a .env file at the project root, if present.

    Secrets like XAI_API_KEY are kept OUT of config.yaml because
    GitHub's secret scanner rejects pushes containing them. The
    operator workflow is: copy .env.example to .env, fill in the
    keys, and .env is gitignored. This loader runs at package
    import so adapters can read os.environ[...] uniformly.

    Implementation is deliberately tiny — no python-dotenv
    dependency. Parses KEY=value lines, skipping comments and
    blanks, with optional surrounding quotes. Existing
    environment variables WIN over .env (so CLI / cron / launchd
    overrides aren't clobbered).
    """
    candidates = [
        _Path.cwd() / ".env",
        _Path(__file__).resolve().parent.parent / ".env",
    ]
    for path in candidates:
        try:
            if not path.exists():
                continue
            for raw in path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Allow optional `export ` prefix shell users sometimes
                # paste in from .envrc-style files.
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                # Strip surrounding matched quotes.
                if (len(val) >= 2 and val[0] == val[-1]
                        and val[0] in ("'", '"')):
                    val = val[1:-1]
                if key and key not in _os.environ:
                    _os.environ[key] = val
            return  # first file found wins
        except Exception:
            # .env parse failure must never block the package import
            continue


_load_dotenv()
del _load_dotenv, _Path, _os

__version__ = "0.1.0"
