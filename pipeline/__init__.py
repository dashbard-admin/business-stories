"""Business & Brand Origin Stories faceless YouTube pipeline."""

# SSL/cert verification defense: many .gov, .edu, and archive.org chains
# rely on roots that the python.org installer doesn't ship by default.
# truststore patches stdlib ssl to use the OS-native keychain on macOS;
# certifi is the fallback for commercial roots.
import os as _os

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

del _os

__version__ = "0.1.0"
