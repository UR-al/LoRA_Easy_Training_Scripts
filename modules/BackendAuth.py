"""The Authorization header the GUI sends to the backend.

With a token set, every backend URL gets 'Authorization: Bearer <token>', a local one (127.0.0.1 / localhost) too:
a backend started with LETS_BACKEND_TOKEN demands it on every request, loopback included (the GUI's own run_local
backend inherits the variable). The token comes from the LETS_BACKEND_TOKEN environment variable or else from the
"backend_token" key of config.json (the environment wins). With no token set nothing is sent, so the default local
setup goes out exactly as before, and a backend's refusal reaches the user like any other backend error.
"""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

TOKEN_ENV = "LETS_BACKEND_TOKEN"
CONFIG_KEY = "backend_token"


def backend_token(config: dict | None = None) -> str:
    """The backend token ("" when none is set). config is config.json's content; read when not given."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if token:
        return token
    if config is None:
        try:
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    token = config.get(CONFIG_KEY) if isinstance(config, dict) else None
    return str(token).strip() if token else ""


def backend_headers(url: str, config: dict | None = None) -> dict[str, str]:
    """Headers for a request to the backend at url: the bearer token when one is set, else nothing."""
    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        host = None
    if not host:  # not a URL requests can send to (no scheme): nothing to authorize
        return {}
    token = backend_token(config)
    return {"Authorization": f"Bearer {token}"} if token else {}
