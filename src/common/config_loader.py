"""Configuration loader. Reads settings.yaml and overlays .env."""
import os
import yaml
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_ENV_PATH = _PROJECT_ROOT / "config" / ".env"


def _load_env():
    """Load .env file into os.environ."""
    if _ENV_PATH.exists():
        with open(_ENV_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def load_config() -> dict[str, Any]:
    """Load YAML config, with env var substitution for ${VAR:default} patterns."""
    _load_env()
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = f.read()

    import re
    import string

    def replace_env(match):
        expr = match.group(1)
        var, sep, default = expr.partition(":")
        val = os.environ.get(var.strip(), default.strip() if default else "")
        return val

    raw = re.sub(r"\$\{([^}]+)\}", replace_env, raw)
    config = yaml.safe_load(raw)
    return config


# Singleton config instance
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config():
    global _config
    _config = load_config()
    return _config
