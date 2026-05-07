from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config() -> dict[str, Any]:
    """Load config/default.yaml and return it as a nested dict."""
    with (CONFIG_DIR / "default.yaml").open() as f:
        return yaml.safe_load(f)
