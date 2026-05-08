"""Platform detection for Project ADA.

Returns a normalized platform name (``"windows"`` or ``"linux"``) so other
modules can dispatch to the correct platform-specific implementation without
ever calling ``sys.platform`` themselves.
"""

from __future__ import annotations

import sys


def current_platform() -> str:
    """Return the current OS as ``"windows"`` or ``"linux"``.

    Raises:
        RuntimeError: If the host OS is neither Windows nor Linux.
    """
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Unsupported platform: {sys.platform!r}")
