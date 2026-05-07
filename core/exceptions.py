"""Custom exception hierarchy for Project ADA.

Every exception carries a ``user_message`` field — a short, human-readable
string safe to speak aloud via TTS. The standard exception ``args[0]`` holds
the technical message intended for logs and developers.
"""

from __future__ import annotations


class ADAError(Exception):
    """Base exception for all Project ADA errors.

    Args:
        message: Technical message for logs and developers.
        user_message: Short, plain-language message safe to speak via TTS.
    """

    def __init__(self, message: str, user_message: str) -> None:
        super().__init__(message)
        self.user_message = user_message


class ServiceUnavailableError(ADAError):
    """A required local service (STT, TTS, Ollama) did not respond."""


class ConfigurationError(ADAError):
    """Configuration is missing a required key or has an invalid value."""


class ToolExecutionError(ADAError):
    """An agent tool raised an unexpected error during execution."""


class VRAMCriticalError(ADAError):
    """VRAM usage is too high to continue safely."""
