"""Pure-assembly factories for the launcher.

Centralises the construction of the long-lived collaborators
(:class:`orchestrator.pipeline.PipelineDeps`) and the per-process
:class:`orchestrator.session.Session` so the entrypoint module stays
focused on lifecycle and signal handling. No I/O happens here — every
client is built but no network calls are issued.
"""

from __future__ import annotations

from typing import Any

from ada_platform.detect import current_platform
from agent.executor import ToolExecutor
from agent.tools import build_registry
from orchestrator.audio_output import AudioPlayer
from orchestrator.barge_in import BargeInListener
from orchestrator.pipeline import PipelineDeps
from orchestrator.session import Session
from services.llm.client import LLMClient
from services.stt.client import STTClient
from services.tts.client import TTSClient


def build_session(config: dict[str, Any]) -> Session:
    """Return a fresh in-memory session tagged with the host platform.

    The ``config`` argument is accepted for symmetry with
    :func:`build_deps` even though it is not consulted today — future
    additions (e.g. preloaded system messages) will read from it.
    """
    return Session(platform=current_platform())


def build_deps(config: dict[str, Any]) -> PipelineDeps:
    """Build the :class:`PipelineDeps` bundle from a loaded config dict.

    The shape mismatch between :func:`agent.tools.build_registry` (dict)
    and :attr:`PipelineDeps.tools` (list) is bridged here so the rest of
    the codebase does not need to know about it.
    """
    registry = build_registry(config)
    return PipelineDeps(
        stt=STTClient(config),
        llm=LLMClient(config),
        tts=TTSClient(config),
        player=AudioPlayer(config),
        tools=list(registry.values()),
        executor=ToolExecutor(registry),
        barge_in=BargeInListener(config),
    )
