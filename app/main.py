"""Entry point for the Project ADA voice assistant.

Boots the local STT and TTS services as child processes, confirms Ollama
is reachable, builds the pipeline dependencies, starts the VRAM monitor,
and drives :func:`orchestrator.pipeline.run_turn` from a console "press
Enter to speak" loop. ``--once`` runs a single turn and exits;
``--external-services`` skips the subprocess management (developer mode
where STT/TTS are already running).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import httpx

from app.deps import build_deps, build_session
from app.services import ServiceProcess
from core.config import load_config
from core.exceptions import ServiceUnavailableError
from core.logger import configure_logging, get_logger
from orchestrator.pipeline import PipelineDeps, run_turn
from orchestrator.session import Session
from orchestrator.vram_monitor import VramMonitor

_log = get_logger(__name__)

_OLLAMA_PROBE_TIMEOUT_S = 5.0
_DEFAULT_STARTUP_TIMEOUT_S = 120.0
_DEFAULT_SHUTDOWN_TIMEOUT_S = 10.0
_INPUT_PROMPT = "Press Enter to speak (Ctrl+C to quit)... "


def main() -> None:
    """Synchronous CLI wrapper around :func:`run`."""
    parser = argparse.ArgumentParser(prog="ada", description="Project ADA voice assistant.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single conversational turn and exit.",
    )
    parser.add_argument(
        "--external-services",
        action="store_true",
        help="Do not spawn STT/TTS subprocesses; assume they are already running.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_config()

    try:
        exit_code = asyncio.run(run(args, config))
    except KeyboardInterrupt:
        _log.info("interrupted")
        exit_code = 0
    except ServiceUnavailableError as e:
        _log.error("startup_failed", error=str(e))
        exit_code = 1
    sys.exit(exit_code)


async def run(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Top-level async entrypoint. Returns the process exit code."""
    await _check_ollama(config)

    services: list[ServiceProcess] = []
    if not args.external_services:
        services = _build_service_processes(config)

    try:
        for svc in services:
            await svc.start()

        deps = build_deps(config)
        session = build_session(config)
        vram = VramMonitor(config)
        await vram.start()

        try:
            if args.once:
                await run_turn(session, deps, config)
            else:
                await _interactive_loop(session, deps, config)
        finally:
            await vram.stop()
    finally:
        for svc in reversed(services):
            await svc.stop()

    return 0


async def _interactive_loop(
    session: Session,
    deps: PipelineDeps,
    config: dict[str, Any],
) -> None:
    """Drive ``run_turn`` once per Enter press until EOF or cancellation.

    When the previous turn ended on a barge-in (the user started speaking
    during playback), skip the Enter prompt and go straight into the
    next capture — the user is already talking.
    """
    skip_prompt = False
    while True:
        if not skip_prompt:
            try:
                await asyncio.to_thread(input, _INPUT_PROMPT)
            except EOFError:
                _log.info("stdin_closed")
                return
        skip_prompt = await run_turn(session, deps, config)


async def _check_ollama(config: dict[str, Any]) -> None:
    """Verify Ollama is reachable and the configured model is pulled.

    Raises :class:`ServiceUnavailableError` if Ollama is unreachable;
    logs a warning (and continues) if the model is not yet present —
    Ollama will pull it lazily on the first chat request.
    """
    llm_cfg = config["llm"]
    url = f"{llm_cfg['ollama_url'].rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=_OLLAMA_PROBE_TIMEOUT_S) as client:
            resp = await client.get(url)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError) as e:
        raise ServiceUnavailableError(
            f"Cannot reach Ollama at {url}: {e}",
            "Ollama is not running. Start it with 'ollama serve' and try again.",
        ) from e

    if resp.status_code != 200:
        raise ServiceUnavailableError(
            f"Ollama returned HTTP {resp.status_code} at {url}",
            "Ollama is not responding correctly.",
        )

    try:
        body: dict[str, Any] = resp.json()
    except ValueError as e:
        raise ServiceUnavailableError(
            f"Ollama returned invalid JSON at {url}: {e}",
            "Ollama is not responding correctly.",
        ) from e

    model_names = {m.get("name", "") for m in body.get("models", [])}
    configured = str(llm_cfg.get("model", ""))
    if configured and configured not in model_names:
        _log.warning(
            "ollama_model_not_pulled",
            model=configured,
            hint=f"Run 'ollama pull {configured}' to avoid a long first-turn delay.",
        )
    else:
        _log.info("ollama_ready", model=configured)


def _build_service_processes(config: dict[str, Any]) -> list[ServiceProcess]:
    """Construct the STT and TTS supervisors in startup order."""
    app_cfg = config.get("app", {})
    startup = float(app_cfg.get("service_startup_timeout_seconds", _DEFAULT_STARTUP_TIMEOUT_S))
    shutdown = float(app_cfg.get("service_shutdown_timeout_seconds", _DEFAULT_SHUTDOWN_TIMEOUT_S))
    return [
        ServiceProcess(
            name="stt",
            module="services.stt.server",
            health_url=f"http://127.0.0.1:{config['stt']['port']}/health",
            startup_timeout_s=startup,
            shutdown_timeout_s=shutdown,
        ),
        ServiceProcess(
            name="tts",
            module="services.tts.server",
            health_url=f"http://127.0.0.1:{config['tts']['port']}/health",
            startup_timeout_s=startup,
            shutdown_timeout_s=shutdown,
        ),
    ]


if __name__ == "__main__":
    main()
