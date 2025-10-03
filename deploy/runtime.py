"""Shared helpers for deployment scripts."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import uvicorn

from analysis.redaction import RedactionSettings, TextRedactor
from configs.loader import DeploymentConfig, load_config
from monitoring import MetricsStore, configure_logging
from ui.server import create_app


@dataclass(slots=True)
class RuntimeOptions:
    config_path: Path
    overrides: dict[str, object]
    include_ui: bool = True
    run_duration: float | None = None


class ServiceRuntime:
    """Launch and manage API/UI services for deployments."""

    def __init__(self, config: DeploymentConfig, include_ui: bool = True) -> None:
        self.config = config
        self.include_ui = include_ui
        self.metrics = MetricsStore()
        self.logger = configure_logging(
            config.logging.directory if config.logging else None
        )
        self.redactor = (
            TextRedactor(RedactionSettings()) if config.enable_redaction else None
        )
        self._api_server: uvicorn.Server | None = None
        self._ui_process: asyncio.subprocess.Process | None = None

    async def run(self, duration: float | None = None) -> None:
        self._prepare_filesystem()
        app = create_app(
            config=self.config,
            metrics_store=self.metrics,
            logger=self.logger,
            redactor=self.redactor,
        )
        api_task = asyncio.create_task(self._serve_api(app))
        tasks = [api_task]
        if self.include_ui:
            tasks.append(asyncio.create_task(self._launch_streamlit()))

        if duration is not None:
            async def _auto_shutdown() -> None:
                await asyncio.sleep(duration)
                await self.shutdown()

            tasks.append(asyncio.create_task(_auto_shutdown()))

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(self.shutdown())
                )
            except NotImplementedError:  # pragma: no cover - Windows compatibility
                pass

        await asyncio.gather(*tasks)

    async def shutdown(self) -> None:
        if self._api_server is not None:
            self._api_server.should_exit = True
        if self._ui_process is not None and self._ui_process.returncode is None:
            with suppress(ProcessLookupError):
                self._ui_process.terminate()
                await asyncio.wait_for(self._ui_process.wait(), timeout=5)

    async def _serve_api(self, app) -> None:
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.api_port,
            log_level="info",
            workers=self.config.workers,
        )
        server = uvicorn.Server(config)
        self._api_server = server
        self.logger.info(
            "api_starting",
            extra={
                "extra_data": {
                    "port": self.config.api_port,
                    "workers": self.config.workers,
                }
            },
        )
        await server.serve()

    async def _launch_streamlit(self) -> None:
        env = {**os.environ, "RAG_API_URL": f"http://localhost:{self.config.api_port}"}
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py"),
            "--server.port",
            str(self.config.ui_port),
        ]
        self.logger.info(
            "ui_starting",
            extra={
                "extra_data": {
                    "port": self.config.ui_port,
                    "api_url": env["RAG_API_URL"],
                }
            },
        )
        self._ui_process = await asyncio.create_subprocess_exec(*command, env=env)
        await self._ui_process.wait()

    def _prepare_filesystem(self) -> None:
        if self.config.index_path:
            self.config.index_path.mkdir(parents=True, exist_ok=True)
        if self.config.logging:
            self.config.logging.directory.mkdir(parents=True, exist_ok=True)


def parse_common_args(argv: Iterable[str], default_config: str) -> RuntimeOptions:
    parser = argparse.ArgumentParser(description="RAG deployment runner")
    parser.add_argument("--config", default=default_config, help="Path to YAML config")
    parser.add_argument("--dataset", help="Override dataset path")
    parser.add_argument("--index-type", help="Index backend override")
    parser.add_argument("--reranker", help="Reranker override")
    parser.add_argument("--port-api", type=int, help="API port override")
    parser.add_argument("--port-ui", type=int, help="UI port override")
    parser.add_argument("--workers", type=int, help="Number of API workers")
    parser.add_argument("--index-path", help="Shared index directory")
    parser.add_argument("--log-dir", help="Directory for structured logs")
    parser.add_argument("--api-key", action="append", help="API keys allowed to access the API")
    parser.add_argument(
        "--rate-limit",
        type=int,
        help="Requests per minute allowed for each client",
    )
    parser.add_argument(
        "--rate-burst",
        type=int,
        help="Burst capacity for rate limiting",
    )
    parser.add_argument("--no-ui", action="store_true", help="Disable Streamlit UI")
    parser.add_argument(
        "--run-duration",
        type=float,
        help="Automatically shutdown after N seconds (useful for tests)",
    )
    args = parser.parse_args(list(argv))
    overrides: dict[str, object] = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.index_type:
        overrides["index_type"] = args.index_type
    if args.reranker:
        overrides["reranker"] = args.reranker
    if args.port_api:
        overrides["api_port"] = args.port_api
    if args.port_ui:
        overrides["ui_port"] = args.port_ui
    if args.workers:
        overrides["workers"] = args.workers
    if args.index_path:
        overrides["index_path"] = args.index_path
    if args.log_dir:
        overrides["logging"] = {"directory": args.log_dir}
    if args.api_key:
        overrides["api_keys"] = args.api_key
    if args.rate_limit:
        overrides["rate_limit"] = {"requests_per_minute": args.rate_limit}
    if args.rate_burst:
        overrides.setdefault("rate_limit", {})["burst"] = args.rate_burst
    return RuntimeOptions(
        config_path=Path(args.config),
        overrides=overrides,
        include_ui=not args.no_ui,
        run_duration=args.run_duration,
    )


def load_runtime_config(options: RuntimeOptions) -> DeploymentConfig:
    return load_config(options.config_path, options.overrides)


__all__ = ["RuntimeOptions", "ServiceRuntime", "parse_common_args", "load_runtime_config"]

