"""Launcher tailored for small-organisation deployments."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .runtime import load_runtime_config, parse_common_args, ServiceRuntime

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "org.yaml"


def main(argv: list[str] | None = None) -> None:
    options = parse_common_args(argv or sys.argv[1:], str(DEFAULT_CONFIG))
    config = load_runtime_config(options)
    runtime = ServiceRuntime(config, include_ui=options.include_ui)

    print("[ragx] Starting organisation deployment")
    print(f"[ragx] Dataset: {config.dataset}")
    if config.index_path:
        print(f"[ragx] Shared index: {config.index_path}")
    if config.logging:
        print(f"[ragx] Logs directory: {config.logging.directory}")
    print(f"[ragx] API workers: {config.workers}")
    if config.rate_limit:
        print(
            f"[ragx] Rate limit: {config.rate_limit.requests_per_minute}/min"
            f" burst={config.rate_limit.burst}"
        )
    if config.api_keys:
        print(f"[ragx] API keys configured: {len(config.api_keys)}")
    print(f"[ragx] API endpoint: http://0.0.0.0:{config.api_port}")
    if options.include_ui:
        print(f"[ragx] UI endpoint: http://0.0.0.0:{config.ui_port}")

    asyncio.run(runtime.run(duration=options.run_duration))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

