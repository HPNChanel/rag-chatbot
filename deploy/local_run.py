"""One-click launcher for local RAG deployments."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .runtime import load_runtime_config, parse_common_args, ServiceRuntime

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "local.yaml"


def main(argv: list[str] | None = None) -> None:
    options = parse_common_args(argv or sys.argv[1:], str(DEFAULT_CONFIG))
    config = load_runtime_config(options)
    runtime = ServiceRuntime(config, include_ui=options.include_ui)
    print("[ragx] Starting local deployment")
    print(f"[ragx] Dataset: {config.dataset}")
    if config.index_path:
        print(f"[ragx] Index directory: {config.index_path}")
    print(f"[ragx] API:    http://localhost:{config.api_port}")
    if options.include_ui:
        print(f"[ragx] UI:     http://localhost:{config.ui_port}")
    asyncio.run(runtime.run(duration=options.run_duration))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

