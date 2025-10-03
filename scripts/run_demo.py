"""Launch the FastAPI server and Streamlit UI for the demo."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG chatbot demo stack")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the FastAPI server")
    parser.add_argument("--port", default="8000", help="Port for the FastAPI server")
    parser.add_argument(
        "--data-path",
        default="data/sample_docs",
        help="Path to the documents directory",
    )
    parser.add_argument(
        "--ui-port",
        default="8501",
        help="Port for the Streamlit app",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root))
    env.setdefault("RAG_API_URL", f"http://{args.host}:{args.port}")

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = repo_root / data_path
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    env.setdefault("RAG_DATA_PATH", str(data_path))

    server_cmd: List[str] = [
        sys.executable,
        "-m",
        "uvicorn",
        "ui.server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    ui_cmd: List[str] = [
        "streamlit",
        "run",
        str(repo_root / "ui" / "streamlit_app.py"),
        "--server.port",
        str(args.ui_port),
    ]

    processes: List[subprocess.Popen[bytes]] = []
    try:
        processes.append(subprocess.Popen(server_cmd, cwd=repo_root, env=env))
        time.sleep(1.0)
        processes.append(subprocess.Popen(ui_cmd, cwd=repo_root, env=env))
        print("Server running at http://{}:{}".format(args.host, args.port))
        print("Streamlit UI running at http://localhost:{}".format(args.ui_port))
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        for proc in processes:
            proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
