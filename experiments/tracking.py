"""Lightweight file-based experiment tracking."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable


class Tracker:
    """Track parameters, metrics, and artifacts under a run directory."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.params_path = self.run_dir / "params.json"
        self.metrics_csv = self.run_dir / "metrics.csv"
        self.metrics_jsonl = self.run_dir / "metrics.jsonl"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        if not self.metrics_csv.exists():
            self.metrics_csv.write_text("step,metric,value\n", encoding="utf-8")

    def log_params(self, **params: Any) -> None:
        existing: Dict[str, Any] = {}
        if self.params_path.exists():
            existing = json.loads(self.params_path.read_text(encoding="utf-8"))
        existing.update(params)
        self.params_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")

    def log_metric(self, metric: str, value: float, *, step: int | None = None) -> None:
        row = {
            "step": step if step is not None else 0,
            "metric": metric,
            "value": value,
        }
        with self.metrics_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "metric", "value"])
            writer.writerow(row)
        with self.metrics_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def log_artifact(self, path: Path | str) -> Path:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact {src} does not exist")
        dest = self.artifacts_dir / src.name
        if src.resolve() == dest.resolve():
            return dest
        dest.write_bytes(src.read_bytes())
        return dest


__all__ = ["Tracker"]
