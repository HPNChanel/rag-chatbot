"""Helpers for loading and validating deployment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class RateLimitConfig(BaseModel):
    """Basic token-bucket configuration for the API server."""

    requests_per_minute: int = Field(..., gt=0, description="Allowed requests per minute")
    burst: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum burst capacity. Defaults to the same value as ``requests_per_minute`` "
            "when not provided."
        ),
    )

    @model_validator(mode="after")
    def _validate_burst(self) -> "RateLimitConfig":  # pragma: no cover - trivial
        if self.burst is None:
            object.__setattr__(self, "burst", self.requests_per_minute)
        return self


class LoggingConfig(BaseModel):
    """Location for structured logs."""

    directory: Path

    @model_validator(mode="after")
    def _ensure_directory(self) -> "LoggingConfig":
        self.directory = self.directory.expanduser().resolve()
        return self


class DeploymentConfig(BaseModel):
    """Schema covering both local and small-organisation deployments."""

    mode: str = Field(description="Deployment mode (local or org)")
    dataset: Path = Field(description="Path to the dataset to ingest")
    index_type: str = Field(description="Index backend to use (faiss, bm25, hybrid)")
    api_port: int = Field(gt=0, lt=65536)
    ui_port: int = Field(gt=0, lt=65536)
    reranker: str | None = None
    index_path: Optional[Path] = None
    workers: int = Field(default=1, gt=0)
    enable_redaction: bool = Field(default=False)
    logging: Optional[LoggingConfig] = None
    rate_limit: Optional[RateLimitConfig] = None
    api_keys: Optional[list[str]] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def _post_init(self) -> "DeploymentConfig":
        mode = self.mode.lower()
        if mode not in {"local", "org"}:
            raise ValueError("mode must be 'local' or 'org'")
        object.__setattr__(self, "mode", mode)
        self.dataset = self.dataset.expanduser().resolve()
        if self.index_path is not None:
            self.index_path = self.index_path.expanduser().resolve()
        if self.api_keys:
            if len({key.strip() for key in self.api_keys if key.strip()}) == 0:
                raise ValueError("api_keys must contain at least one non-empty key")
            self.api_keys = [key.strip() for key in self.api_keys if key.strip()]
        if self.mode == "org" and not self.index_path:
            raise ValueError("Organisation deployments require an index_path")
        if self.mode == "org" and not self.logging:
            raise ValueError("Organisation deployments require a logging directory")
        return self


DEFAULT_VALUES: Dict[str, Any] = {
    "mode": "local",
    "dataset": "data/sample_docs",
    "index_type": "faiss",
    "api_port": 8000,
    "ui_port": 8501,
    "reranker": "coverage",
    "workers": 1,
}


def _deep_merge(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> DeploymentConfig:
    """Load a deployment config, merging defaults, file contents, and overrides."""

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = DEFAULT_VALUES.copy()
    with config_path.open("r", encoding="utf-8") as handle:
        file_values = yaml.safe_load(handle) or {}
    if not isinstance(file_values, Mapping):
        raise TypeError("Config file must contain a mapping at the top level")
    merged: Dict[str, Any] = _deep_merge(raw, dict(file_values))  # type: ignore[arg-type]
    if overrides:
        merged = _deep_merge(merged, dict(overrides))
    try:
        return DeploymentConfig(**merged)
    except ValidationError as exc:  # pragma: no cover - handled via tests, defensive
        raise ValueError(f"Invalid deployment config: {exc}") from exc


__all__ = [
    "DeploymentConfig",
    "LoggingConfig",
    "RateLimitConfig",
    "load_config",
]

