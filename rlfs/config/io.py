# rlfs/config/io.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


def _ensure_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(obj) if is_dataclass(obj) else obj
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = _ensure_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
    # Assumes cls is a dataclass with keyword args
    return cls(**d)  # type: ignore[arg-type]


def save_yaml(obj: Any, path: str | Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is not installed; cannot save YAML.") from e

    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(obj) if is_dataclass(obj) else obj

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is not installed; cannot load YAML.") from e

    path = _ensure_path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} did not contain a dict.")
    return data
