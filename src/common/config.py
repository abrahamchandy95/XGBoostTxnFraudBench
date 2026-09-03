from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import yaml
from pydantic import TypeAdapter


Config: TypeAlias = dict[str, Any]
T = TypeVar("T")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root() / value


def load_raw_config(path: str | Path = "config.yaml") -> Config:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"configuration file does not exist: {resolved}; "
            "copy config.stage1.example.yaml to config.yaml"
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {resolved}")
    return payload


def load_config(
    section: str,
    adapter: TypeAdapter[T],
    path: str | Path = "config.yaml",
) -> T:
    config = load_raw_config(path)
    if section not in config:
        raise KeyError(f"configuration section {section!r} is missing")
    return adapter.validate_python(config[section])
