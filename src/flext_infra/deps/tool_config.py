"""Tool configuration loader for flext-infra dependency management."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files

from pydantic import ValidationError
from yaml import YAMLError, safe_load

from flext_core import r
from flext_infra import c, t
from flext_infra.deps._models import ToolConfigDocument


@lru_cache(maxsize=1)
def _load_tool_config_cached() -> r[ToolConfigDocument]:
    """Load, validate, and cache tool_config.yml."""
    try:
        raw_text = (
            files("flext_infra.deps")
            .joinpath("tool_config.yml")
            .read_text(
                encoding=c.Infra.Encoding.DEFAULT,
            )
        )
        parsed_raw: t.Infra.TomlValue | None = safe_load(raw_text)
        if not isinstance(parsed_raw, Mapping):
            return r[ToolConfigDocument].fail(
                "tool_config.yml must contain a top-level mapping",
            )
        payload: t.Infra.TomlConfig = dict(parsed_raw.items())
        validated = ToolConfigDocument(payload)
        return r[ToolConfigDocument].ok(validated)
    except (FileNotFoundError, OSError, YAMLError, ValidationError, TypeError) as exc:
        return r[ToolConfigDocument].fail(
            f"failed to load tool_config.yml: {exc}",
        )


def load_tool_config() -> r[ToolConfigDocument]:
    """Public cached accessor for tool_config.yml."""
    return _load_tool_config_cached()


__all__ = [
    "load_tool_config",
]
