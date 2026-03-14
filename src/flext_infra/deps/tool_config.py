"""Tool configuration loader for flext-infra dependency management."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files

from pydantic import ValidationError
from yaml import YAMLError, safe_load

from flext_core import r
from flext_infra import c, t
from flext_infra.deps._models import ToolConfigDocument


class FlextInfraDependencyToolConfig:
    """Load and cache dependency tool configuration."""

    _cache: r[ToolConfigDocument] | None = None

    @staticmethod
    def _load_tool_config_cached() -> r[ToolConfigDocument]:
        """Load, validate, and cache tool_config.yml."""
        if FlextInfraDependencyToolConfig._cache is not None:
            return FlextInfraDependencyToolConfig._cache
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
                result = r[ToolConfigDocument].fail(
                    "tool_config.yml must contain a top-level mapping",
                )
                FlextInfraDependencyToolConfig._cache = result
                return result
            payload: t.Infra.TomlConfig = dict(parsed_raw.items())
            validated = ToolConfigDocument.model_validate(payload)
            result = r[ToolConfigDocument].ok(validated)
            FlextInfraDependencyToolConfig._cache = result
            return result
        except (
            FileNotFoundError,
            OSError,
            YAMLError,
            ValidationError,
            TypeError,
        ) as exc:
            result = r[ToolConfigDocument].fail(
                f"failed to load tool_config.yml: {exc}",
            )
            FlextInfraDependencyToolConfig._cache = result
            return result

    @staticmethod
    def load_tool_config() -> r[ToolConfigDocument]:
        """Public cached accessor for tool_config.yml."""
        return FlextInfraDependencyToolConfig._load_tool_config_cached()


load_tool_config = FlextInfraDependencyToolConfig.load_tool_config


__all__ = [
    "FlextInfraDependencyToolConfig",
    "load_tool_config",
]
