"""Settings helpers for parameter access and manipulation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from flext_core import FlextUtilitiesGuardsTypeModel, c, e, p, r, t


class FlextUtilitiesSettings:
    """Settings utilities for parameter access and manipulation."""

    type _ResolvedParameter = t.ValueOrModel | p.Model | None

    @staticmethod
    def _duck_dump_get_parameter(
        obj: p.HasModelDump | p.Model | p.Base,
        parameter: str,
    ) -> tuple[bool, FlextUtilitiesSettings._ResolvedParameter]:
        """Get parameter from duck-typed model_dump() result.

        Instead of iterating the dump dict (which has Unknown types from getattr),
        directly look up the parameter using subscript access.
        """
        model_dump_fn = getattr(obj, "model_dump", None)
        if model_dump_fn is None or not callable(model_dump_fn):
            return (False, None)
        raw = model_dump_fn()
        get_fn = getattr(raw, "get", None)
        if get_fn is None or not callable(get_fn):
            return (False, None)

        class _Sentinel:
            pass

        sentinel = _Sentinel()
        val = get_fn(parameter, sentinel)
        if val is sentinel:
            return (False, None)
        if val is None or isinstance(val, (str, int, float, bool, datetime, Path)):
            return (True, val)
        if isinstance(val, p.Model) and FlextUtilitiesGuardsTypeModel.base_model(val):
            return (True, val)
        return (True, str(val))

    @staticmethod
    def resolve_log_level_from_settings() -> int:
        """Resolve numeric log level from default constant."""
        default_log_level = c.DEFAULT_LEVEL.upper()
        return getattr(logging, default_log_level, logging.INFO)

    @staticmethod
    def resolve_env_file() -> str:
        """Resolve .env file path from FLEXT_ENV_FILE env var."""
        # Bootstrap: resolves env file before FlextSettings exists (AGENTS.md §2.6 exception)
        custom_env_file = os.environ.get(c.ENV_FILE_ENV_VAR)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            return custom_env_file
        default_path = Path.cwd() / c.ENV_FILE_DEFAULT
        if default_path.exists():
            return str(default_path.resolve())
        return c.ENV_FILE_DEFAULT

    _NOT_FOUND: tuple[bool, None] = (False, None)

    @staticmethod
    def _try_get_attr(
        obj: p.HasModelDump | p.Model | p.Base,
        parameter: str,
    ) -> tuple[bool, FlextUtilitiesSettings._ResolvedParameter]:
        """Try direct attribute access, returning (found, value) tuple."""
        obj_vars: t.RecursiveContainerMapping = (
            vars(obj) if hasattr(obj, "__dict__") else {}
        )
        if parameter not in obj_vars:
            return FlextUtilitiesSettings._NOT_FOUND
        value = obj_vars[parameter]
        return (True, value)

    @staticmethod
    def _try_get_from_dict_like(
        obj: Mapping[str, t.ValueOrModel],
        parameter: str,
    ) -> tuple[bool, FlextUtilitiesSettings._ResolvedParameter]:
        """Try dict-like key lookup, returning (found, value) tuple."""
        if parameter in obj:
            return (True, obj[parameter])
        return FlextUtilitiesSettings._NOT_FOUND

    @staticmethod
    def _try_get_from_duck_model_dump(
        obj: p.HasModelDump | p.Model | p.Base,
        parameter: str,
    ) -> tuple[bool, FlextUtilitiesSettings._ResolvedParameter]:
        try:
            return FlextUtilitiesSettings._duck_dump_get_parameter(obj, parameter)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        return FlextUtilitiesSettings._NOT_FOUND

    @staticmethod
    def _try_get_from_model_dump(
        obj: p.HasModelDump,
        parameter: str,
    ) -> tuple[bool, FlextUtilitiesSettings._ResolvedParameter]:
        """Try model_dump() field lookup, returning (found, value) tuple."""
        return FlextUtilitiesSettings._try_get_from_duck_model_dump(obj, parameter)

    @staticmethod
    def register_factory(
        container: p.Container,
        name: str,
        factory: Callable[[], t.Scalar | t.ConfigMap | t.Dict],
    ) -> p.Result[bool]:
        """Register factory in DI container with optional caching."""
        _ = container.factory(name, factory)
        resolved = container.resolve(name)
        if resolved.failure:
            return e.fail_operation(
                "resolve registered config factory",
                resolved.error or c.ERR_CONFIG_FACTORY_REGISTRATION_FAILED,
            )
        return r[bool].ok(True)

    @staticmethod
    def resolve_effective_log_level(
        *,
        trace: bool,
        debug: bool,
        log_level: c.LogLevel,
    ) -> c.LogLevel:
        """Resolve log level: DEBUG if trace, INFO if debug, else log_level."""
        if trace:
            return c.LogLevel.DEBUG
        if debug:
            return c.LogLevel.INFO
        return log_level

    @staticmethod
    def validate_database_url_scheme(url: str) -> None:
        """Validate database URL scheme is postgresql://, mysql://, or sqlite://."""
        if url and not url.startswith(("postgresql://", "mysql://", "sqlite://")):
            raise ValueError(c.ERR_CONFIG_INVALID_DB_URL_SCHEME)

    @staticmethod
    def validate_trace_requires_debug(*, trace: bool, debug: bool) -> None:
        """Validate that trace mode requires debug mode."""
        if trace and not debug:
            raise ValueError(c.ERR_CONFIG_TRACE_REQUIRES_DEBUG)


__all__: list[str] = ["FlextUtilitiesSettings"]
