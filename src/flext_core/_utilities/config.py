"""FlextUtilitiesConfig - minimal declarative config helpers (ADR-005).

Runtime-minimal config primitives for flext-core self-configuration: TOML load
(stdlib ``tomllib``), deep merge, and ``${VAR}`` env expansion (stdlib
``string.Template``). **No Jinja2, no YAML, no JSON Schema here** — those live in
``flext-cli`` (``u.Cli.config_load`` / ``u.Cli.render_template`` /
``u.Cli.yaml_validate_schema``), which imports and amplifies these primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from flext_core._constants.config import FlextConstantsConfig as c
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore as g
from flext_core._utilities.reliability import FlextUtilitiesReliability as rel
from flext_core.result import r

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_core import FlextProtocols as p


class FlextUtilitiesConfig:
    """Minimal stdlib-backed config load, merge, and env-override helpers."""

    _EXPAND_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^{}]*))?\}"
    )

    @staticmethod
    def _expand_one(match: re.Match[str], env: Mapping[str, str]) -> str:
        """Resolve one ``${VAR}`` / ``${VAR:-default}`` match against ``env``."""
        name = match.group("name")
        if name in env:
            return env[name]
        default = match.group("default")
        return default if default is not None else ""

    @staticmethod
    def _expand_str(value: str, env: Mapping[str, str]) -> str:
        """Expand innermost ``${...}`` repeatedly so nested defaults resolve."""
        current = value
        for _ in range(c.CONFIG_EXPAND_MAX_PASSES):
            expanded = FlextUtilitiesConfig._EXPAND_PATTERN.sub(
                lambda match: FlextUtilitiesConfig._expand_one(match, env), current
            )
            if expanded == current:
                return expanded
            current = expanded
        return current

    @staticmethod
    def config_load(path: Path) -> p.Result[t.JsonMapping]:
        """Load and parse a TOML config source into a validated mapping.

        Fail-closed: a missing file, parse error, or non-mapping top level is a
        failed ``r[T]``, never a raised exception escaping ``config_load``.
        """
        if not path.is_file():
            return r[t.JsonMapping].fail(f"{c.ERR_CONFIG_READ_FAILED}: {path}")
        parsed = rel.try_(
            lambda: tomllib.loads(path.read_text(encoding=c.CONFIG_DEFAULT_ENCODING)),
            catch=(OSError, tomllib.TOMLDecodeError),
            op_name="config_load",
        )
        if parsed.failure:
            return r[t.JsonMapping].fail(
                parsed.error or f"{c.ERR_CONFIG_PARSE_FAILED}: {path}"
            )
        payload = parsed.value
        if not g.mapping(payload):
            return r[t.JsonMapping].fail(f"{c.ERR_CONFIG_NOT_MAPPING}: {path}")
        return r[t.JsonMapping].ok(payload)

    @staticmethod
    def config_merge(base: t.JsonMapping, override: t.JsonMapping) -> t.JsonDict:
        """Deep-merge ``override`` onto ``base``, returning a new mapping."""
        merged: dict[str, t.JsonValue] = dict(base)
        for key, value in override.items():
            current = merged.get(key)
            if g.mapping(current) and g.mapping(value):
                nested: t.JsonValue = dict(
                    FlextUtilitiesConfig.config_merge(current, value)
                )
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    @staticmethod
    def config_env_override(value: t.JsonValue, env: Mapping[str, str]) -> t.JsonValue:
        """Expand ``${VAR}`` / ``${VAR:-default}`` placeholders in string leaves.

        Recurses through mappings and sequences; non-string leaves pass through
        unchanged. ``${VAR}`` resolves to ``env[VAR]`` or ``""`` when absent;
        ``${VAR:-default}`` resolves to ``env[VAR]`` or ``default`` when absent.
        """
        if isinstance(value, str):
            return FlextUtilitiesConfig._expand_str(value, env)
        if g.mapping(value):
            return {
                key: FlextUtilitiesConfig.config_env_override(item, env)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                FlextUtilitiesConfig.config_env_override(item, env) for item in value
            ]
        return value


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesConfig"]
