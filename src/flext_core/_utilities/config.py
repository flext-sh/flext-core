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
from typing import TYPE_CHECKING, ClassVar, cast

import yaml

from flext_core import FlextStrictYamlConfigSource, r
from flext_core._constants import FlextConstantsConfig
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities import (
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesReliability,
)

if TYPE_CHECKING:
    from flext_core import p

if TYPE_CHECKING:
    from collections.abc import Mapping


class FlextUtilitiesConfig:
    """Minimal stdlib-backed config load, merge, and env-override helpers."""

    class Yaml:
        """Owner facade for the yaml primitives config consumers route through.

        Consumer projects must reach yaml via ``u.Config.Yaml`` so no consumer
        module imports yaml directly (transport ownership stays with
        flext-core, ENFORCE-070).
        """

        YAMLError = yaml.YAMLError

        @staticmethod
        def safe_load(stream: str) -> FlextTypingBase.JsonValue:
            """Parse a YAML string → validated JSON value.

            Returns:
                The resulting ``t.JsonValue``.

            """
            return cast("FlextTypingBase.JsonValue", yaml.safe_load(stream))

        @staticmethod
        def safe_dump(
            data: FlextTypingBase.JsonValue | FlextTypingBase.JsonMapping,
            *,
            sort_keys: bool = False,
            indent: int = 2,
            allow_unicode: bool = True,
            default_flow_style: bool = False,
        ) -> str:
            """Serialize a JSON value → YAML string.

            Returns:
                The resulting ``str``.

            """
            return yaml.safe_dump(
                data,
                sort_keys=sort_keys,
                indent=indent,
                allow_unicode=allow_unicode,
                default_flow_style=default_flow_style,
            )

        @staticmethod
        def safe_load_file(path: Path) -> FlextTypingBase.JsonValue:
            """Load a YAML file → validated JSON value.

            Returns:
                The resulting ``t.JsonValue``.

            """
            with path.open(encoding="utf-8") as fh:
                return cast("FlextTypingBase.JsonValue", yaml.safe_load(fh))

        @staticmethod
        def unique_key_load(stream: str) -> FlextTypingBase.JsonValue:
            """Parse YAML rejecting duplicate mapping keys at every depth.

            Why: duplicate config keys silently overwrite their predecessor at
            plain ``safe_load`` time, so a config consumer that must treat a
            duplicated key as an error (one owner per key) needs the parse-time
            guard. The loader is the same one the settings sources use, so the
            contract has exactly one implementation and this facade is its only
            public door. It is a ``yaml.SafeLoader`` subclass — the safe
            constructor set, no object-deserialization primitives — so the
            arbitrary-constructor unsafe loader path stays out of this module.

            Returns:
                The resulting ``t.JsonValue``.

            Raises ``yaml.YAMLError`` on malformed input or a duplicate
            mapping key (propagated from the canonical loader).

            """
            return FlextStrictYamlConfigSource.unique_key_load(stream)

        @staticmethod
        def yaml_safe_load(path: Path) -> p.Result[FlextTypingBase.JsonMapping]:
            """Load a YAML file → ``r[JsonMapping]``.

            Returns:
                The resulting ``p.Result[t.JsonMapping]``.

            """
            if not path.is_file():
                return r[FlextTypingBase.JsonMapping].fail(
                    f"YAML file not found: {path}",
                )
            try:
                loaded = FlextUtilitiesConfig.Yaml.safe_load_file(path)
            except yaml.YAMLError as exc:
                return r[FlextTypingBase.JsonMapping].fail(
                    f"YAML parse error: {exc}",
                    exception=exc,
                )
            except OSError as exc:
                return r[FlextTypingBase.JsonMapping].fail(
                    f"YAML read error: {exc}",
                    exception=exc,
                )
            if not FlextUtilitiesGuardsTypeCore.mapping(loaded):
                return r[FlextTypingBase.JsonMapping].fail(
                    f"YAML top level is not a mapping: {path}",
                )
            return r[FlextTypingBase.JsonMapping].ok(loaded)

        @staticmethod
        def yaml_dump(
            path: Path,
            data: FlextTypingBase.JsonMapping,
            *,
            sort_keys: bool = False,
            indent: int = 2,
        ) -> p.Result[bool]:
            """Write a payload as YAML file → ``r[bool]``.

            Returns:
                The resulting ``p.Result[bool]``.

            """
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                validated = FlextUtilitiesConfig.Yaml.safe_dump(
                    data,
                    sort_keys=sort_keys,
                    indent=indent,
                )
                with path.open("w", encoding="utf-8") as fh:
                    fh.write(validated)
                return r[bool].ok(value=True)
            except OSError as exc:
                return r[bool].fail(f"YAML write error: {exc}", exception=exc)

    _EXPAND_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^{}]*))?\}",
    )

    @staticmethod
    def _expand_one(match: re.Match[str], env: Mapping[str, str]) -> str:
        """Resolve one ``${VAR}`` / ``${VAR:-default}`` match against ``env``.

        Returns:
            The resulting ``str``.

        """
        name = match.group("name")
        if name in env:
            return env[name]
        default = match.group("default")
        return default if default is not None else ""

    @staticmethod
    def _expand_str(value: str, env: Mapping[str, str]) -> str:
        """Expand innermost ``${...}`` repeatedly so nested defaults resolve.

        Returns:
            The resulting ``str``.

        """
        current = value

        def _expand_match(match: re.Match[str]) -> str:
            return FlextUtilitiesConfig._expand_one(match, env)

        for _ in range(FlextConstantsConfig.CONFIG_EXPAND_MAX_PASSES):
            expanded = FlextUtilitiesConfig._EXPAND_PATTERN.sub(_expand_match, current)
            if expanded == current:
                return expanded
            current = expanded
        return current

    @staticmethod
    def config_load(path: Path) -> p.Result[FlextTypingBase.JsonMapping]:
        """Load and parse a TOML config source into a validated mapping.

        Fail-closed: a missing file, parse error, or non-mapping top level is a
        failed ``r[T]``, never a raised exception escaping ``config_load``.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        if not path.is_file():
            return r[FlextTypingBase.JsonMapping].fail(
                f"{FlextConstantsConfig.ERR_CONFIG_READ_FAILED}: {path}",
            )
        parsed = FlextUtilitiesReliability.try_(
            lambda: tomllib.loads(
                path.read_text(encoding=FlextConstantsConfig.CONFIG_DEFAULT_ENCODING),
            ),
            catch=(OSError, tomllib.TOMLDecodeError),
            op_name="config_load",
        )
        if parsed.failure:
            return r[FlextTypingBase.JsonMapping].from_failure(parsed)
        payload = parsed.value
        if not FlextUtilitiesGuardsTypeCore.mapping(payload):
            return r[FlextTypingBase.JsonMapping].fail(
                f"{FlextConstantsConfig.ERR_CONFIG_NOT_MAPPING}: {path}",
            )
        return r[FlextTypingBase.JsonMapping].ok(payload)

    @staticmethod
    def config_merge(
        base: FlextTypingBase.JsonMapping,
        override: FlextTypingBase.JsonMapping,
    ) -> FlextTypingBase.JsonDict:
        """Deep-merge ``override`` onto ``base``, returning a new mapping.

        Returns:
            The resulting ``t.JsonDict``.

        """
        merged: dict[str, FlextTypingBase.JsonValue] = dict(base)
        for key, value in override.items():
            current = merged.get(key)
            if FlextUtilitiesGuardsTypeCore.mapping(
                current,
            ) and FlextUtilitiesGuardsTypeCore.mapping(value):
                nested: FlextTypingBase.JsonValue = dict(
                    FlextUtilitiesConfig.config_merge(current, value),
                )
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    @staticmethod
    def config_env_override(
        value: FlextTypingBase.JsonValue,
        env: Mapping[str, str],
    ) -> FlextTypingBase.JsonValue:
        """Expand ``${VAR}`` / ``${VAR:-default}`` placeholders in string leaves.

        Recurses through mappings and sequences; non-string leaves pass through
        unchanged. ``${VAR}`` resolves to ``env[VAR]`` or ``""`` when absent;
        ``${VAR:-default}`` resolves to ``env[VAR]`` or ``default`` when absent.

        Returns:
            The resulting ``t.JsonValue``.

        """
        if isinstance(value, str):
            return FlextUtilitiesConfig._expand_str(value, env)
        if FlextUtilitiesGuardsTypeCore.mapping(value):
            return {
                key: FlextUtilitiesConfig.config_env_override(item, env)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                FlextUtilitiesConfig.config_env_override(item, env) for item in value
            ]
        return value


__all__: FlextTypingBase.MutableSequenceOf[str] = ["FlextUtilitiesConfig"]
