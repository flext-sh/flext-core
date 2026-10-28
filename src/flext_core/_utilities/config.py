"""FlextUtilitiesConfig - minimal declarative config helpers (ADR-005).

Runtime config primitives for flext-core self-configuration: TOML loading,
strict YAML configuration ingress, deep merge and ``${VAR}`` expansion.
The CLI composes these primitives with template rendering, YAML editing and
schema validation through its own public utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import yaml
from pydantic import ValidationError

from flext_core import FlextStrictYamlConfigSource, c, r, t
from flext_core._utilities import (
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesReliability,
    FlextUtilitiesTypingAdapterJson,
)

if TYPE_CHECKING:
    from flext_core import p


class FlextUtilitiesConfig:
    """Minimal stdlib-backed config load, merge, and env-override helpers."""

    class Yaml:
        """Owner facade for the yaml primitives config consumers route through.

        Consumer projects reach these primitives through ``u.Yaml`` so no consumer
        module imports yaml directly (transport ownership stays with
        flext-core, ENFORCE-070).
        """

        YAMLError = yaml.YAMLError

        @staticmethod
        def safe_load(stream: str) -> t.JsonValue:
            """Parse a YAML string → validated JSON value.

            Returns:
                The resulting ``t.JsonValue``.

            """
            adapter = FlextUtilitiesTypingAdapterJson.json_value_adapter()
            return adapter.validate_python(yaml.safe_load(stream))

        @staticmethod
        def safe_dump(
            data: t.JsonValue | t.JsonMapping,
            *,
            sort_keys: bool = False,
            indent: int = 2,
            allow_unicode: bool = True,
            default_flow_style: bool = False,
        ) -> str:
            """Serialize a JSON value → YAML string through the libyaml emitter.

            The read path already parses through libyaml; the emit path uses
            the same native safe dumper instead of the pure-Python one.

            Returns:
                The resulting ``str``.

            """
            return yaml.dump(
                data,
                Dumper=yaml.CSafeDumper,
                sort_keys=sort_keys,
                indent=indent,
                allow_unicode=allow_unicode,
                default_flow_style=default_flow_style,
            )

        @staticmethod
        def safe_load_file(path: Path) -> t.JsonValue:
            """Load a YAML file → validated JSON value.

            Returns:
                The resulting ``t.JsonValue``.

            """
            adapter = FlextUtilitiesTypingAdapterJson.json_value_adapter()
            with path.open(encoding="utf-8") as fh:
                return adapter.validate_python(yaml.safe_load(fh))

        @staticmethod
        def unique_key_load(stream: str) -> t.JsonValue:
            """Public door to the settings sources' strict unique-key YAML loader.

            Returns:
                The parsed JSON value.

            """
            return FlextStrictYamlConfigSource.unique_key_load(stream)

        @staticmethod
        def unique_key_load_all(
            text: str,
        ) -> p.Result[tuple[t.JsonValue, ...]]:
            """Read every YAML document through the strict loader → ``r[...]``.

            The one-way read path: YAML 1.2 scalars, unique keys, libyaml
            parsing. Formatting-preserving writes keep their round-trip engine.

            Returns:
                Every document in stream order, or the first parse failure.

            """
            result = r[tuple[t.JsonValue, ...]]
            try:
                return result.ok(FlextStrictYamlConfigSource.unique_key_load_all(text))
            except yaml.YAMLError as exc:
                return result.fail(f"YAML parse error: {exc}", exception=exc)
            except ValidationError as exc:
                return result.fail(
                    f"YAML value is not JSON-compatible: {exc}", exception=exc
                )

        @staticmethod
        def yaml_safe_load(path: Path) -> p.Result[t.JsonMapping]:
            """Load a YAML file → ``r[JsonMapping]``.

            Returns:
                The resulting ``p.Result[t.JsonMapping]``.

            """
            if not path.is_file():
                return r[t.JsonMapping].fail(
                    f"YAML file not found: {path}",
                )
            try:
                loaded = FlextUtilitiesConfig.Yaml.safe_load_file(path)
            except yaml.YAMLError as exc:
                return r[t.JsonMapping].fail(
                    f"YAML parse error: {exc}",
                    exception=exc,
                )
            except OSError as exc:
                return r[t.JsonMapping].fail(
                    f"YAML read error: {exc}",
                    exception=exc,
                )
            except ValidationError as exc:
                return r[t.JsonMapping].fail(
                    f"YAML value is not JSON-compatible: {path}: {exc}",
                    exception=exc,
                )
            if not isinstance(loaded, Mapping):
                return r[t.JsonMapping].fail(
                    f"YAML top level is not a mapping: {path}",
                )
            return r[t.JsonMapping].ok(loaded)

        @staticmethod
        def yaml_dump(
            path: Path,
            data: t.JsonMapping,
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

        for _ in range(c.CONFIG_EXPAND_MAX_PASSES):
            expanded = FlextUtilitiesConfig._EXPAND_PATTERN.sub(_expand_match, current)
            if expanded == current:
                return expanded
            current = expanded
        return current

    @staticmethod
    def config_load(path: Path) -> p.Result[t.JsonMapping]:
        """Load and parse a TOML config source into a validated mapping.

        Fail-closed: a missing file, parse error, or non-mapping top level is a
        failed ``r[T]``, never a raised exception escaping ``config_load``.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        if not path.is_file():
            return r[t.JsonMapping].fail(
                f"{c.ERR_CONFIG_READ_FAILED}: {path}",
            )
        parsed = FlextUtilitiesReliability.try_(
            lambda: tomllib.loads(
                path.read_text(encoding=c.CONFIG_DEFAULT_ENCODING),
            ),
            catch=(OSError, tomllib.TOMLDecodeError),
            op_name="config_load",
        )
        if parsed.failure:
            return r[t.JsonMapping].from_failure(parsed)
        payload = parsed.value
        if not FlextUtilitiesGuardsTypeCore.mapping(payload):
            return r[t.JsonMapping].fail(
                f"{c.ERR_CONFIG_NOT_MAPPING}: {path}",
            )
        return r[t.JsonMapping].ok(payload)

    @staticmethod
    def config_merge(
        base: t.JsonMapping,
        override: t.JsonMapping,
    ) -> t.JsonDict:
        """Deep-merge ``override`` onto ``base``; ``override`` values replace lists.

        Returns:
            The merged mapping.

        """
        return FlextStrictYamlConfigSource.merge_values(
            base, override, concatenate_lists=False
        )

    @staticmethod
    def config_env_override(value: t.JsonValue, env: Mapping[str, ...]) -> t.JsonValue:
        """Expand ``${VAR}`` / ``${VAR:-default}`` placeholders in string leaves.

        Recurses through mappings and sequences; non-string leaves pass through
        unchanged. ``${VAR}`` resolves to ``env[VAR]`` or ``""`` when absent;
        ``${VAR:-default}`` resolves to ``env[VAR]`` or ``default`` when absent.

        Returns:
            The resulting ``t.JsonValue``.

        """
        if isinstance(value, str):
            return FlextUtilitiesConfig._expand_str(value, env)
        if isinstance(value, Mapping):
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
