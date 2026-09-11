"""Round-trip YAML conversion helpers behind ``u.Cli.yaml_*``.

Plain<->commented conversions and ruamel scalar normalization. Composed into
``FlextCliUtilitiesYaml`` via MRO in ``yaml.py``.

NOTE (multi-agent): mro-i6nq.13 — extracted from the removed
``_yaml_roundtrip_parts/..._part_01`` (conversion half). The engine +
load/dump surface lives in ``_yaml/_engine.py``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import SupportsFloat, SupportsIndex, SupportsInt, TypeGuard, cast, overload

from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, LiteralScalarString

from flext_cli import t

#: YAML 1.1 boolean/null tokens that must be quoted to survive a round-trip.
#: Single consumer (``yaml_deep_to_commented``); promote to ``c.Cli`` if a
#: second consumer appears. NOTE (multi-agent): domain data, not a lint token.
_YAML_1_1_IMPLICIT_STRING_VALUES = frozenset({
    "y",
    "yes",
    "n",
    "no",
    "true",
    "false",
    "on",
    "off",
    "null",
    "~",
})


class FlextCliUtilitiesYamlConvertMixin:
    """Plain<->commented conversion and scalar normalization for YAML trees."""

    @staticmethod
    def yaml_to_plain(data: t.Cli.YamlNode) -> t.Cli.YamlValue:
        """Recursively convert ruamel containers into plain Python values."""
        if isinstance(data, dict):
            return {
                key: FlextCliUtilitiesYamlConvertMixin.yaml_to_plain(value)
                for key, value in data.items()
            }
        if isinstance(data, list):
            return [
                FlextCliUtilitiesYamlConvertMixin.yaml_to_plain(item) for item in data
            ]
        return data

    @overload
    @staticmethod
    def yaml_deep_to_commented(data: CommentedMap) -> CommentedMap: ...

    @overload
    @staticmethod
    def yaml_deep_to_commented(data: CommentedSeq) -> CommentedSeq: ...

    @overload
    @staticmethod
    def yaml_deep_to_commented(data: Mapping[str, t.Cli.YamlValue]) -> CommentedMap: ...

    @overload
    @staticmethod
    def yaml_deep_to_commented(data: list[t.Cli.YamlValue]) -> CommentedSeq: ...

    @overload
    @staticmethod
    def yaml_deep_to_commented(data: t.Cli.YamlScalar) -> t.Cli.YamlScalar: ...

    @staticmethod
    def yaml_deep_to_commented(data: t.Cli.YamlValue) -> t.Cli.YamlNode:
        """Recursively convert plain dict/list into CommentedMap/CommentedSeq.

        Existing CommentedMap/CommentedSeq nodes are preserved. Multi-line
        strings become LiteralScalarString; YAML 1.1 implicit string tokens are
        double-quoted so they survive a round-trip as strings.
        """
        if isinstance(data, CommentedMap | CommentedSeq):
            return data
        if isinstance(data, Mapping):
            node = CommentedMap()
            for key, value in data.items():
                node[key] = FlextCliUtilitiesYamlConvertMixin.yaml_deep_to_commented(
                    value
                )
            return node
        if FlextCliUtilitiesYamlConvertMixin.yaml_is_sequence(data):
            return CommentedSeq(
                FlextCliUtilitiesYamlConvertMixin.yaml_deep_to_commented(item)
                for item in data
            )
        if isinstance(data, str):
            if "\n" in data:
                return cast("t.Cli.YamlScalar", LiteralScalarString(data))
            if data.lower() in _YAML_1_1_IMPLICIT_STRING_VALUES:
                return cast("t.Cli.YamlScalar", DoubleQuotedScalarString(data))
        if data is not None and not isinstance(data, (str, int, float, bool)):
            msg = f"unsupported YAML value type: {type(data).__name__}"
            raise TypeError(msg)
        return data

    @staticmethod
    def yaml_is_sequence(value: t.Cli.YamlValue) -> TypeGuard[t.Cli.YamlSequence]:
        """Return True for YAML sequence nodes while keeping strings scalar.

        NOTE (multi-agent): deliberately excludes ``tuple`` (unlike the legacy
        charts helper) — ``t.Cli.YamlValue`` cannot type a tuple, so a runtime
        tuple now fails loud in ``yaml_deep_to_commented`` instead of being
        silently treated as a sequence.
        """
        return isinstance(value, (CommentedSeq, list))

    @staticmethod
    def yaml_normalize_scalar(value: t.Cli.YamlValue) -> t.Cli.YamlValue:
        """Normalize ruamel scalar wrappers to plain Python scalars."""
        if isinstance(value, str):
            return FlextCliUtilitiesYamlConvertMixin.yaml_plain_str(value)
        if isinstance(value, bool):
            return FlextCliUtilitiesYamlConvertMixin.yaml_plain_bool(value)
        if isinstance(value, int):
            return FlextCliUtilitiesYamlConvertMixin.yaml_plain_int(value)
        if isinstance(value, float):
            return FlextCliUtilitiesYamlConvertMixin.yaml_plain_float(value)
        return value

    @staticmethod
    def yaml_plain_str(value: t.Cli.YamlScalar) -> str:
        """Return *value* as a plain builtin str (unwrap ruamel subclasses)."""
        return value if type(value) is str else str(value)

    @staticmethod
    def yaml_plain_bool(value: t.Cli.YamlScalar) -> bool:
        """Return *value* as a plain builtin bool (unwrap ruamel subclasses)."""
        return value if type(value) is bool else bool(value)

    @staticmethod
    def yaml_plain_int(value: SupportsInt | SupportsIndex) -> int:
        """Return *value* as a plain builtin int (unwrap ruamel subclasses).

        NOTE (multi-agent): ``SupportsInt | SupportsIndex`` is the real domain
        contract — only ``int`` and ruamel int subclasses (which implement both
        protocols) arrive here. Never widen to ``object`` (hides str/object and
        breaks the 4-checker gate).
        """
        return value if type(value) is int else int(value)

    @staticmethod
    def yaml_plain_float(value: SupportsFloat | SupportsIndex) -> float:
        """Return *value* as a plain builtin float (unwrap ruamel subclasses).

        NOTE (multi-agent): same contract as ``yaml_plain_int`` — keep the
        ``SupportsFloat | SupportsIndex`` union; never widen to ``object``.
        """
        return value if type(value) is float else float(value)

    @staticmethod
    def _yaml_coerce_node(value: t.Cli.YamlValue) -> t.Cli.YamlNode:
        """Validate a parsed ruamel root against the supported node contract."""
        if isinstance(value, (CommentedMap, CommentedSeq, str, int, float, bool)):
            return value
        if value is None:
            return None
        if isinstance(
            value, Mapping
        ) or FlextCliUtilitiesYamlConvertMixin.yaml_is_sequence(value):
            return FlextCliUtilitiesYamlConvertMixin.yaml_deep_to_commented(value)
        msg = f"unsupported YAML root type: {type(value).__name__}"
        raise TypeError(msg)


__all__: list[str] = ["FlextCliUtilitiesYamlConvertMixin"]
