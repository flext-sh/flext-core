"""Generic YAML helpers shared through ``u.Cli.yaml_*``.

Follows the same pattern as ``toml.py`` — generic operations that any
project can reuse, prefixed with ``yaml_`` for namespace clarity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from yaml import safe_dump, safe_load

from flext_cli import c, p, r, t
from flext_cli._utilities._yaml._editing import FlextCliUtilitiesYamlEditingMixin
from flext_cli._utilities.json import FlextCliUtilitiesJson
from flext_core import u

_EMPTY_JSON_MAPPING: t.JsonMapping = MappingProxyType({})
_EMPTY_JSON_SEQUENCE: t.SequenceOf[t.JsonValue] = ()


class FlextCliUtilitiesYaml(FlextCliUtilitiesYamlEditingMixin):
    """Generic YAML read, parse, dump, and validation helpers.

    All YAML operations across the workspace delegate here.
    Projects needing domain-specific normalization wrap these methods.

    NOTE (multi-agent): mro-i6nq.13 — round-trip (comment/quote-preserving)
    operations are composed from the _yaml/{_editing,_engine,_convert} mixin
    chain (replacing the numbered _yaml_roundtrip_parts); the one-way PyYAML
    helpers in this class body stay for plain read/write. Do not add a
    second ruamel engine in this class or in any leaf module.
    """

    _module_logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def yaml_safe_load(path: Path) -> p.Result[t.JsonMapping]:
        """Load a YAML file → ``r[JsonMapping]``.

        Returns ``r.ok(mapping)`` on success, ``r.fail(msg)`` on missing,
        parse error, or non-mapping content.

        Example::

            data = u.Cli.yaml_safe_load(path).unwrap_or({})
        """
        if not path.is_file():
            return r[t.JsonMapping].fail(f"YAML file not found: {path}")
        try:
            raw = path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        except OSError as exc:
            return r[t.JsonMapping].fail(f"YAML read error: {exc}")
        return FlextCliUtilitiesYaml.yaml_parse(raw)

    @staticmethod
    def yaml_parse(text: str) -> p.Result[t.JsonMapping]:
        """Parse a YAML string → ``r[JsonMapping]``.

        Returns a validated mapping or failure.
        """
        # NOTE (multi-agent): the canonical ruamel engine rejects duplicate keys;
        # PyYAML safe_load was last-wins and could conceal contradictory config.
        loaded = FlextCliUtilitiesYaml.yaml_roundtrip_load_map_text(text)
        if loaded.failure:
            return r[t.JsonMapping].fail(
                loaded.error or "YAML parse error", exception=loaded.exception
            )
        parsed = FlextCliUtilitiesYaml.yaml_to_plain(loaded.value)
        try:
            validated = t.Cli.YAML_DICT_ADAPTER.validate_python(parsed)
        except c.ValidationError as exc:
            return r[t.JsonMapping].fail(f"YAML validation error: {exc}")
        return r[t.JsonMapping].ok(validated)

    @staticmethod
    def yaml_load_mapping(
        path: Path, *, default: t.JsonMapping | None = None
    ) -> t.JsonMapping:
        """Load YAML file returning a mapping, or *default* (empty dict) on any error.

        Ergonomic shorthand — use ``yaml_safe_load`` when you need ``r[T]`` semantics.
        """
        return FlextCliUtilitiesYaml.yaml_safe_load(path).unwrap_or(
            default if default is not None else _EMPTY_JSON_MAPPING
        )

    @staticmethod
    def _yaml_parse_list(path: Path) -> t.SequenceOf[t.JsonValue]:
        """Parse *path* as a top-level YAML list; raises on any failure."""
        raw = path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        parsed = safe_load(raw)
        if not isinstance(parsed, list):
            msg = f"YAML content is not a list: {type(parsed).__name__}"
            raise TypeError(msg)
        validated: t.SequenceOf[t.JsonValue] = t.Cli.YAML_SEQ_ADAPTER.validate_python(
            parsed
        )
        return validated

    @staticmethod
    def yaml_load_list(path: Path) -> t.SequenceOf[t.JsonValue]:
        """Load YAML file expecting a list at top level.

        Returns an empty list on missing file, parse error, non-list content,
        or validation failure — the failure itself is propagated through
        ``u.try_`` at the boundary rather than swallowed inline.
        """
        if not path.is_file():
            return _EMPTY_JSON_SEQUENCE
        return u.try_(
            lambda: FlextCliUtilitiesYaml._yaml_parse_list(path),
            catch=(OSError, c.Cli.YamlParseError, TypeError, c.ValidationError),
            op_name="yaml_load_list",
        ).unwrap_or(_EMPTY_JSON_SEQUENCE)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @staticmethod
    def yaml_dump(
        path: Path, data: t.JsonPayload, *, sort_keys: bool = False, indent: int = 2
    ) -> p.Result[bool]:
        """Write *data* to a YAML file → ``r[bool]``.

        Creates parent directories if needed.

        Example::

            u.Cli.yaml_dump(path, {"key": "val"})
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            validated = FlextCliUtilitiesJson.normalize_json_value(data)
            with path.open("w", encoding=c.Cli.ENCODING_DEFAULT) as fh:
                safe_dump(
                    validated,
                    fh,
                    default_flow_style=False,
                    sort_keys=sort_keys,
                    allow_unicode=True,
                    indent=indent,
                )
            return r[bool].ok(True)
        except (OSError, c.Cli.YamlParseError, ValueError, TypeError) as exc:
            return r[bool].fail(f"YAML write error: {exc}")

    @staticmethod
    def yaml_dump_str(
        data: t.JsonPayload, *, sort_keys: bool = False, indent: int = 2
    ) -> str:
        """Serialize *data* to a YAML string.

        Returns empty string on serialization failure.

        Example::

            text = u.Cli.yaml_dump_str(payload)
        """
        try:
            validated = FlextCliUtilitiesJson.normalize_json_value(data)
            serialized: str = safe_dump(
                validated,
                default_flow_style=False,
                sort_keys=sort_keys,
                allow_unicode=True,
                indent=indent,
            )
        except (c.Cli.YamlParseError, ValueError, TypeError):
            return ""
        else:
            return serialized


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesYaml"]
