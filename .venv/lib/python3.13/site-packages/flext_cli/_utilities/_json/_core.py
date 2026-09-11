"""Core JSON serialization, file I/O, and normalization behind ``u.Cli.json_*``.

Composed into ``FlextCliUtilitiesJson`` via MRO in ``json.py``. All methods use
the ``json_`` prefix and canonical ``t.Cli.JSON_*_ADAPTER`` Pydantic adapters.

NOTE (multi-agent): mro-i6nq.13 — merged the removed numbered
``_json_parts`` serialization (part_03) and file-I/O (part_01) halves into one
cohesive core module. Navigation/extraction helpers live in
``_json/_navigate.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from flext_cli import c, m, p, r, t
from flext_core import u

_EMPTY_JSON_MAPPING: t.JsonMapping = MappingProxyType({})
_EMPTY_JSON_SEQUENCE: t.SequenceOf[t.JsonValue] = ()


class FlextCliUtilitiesJsonCoreMixin:
    """Core JSON serialization, file read/write, and value normalization."""

    _module_logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    @staticmethod
    def json_dumps(
        value: t.JsonValue, *, sort_keys: bool = False, indent: int | None = None
    ) -> p.Result[str]:
        """Serialize a JSON-compatible value to a string via canonical adapters."""
        normalized = (
            FlextCliUtilitiesJsonCoreMixin.json_sort_keys(value) if sort_keys else value
        )
        return u.try_(
            lambda: t.Cli.JSON_VALUE_ADAPTER.dump_json(
                normalized, indent=indent
            ).decode(c.Cli.ENCODING_DEFAULT),
            catch=(c.ValidationError, ValueError, TypeError),
            op_name="json_dumps",
        )

    @staticmethod
    def json_loads(raw: str | bytes) -> p.Result[t.JsonValue]:
        """Parse a JSON-encoded string/bytes into a JSON-compatible value."""
        return u.try_(
            lambda: t.Cli.JSON_VALUE_ADAPTER.validate_json(raw),
            catch=(c.ValidationError, ValueError),
            op_name="json_loads",
        )

    @staticmethod
    def json_sort_keys(data: t.JsonValue) -> t.JsonValue:
        """Recursively sort dictionary keys in a JSON structure."""
        if isinstance(data, Mapping):
            validated = t.Cli.JSON_MAPPING_ADAPTER.validate_python(data)
            return {
                key: FlextCliUtilitiesJsonCoreMixin.json_sort_keys(
                    t.Cli.JSON_VALUE_ADAPTER.validate_python(value)
                )
                for key, value in sorted(validated.items())
            }
        if isinstance(data, list):
            items = t.Cli.JSON_LIST_ADAPTER.validate_python(data)
            return [
                FlextCliUtilitiesJsonCoreMixin.json_sort_keys(
                    t.Cli.JSON_VALUE_ADAPTER.validate_python(item)
                )
                for item in items
            ]
        return data

    @staticmethod
    def normalize_json_value(item: t.JsonPayload) -> t.JsonValue:
        """Normalize any runtime value to JSON-compatible output (Pydantic-native)."""
        return u.normalize_to_json_value(item)

    @staticmethod
    def _json_write_content(
        payload: t.JsonPayload, options: m.Cli.JsonWriteOptions
    ) -> str:
        """Serialize a JSON payload using canonical write options."""
        validated = FlextCliUtilitiesJsonCoreMixin.normalize_json_value(payload)
        normalized = (
            FlextCliUtilitiesJsonCoreMixin.json_sort_keys(validated)
            if options.sort_keys
            else validated
        )
        payload_bytes: bytes = t.Cli.JSON_VALUE_ADAPTER.dump_json(
            normalized, indent=options.indent, ensure_ascii=options.ensure_ascii
        )
        return payload_bytes.decode(c.Cli.ENCODING_DEFAULT) + "\n"

    @staticmethod
    def json_read(path: Path) -> p.Result[t.JsonMapping]:
        """Read and parse a JSON file.

        Missing files fail loud; callers decide whether absence is valid.
        """
        if not path.exists():
            return r[t.JsonMapping].fail(f"json_read: file not found: {path}")
        loaded = u.try_(
            lambda: t.Cli.JSON_VALUE_ADAPTER.validate_json(
                path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            ),
            catch=(c.ValidationError, OSError),
            op_name="json_read",
        )
        if loaded.failure:
            return r[t.JsonMapping].fail(loaded.error or "json_read failed")
        if not isinstance(loaded.value, Mapping):
            return r[t.JsonMapping].fail("json_read: root must be an object")
        return r[t.JsonMapping].ok(
            t.Cli.JSON_MAPPING_ADAPTER.validate_python(loaded.value)
        )

    @staticmethod
    def json_write(
        path: Path,
        payload: t.JsonPayload,
        options: m.Cli.JsonWriteOptions | None = None,
    ) -> p.Result[bool]:
        """Write any Pydantic-serializable payload to a JSON file."""
        opts = options or m.Cli.JsonWriteOptions()

        def _write() -> bool:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = FlextCliUtilitiesJsonCoreMixin._json_write_content(payload, opts)
            _ = path.write_text(content, encoding=c.Cli.ENCODING_DEFAULT)
            return True

        written = u.try_(_write, catch=c.EXC_OS_VALIDATION, op_name="json_write")
        if written.failure:
            FlextCliUtilitiesJsonCoreMixin._module_logger.debug(
                "json_write failed", error=written.error, exc_info=False
            )
        return written

    @staticmethod
    def json_parse(text: str) -> p.Result[t.JsonValue]:
        """Parse a JSON string into a validated JsonValue."""
        return u.try_(
            lambda: t.Cli.JSON_VALUE_ADAPTER.validate_json(text),
            catch=c.EXC_VALIDATION_VALUE,
            op_name="json_parse",
        )

    @staticmethod
    def json_as_mapping(value: t.JsonPayload | None) -> t.JsonMapping:
        """Normalize any JSON-compatible value into a mapping."""
        if value is None:
            return _EMPTY_JSON_MAPPING
        normalized: t.JsonValue = u.normalize_to_json_value(value)
        if not isinstance(normalized, Mapping):
            return _EMPTY_JSON_MAPPING
        return t.Cli.JSON_MAPPING_ADAPTER.validate_python(normalized)

    @staticmethod
    def json_as_sequence(value: t.JsonPayload | None) -> t.SequenceOf[t.JsonValue]:
        """Normalize any JSON-compatible value into a JSON sequence."""
        if value is None:
            return _EMPTY_JSON_SEQUENCE
        normalized: t.JsonValue = u.normalize_to_json_value(value)
        if not isinstance(normalized, Sequence) or isinstance(normalized, str | bytes):
            return _EMPTY_JSON_SEQUENCE
        return t.Cli.JSON_LIST_ADAPTER.validate_python(normalized)


__all__: list[str] = ["FlextCliUtilitiesJsonCoreMixin"]
