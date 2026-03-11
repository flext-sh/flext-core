"""I/O helper functions for infrastructure file operations.

Centralizes JSON I/O operations with r-wrapped APIs for explicit error handling.
Merged from json_io.py and _utilities/io.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, t


class FlextInfraUtilitiesIo:
    """I/O convenience helpers for JSON file operations.

    Provides r-wrapped APIs (read, write, parse, serialize) for explicit
    error handling. All methods are static and do not require instantiation.

    Usage via namespace::

        from flext_infra import u

        result = u.Infra.read_json(path)
    """

    @staticmethod
    def read_json(path: Path) -> r[Mapping[str, t.ContainerValue]]:
        """Read and parse a JSON file.

        Args:
            path: Source file path.

        Returns:
            r with parsed JSON data. Returns empty mapping if file absent.

        """
        if not path.exists():
            return r[t.ConfigurationMapping].ok({})
        try:
            loaded_obj: t.ContainerValue = json.loads(
                path.read_text(encoding=c.Infra.Encoding.DEFAULT),
            )
            if not isinstance(loaded_obj, dict):
                return r[t.ConfigurationMapping].fail("JSON root must be object")
            parser = TypeAdapter(dict[str, t.ContainerValue])
            data = parser.validate_python(loaded_obj, strict=True)
            return r[t.ConfigurationMapping].ok(data)
        except ValidationError as exc:
            return r[t.ConfigurationMapping].fail(
                f"JSON object validation error: {exc}",
            )
        except (json.JSONDecodeError, OSError) as exc:
            return r[t.ConfigurationMapping].fail(f"JSON read error: {exc}")

    @staticmethod
    def write_json(
        path: Path,
        payload: t.ContainerValue,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> r[bool]:
        """Write a JSON payload to a file.

        Creates parent directories as needed.

        Args:
            path: Destination file path.
            payload: Data to serialize as JSON.
            sort_keys: If True, sort dictionary keys alphabetically.
            ensure_ascii: If True, escape non-ASCII characters.
            indent: JSON indentation level (default 2).

        Returns:
            r[bool] with True on success.

        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw_payload: t.ContainerValue = (
                payload.model_dump() if isinstance(payload, BaseModel) else payload
            )
            content = (
                json.dumps(
                    raw_payload,
                    indent=indent,
                    sort_keys=sort_keys,
                    ensure_ascii=ensure_ascii,
                )
                + "\n"
            )
            _ = path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
        except (TypeError, OSError) as exc:
            return r[bool].fail(f"JSON write error: {exc}")
        return r[bool].ok(True)

    @staticmethod
    def parse(text: str) -> r[t.ContainerValue]:
        """Parse a JSON string.

        Args:
            text: Raw JSON string.

        Returns:
            r with parsed JSON value.

        """
        try:
            parsed: t.ContainerValue = json.loads(text)
            return r[t.ContainerValue].ok(parsed)
        except (json.JSONDecodeError, ValueError) as exc:
            return r[t.ContainerValue].fail(f"JSON parse error: {exc}")

    @staticmethod
    def serialize(
        data: t.ContainerValue,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int | None = 2,
    ) -> r[str]:
        """Serialize a Python object to a JSON string.

        Args:
            data: JSON-serializable container value.
            sort_keys: If True, sort dictionary keys alphabetically.
            ensure_ascii: If True, escape non-ASCII characters.
            indent: JSON indentation level (None for compact output).

        Returns:
            r with JSON string.

        """
        try:
            raw_data: t.ContainerValue = (
                data.model_dump() if isinstance(data, BaseModel) else data
            )
            return r[str].ok(
                json.dumps(
                    raw_data,
                    indent=indent,
                    sort_keys=sort_keys,
                    ensure_ascii=ensure_ascii,
                ),
            )
        except (TypeError, ValueError) as exc:
            return r[str].fail(f"JSON serialize error: {exc}")


__all__ = ["FlextInfraUtilitiesIo"]
