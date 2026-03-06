"""JSON I/O service for reading, writing, parsing and serialising JSON.

Provides two APIs:
  * **r-wrapped** (``read``, ``write``, ``parse``, ``serialize``) for callers
    that need full error control.
  * **Direct-return** (``load``, ``dump``, ``loads``, ``dumps``, ``is_json``)
    for zero-boilerplate call-sites that accept sensible defaults on error.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import overload, override

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import r, s
from flext_infra import c, t


class FlextInfraJsonService(s[bool]):
    """Infrastructure service for JSON I/O.

    **r-wrapped API** (full error info)::

        result = svc.read(path)  # r[Mapping]
        result = svc.write(path, data)  # r[bool]
        result = svc.parse(text)  # r[object]
        result = svc.serialize(data)  # r[str]

    **Direct-return API** (zero boilerplate)::

        data = svc.load(path)  # dict | None
        ok = svc.dump(path, data)  # bool
        value = svc.loads(text)  # object | None
        value = svc.loads(text, {})  # object (fallback)
        text = svc.dumps(data)  # str
        valid = svc.is_json(text)  # bool
    """

    def __init__(self) -> None:
        """Initialise the JSON service."""
        super().__init__()

    @override
    def execute(self) -> r[bool]:
        """Execute the service (required by s base class)."""
        return r[bool].ok(True)

    # ------------------------------------------------------------------
    #  r-wrapped API
    # ------------------------------------------------------------------

    def read(self, path: Path) -> r[Mapping[str, t.ContainerValue]]:
        """Read and parse a JSON **file**.

        Returns:
            r with parsed dict.  Empty mapping when file is absent.

        """
        if not path.exists():
            return r[t.ConfigurationMapping].ok({})
        try:
            loaded_obj: t.ContainerValue = json.loads(
                path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            )
            if not isinstance(loaded_obj, dict):
                return r[t.ConfigurationMapping].fail(
                    "JSON root must be object",
                )
            parser = TypeAdapter(dict[str, t.ContainerValue])
            data = parser.validate_python(loaded_obj, strict=True)
            return r[t.ConfigurationMapping].ok(data)
        except ValidationError as exc:
            return r[t.ConfigurationMapping].fail(
                f"JSON object validation error: {exc}"
            )
        except (json.JSONDecodeError, OSError) as exc:
            return r[t.ConfigurationMapping].fail(f"JSON read error: {exc}")

    def write(
        self,
        path: Path,
        payload: object,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> r[bool]:
        """Write a JSON payload to a **file** (r-wrapped).

        Creates parent directories as needed.  Accepts any JSON-serialisable
        object: ``dict``, ``list``, ``BaseModel``, ``Mapping``, etc.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = payload.model_dump() if isinstance(payload, BaseModel) else payload
            content = (
                json.dumps(
                    data,
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

    def parse(
        self,
        text: str,
        *,
        fallback: object | None = None,
    ) -> r[object]:
        """Parse a JSON **string** (r-wrapped).

        Args:
            text: Raw JSON string.
            fallback: If supplied and *text* is empty/blank, this value is
                returned as a success instead of attempting to parse.

        """
        stripped = text.strip()
        if not stripped and fallback is not None:
            return r[object].ok(fallback)
        try:
            return r[object].ok(json.loads(stripped or text))
        except (json.JSONDecodeError, ValueError) as exc:
            return r[object].fail(f"JSON parse error: {exc}")

    def serialize(
        self,
        data: object,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int | None = 2,
    ) -> r[str]:
        """Serialise a Python object to a JSON **string** (r-wrapped).

        Accepts ``BaseModel``, ``dict``, ``list``, or any JSON-serialisable
        value.
        """
        try:
            raw = data.model_dump() if isinstance(data, BaseModel) else data
            return r[str].ok(
                json.dumps(
                    raw,
                    indent=indent,
                    sort_keys=sort_keys,
                    ensure_ascii=ensure_ascii,
                )
            )
        except (TypeError, ValueError) as exc:
            return r[str].fail(f"JSON serialize error: {exc}")

    # ------------------------------------------------------------------
    #  Direct-return API  (zero boilerplate)
    # ------------------------------------------------------------------

    def load(self, path: Path) -> dict[str, object] | None:
        """Read a JSON file → ``dict`` or ``None`` on any error."""
        result = self.read(path)
        if result.is_failure:
            return None
        raw = result.value
        return dict(raw) if raw is not None else None

    def dump(
        self,
        path: Path,
        payload: object,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> bool:
        """Write JSON to *path* — fire-and-forget.  Returns ``True`` on success."""
        return self.write(
            path,
            payload,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            indent=indent,
        ).is_success

    @overload
    def loads(self, text: str) -> object | None: ...
    @overload
    def loads(self, text: str, fallback: dict[str, object]) -> dict[str, object]: ...
    @overload
    def loads(self, text: str, fallback: list[object]) -> list[object]: ...

    def loads(
        self,
        text: str,
        fallback: object | None = None,
    ) -> object | None:
        """Parse a JSON string → value directly, or *fallback* on error.

        Without *fallback*: returns ``None`` on error.
        With *fallback*: returns *fallback* on error (never ``None``).
        """
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return fallback

    def dumps(
        self,
        data: object,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int | None = 2,
    ) -> str:
        """Serialise to JSON string directly.  Returns ``""`` on error."""
        result = self.serialize(
            data,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            indent=indent,
        )
        return result.value if result.is_success else ""

    def is_json(self, text: str) -> bool:
        """Return ``True`` if *text* is valid JSON."""
        try:
            json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return False
        return True


# Module-level convenience functions — delegate to u.Infra.Io namespace


def read_json(path: Path) -> r[Mapping[str, t.ContainerValue]]:
    """Read and parse a JSON file (convenience function)."""
    return FlextInfraJsonService().read(path)


def write_json(
    path: Path,
    payload: object,
    *,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> r[bool]:
    """Write a JSON payload to a file (convenience function)."""
    return FlextInfraJsonService().write(
        path, payload, sort_keys=sort_keys, ensure_ascii=ensure_ascii
    )


__all__ = ["FlextInfraJsonService", "read_json", "write_json"]
