"""JSON I/O service for reading, writing, parsing and serialising JSON.

Provides only canonical r-wrapped APIs (``read``, ``write``, ``parse``,
``serialize``) for explicit error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import override

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import r, s
from flext_infra import c, t


class FlextInfraJsonService(s[bool]):
    """Infrastructure service for JSON I/O.

    **r-wrapped API** (full error info)::

        result = svc.read(path)  # r[Mapping]
        result = svc.write(path, data)  # r[bool]
        result = svc.parse(text)  # r[t.ContainerValue]
        result = svc.serialize(data)  # r[str]
    """

    def __init__(self) -> None:
        """Initialise the JSON service."""
        super().__init__()

    @override
    def execute(self) -> r[bool]:
        """Execute the service (required by s base class)."""
        return r[bool].ok(True)

    def read(self, path: Path) -> r[Mapping[str, t.ContainerValue]]:
        """Read and parse a JSON **file**.

        Returns:
            r with parsed dict.  Empty mapping when file is absent.

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

    def write(
        self,
        path: Path,
        payload: t.ContainerValue,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> r[bool]:
        """Write a JSON payload to a **file** (r-wrapped).

        Creates parent directories as needed.
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

    def parse(self, text: str) -> r[t.ContainerValue]:
        """Parse a JSON **string** (r-wrapped).

        Args:
            text: Raw JSON string.

        """
        try:
            parsed: t.ContainerValue = json.loads(text)
            return r[t.ContainerValue].ok(parsed)
        except (json.JSONDecodeError, ValueError) as exc:
            return r[t.ContainerValue].fail(f"JSON parse error: {exc}")

    def serialize(
        self,
        data: t.ContainerValue,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int | None = 2,
    ) -> r[str]:
        """Serialise a Python object to a JSON **string** (r-wrapped).

        Accepts a JSON-serializable container value.
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


__all__ = ["FlextInfraJsonService"]
