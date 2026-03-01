"""JSON I/O service for reading and writing JSON files.

Wraps JSON operations with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from flext_core import FlextResult, FlextService, r, t
from pydantic import BaseModel

from flext_infra import c


class FlextInfraJsonService(FlextService[bool]):
    """Infrastructure service for JSON file I/O.

    Provides FlextResult-wrapped JSON read/write operations, replacing
    the bare functions from ``scripts/libs/json_io.py``.
    """

    def __init__(self) -> None:
        """Initialize the JSON service."""
        super().__init__()

    def read(self, path: Path) -> FlextResult[Mapping[str, t.ConfigMapValue]]:
        """Read and parse a JSON file.

        Args:
            path: Source file path.

        Returns:
            FlextResult with parsed JSON data. Returns empty mapping
            if the file does not exist.

        """
        if not path.exists():
            return r[Mapping[str, t.ConfigMapValue]].ok({})
        try:
            loaded = json.loads(path.read_text(encoding=c.Encoding.DEFAULT))
            if not isinstance(loaded, dict):
                return r[Mapping[str, t.ConfigMapValue]].fail(
                    "JSON root must be object",
                )
            data: Mapping[str, t.ConfigMapValue] = loaded
            return r[Mapping[str, t.ConfigMapValue]].ok(data)
        except (json.JSONDecodeError, OSError) as exc:
            return r[Mapping[str, t.ConfigMapValue]].fail(f"JSON read error: {exc}")

    def write(
        self,
        path: Path,
        payload: BaseModel | Mapping[str, t.ConfigMapValue],
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
    ) -> FlextResult[bool]:
        """Write a JSON payload to a file.

        Creates parent directories as needed.

        Args:
            path: Destination file path.
            payload: Data to serialize as JSON.
            sort_keys: If True, sort dictionary keys alphabetically.
            ensure_ascii: If True, escape non-ASCII characters.

        Returns:
            FlextResult[bool] with True on success.

        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = payload.model_dump() if isinstance(payload, BaseModel) else payload
            content = (
                json.dumps(
                    data,
                    indent=2,
                    sort_keys=sort_keys,
                    ensure_ascii=ensure_ascii,
                )
                + "\n"
            )
            _ = path.write_text(content, encoding=c.Encoding.DEFAULT)
        except (TypeError, OSError) as exc:
            return r[bool].fail(f"JSON write error: {exc}")
        return r[bool].ok(True)

    def execute(self) -> FlextResult[bool]:
        """Execute the service (required by FlextService base class)."""
        return r[bool].ok(True)


__all__ = ["FlextInfraJsonService"]
