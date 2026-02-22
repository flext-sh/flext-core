"""JSON I/O service for reading and writing JSON files.

Wraps JSON operations with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from flext_core.result import FlextResult, r

from flext_infra.constants import ic


class JsonService:
    """Infrastructure service for JSON file I/O.

    Provides FlextResult-wrapped JSON read/write operations, replacing
    the bare functions from ``scripts/libs/json_io.py``.
    """

    def read(self, path: Path) -> FlextResult[dict[str, object]]:
        """Read and parse a JSON file.

        Args:
            path: Source file path.

        Returns:
            FlextResult with parsed JSON data. Returns empty dict
            if the file does not exist.

        """
        if not path.exists():
            return r[dict[str, object]].ok({})
        try:
            data = cast(
                "dict[str, object]",
                json.loads(path.read_text(encoding=ic.Encoding.DEFAULT)),
            )
            return r[dict[str, object]].ok(data)
        except (json.JSONDecodeError, OSError) as exc:
            return r[dict[str, object]].fail(f"JSON read error: {exc}")

    def write(
        self,
        path: Path,
        payload: object,
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
            content = (
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=sort_keys,
                    ensure_ascii=ensure_ascii,
                )
                + "\n"
            )
            _ = path.write_text(content, encoding=ic.Encoding.DEFAULT)
            return r[bool].ok(True)
        except (TypeError, OSError) as exc:
            return r[bool].fail(f"JSON write error: {exc}")


__all__ = ["JsonService"]
