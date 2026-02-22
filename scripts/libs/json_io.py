"""Unified JSON read/write utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from .config import DEFAULT_ENCODING


def write_json(
    path: Path,
    payload: object,
    *,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    """Write a JSON payload to *path*, creating parent dirs as needed.

    Args:
        path: Destination file path.
        payload: Data to serialize as JSON.
        sort_keys: If True, sort dictionary keys alphabetically.
        ensure_ascii: If True, escape non-ASCII characters.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
        + "\n",
        encoding=DEFAULT_ENCODING,
    )


def read_json(path: Path) -> dict[str, object]:
    """Read JSON from *path*.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON as dict. Returns empty dict if file doesn't exist.

    """
    if not path.exists():
        return {}
    return cast(
        "dict[str, object]", json.loads(path.read_text(encoding=DEFAULT_ENCODING))
    )


__all__ = ["read_json", "write_json"]
