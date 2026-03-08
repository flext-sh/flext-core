r"""FlextUtilitiesFileOps - Side-effect file/IO operations returning None.

Wraps stdlib Path.write_text / write_bytes / sys.stdout.write with signatures
that discard the byte-count return value, eliminating reportUnusedCallResult
warnings across the entire workspace without resorting to ``_ =`` suppression.

Usage (via facade):
    from flext_core import u
    u.write_file(path, content, encoding=c.Infra.Encoding.DEFAULT)
    u.write_file(path, raw_bytes)
    u.write_stdout("message\\\\n")

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import overload


class FlextUtilitiesFileOps:
    """Side-effect file/IO write operations that return None."""

    @staticmethod
    @overload
    def write_file(path: Path, content: str, *, encoding: str = ...) -> None: ...

    @staticmethod
    @overload
    def write_file(path: Path, content: bytes) -> None: ...

    @staticmethod
    def write_file(
        path: Path, content: str | bytes, *, encoding: str = "utf-8"
    ) -> None:
        """Write content to *path*, discarding the byte-count return value.

        Overloaded for ``str`` (text mode) and ``bytes`` (binary mode).
        Replaces direct ``path.write_text(...)`` / ``path.write_bytes(...)``
        calls where the ``int`` return value is intentionally unused.
        """
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding=encoding)

    @staticmethod
    def write_stdout(text: str) -> None:
        """Write *text* to stdout and flush, discarding the byte-count.

        Replaces ``sys.stdout.write(...)`` where the ``int`` return is unused.
        """
        sys.stdout.write(text)
        sys.stdout.flush()


__all__ = ["FlextUtilitiesFileOps"]
