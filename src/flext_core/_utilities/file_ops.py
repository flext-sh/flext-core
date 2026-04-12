r"""FlextUtilitiesFileOps - Side-effect file/IO operations returning None.

Wraps stdlib Path.write_text / write_bytes / sys.stdout.write with signatures
that discard the byte-count return value, eliminating reportUnusedCallResult
warnings across the entire workspace without resorting to ``_ =`` suppression.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import c


class FlextUtilitiesFileOps:
    """Side-effect file/IO write operations that return None."""

    @staticmethod
    def write_file(
        path: Path,
        content: str,
        *,
        encoding: str = c.DEFAULT_ENCODING,
    ) -> None:
        """Write content to *path*, discarding the byte-count return value.

        Replaces direct ``path.write_text(...)`` calls where the ``int``
        return value is intentionally unused.
        """
        _ = path.write_text(content, encoding=encoding)


__all__: list[str] = ["FlextUtilitiesFileOps"]
