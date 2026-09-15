"""Atomic file primitives for fleet-wide consumption (R4 gates-as-products).

Generic domain utilities: atomic O_APPEND append plus atomic full-write
(requested in WS-F4); failures return ``r.Fail`` at the ``u`` boundary and
success carries the typed byte-count payload (results never succeed with
``None``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

from flext_core import p
from ..result import FlextResult as r


class FlextUtilitiesFiles:
    """Atomic file primitives owned once by flext-core ``u``."""

    @staticmethod
    def append_atomic(
        path: Path, data: str, *, encoding: str = "utf-8"
    ) -> p.ResultView[int]:
        """Atomically append text to a file through ``O_APPEND``.

        The append flag keeps concurrent writers line-atomic; creation is
        implicit for a first write. Failures escape as ``r.Fail``.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError as exc:
            return cast(p.ResultView[int], r.fail(f"atomic append open failed: {exc}", exception=exc))
        try:
            written = os.write(descriptor, data.encode(encoding))
        except OSError as exc:
            return cast(p.ResultView[int], r.fail(f"atomic append write failed: {exc}", exception=exc))
        finally:
            os.close(descriptor)
        return cast(p.ResultView[int], r[int].ok(written))

    @staticmethod
    def write_atomic(
        path: Path, data: str, *, encoding: str = "utf-8"
    ) -> p.ResultView[int]:
        """Atomically replace a file's contents via temp file + rename."""
        payload = data.encode(encoding)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(payload)
                staged = Path(tmp.name)
        except OSError as exc:
            return cast(p.ResultView[int], r.fail(f"atomic write stage failed: {exc}", exception=exc))
        try:
            staged.replace(path)
        except OSError as exc:
            staged.unlink(missing_ok=True)
            return cast(p.ResultView[int], r.fail(f"atomic write rename failed: {exc}", exception=exc))
        return cast(p.ResultView[int], r[int].ok(len(payload)))


__all__: list[str] = ["FlextUtilitiesFiles"]
