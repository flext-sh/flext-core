"""Fileno-backed Git stdin boundary."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from typing import BinaryIO


@contextmanager
def git_stdin(data: bytes | None) -> Generator[BinaryIO | None]:
    """Yield a seekable stream accepted by GitPython's ``istream`` boundary."""
    if data is None:
        yield None
        return
    with tempfile.TemporaryFile() as stream:
        stream.write(data)
        stream.seek(0)
        yield stream


__all__: list[str] = ["git_stdin"]
