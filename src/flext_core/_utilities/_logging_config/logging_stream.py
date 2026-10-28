"""Native text stream delegation for the logging output owner.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from types import TracebackType
from typing import TextIO, override

from flext_core import t
from flext_core._utilities import FlextUtilitiesTypingAdapterScalars


class FlextUtilitiesLoggingStream:
    """Own stream operations independently from queued writer lifecycle."""

    class StreamMetadata(TextIO):
        """Own the native stream identity and metadata without fabrication."""

        def __init__(self, stream: t.TextStream) -> None:
            self.stream = stream
            self._use_live_stdout = stream is sys.stdout
            self._use_live_stderr = stream is sys.stderr
            self._closed = False

        @property
        def _target_stream(self) -> t.TextStream:
            if self._use_live_stdout:
                return sys.stdout
            if self._use_live_stderr:
                return sys.stderr
            return self.stream

        @property
        @override
        def mode(self) -> str:
            return self._target_stream.mode

        @property
        @override
        def name(self) -> str | int:
            name = self._target_stream.name
            if isinstance(name, str | int):
                return name
            msg = "Text stream name must be a path or file descriptor"
            raise TypeError(msg)

        @property
        @override
        def buffer(self) -> t.BinaryStream:
            return self._target_stream.buffer

        @property
        @override
        def line_buffering(self) -> int:
            return self._target_stream.line_buffering

        @property
        @override
        def encoding(self) -> str:
            return self._target_stream.encoding

        @property
        @override
        def errors(self) -> str | None:
            return self._target_stream.errors

        @property
        @override
        def newlines(self) -> str | tuple[str, ...] | None:
            newlines = self._target_stream.newlines
            if newlines is None or isinstance(newlines, str):
                return newlines
            if isinstance(newlines, tuple):
                scalars = FlextUtilitiesTypingAdapterScalars
                return scalars.strict_str_tuple_adapter().validate_python(newlines)
            msg = "Text stream newlines must be a string or tuple of strings"
            raise TypeError(msg)

        @property
        @override
        def closed(self) -> bool:
            return self._closed

    class TextStream(StreamMetadata):
        """Delegate stream operations independently from their metadata."""

        @override
        def close(self) -> None:
            if not self.closed:
                self.flush()
                self._closed = True

        @override
        def fileno(self) -> int:
            self._check_open()
            return self._target_stream.fileno()

        @override
        def isatty(self) -> bool:
            self._check_open()
            return self._target_stream.isatty()

        @override
        def readable(self) -> bool:
            self._check_open()
            return self._target_stream.readable()

        @override
        def read(self, n: int = -1, /) -> str:
            self.flush()
            return self._target_stream.read(n)

        @override
        def readline(self, limit: int = -1, /) -> str:
            self.flush()
            return self._target_stream.readline(limit)

        @override
        def readlines(self, hint: int = -1, /) -> list[str]:
            self.flush()
            return self._target_stream.readlines(hint)

        @override
        def seek(self, offset: int, whence: int = 0, /) -> int:
            self.flush()
            return self._target_stream.seek(offset, whence)

        @override
        def seekable(self) -> bool:
            self._check_open()
            return self._target_stream.seekable()

        @override
        def tell(self) -> int:
            self.flush()
            return self._target_stream.tell()

        @override
        def truncate(self, size: int | None = None, /) -> int:
            self.flush()
            return self._target_stream.truncate(size)

        @override
        def writable(self) -> bool:
            self._check_open()
            return self._target_stream.writable()

        @override
        def write(self, s: str, /) -> int:
            self._check_open()
            return self._target_stream.write(s)

        @override
        def flush(self) -> None:
            self._check_open()
            self._target_stream.flush()

        @override
        def writelines(self, lines: Iterable[str], /) -> None:
            for line in lines:
                _ = self.write(line)

        @override
        def __iter__(self) -> t.TextStreamIterator:
            return self

        @override
        def __next__(self) -> str:
            self.flush()
            return next(self._target_stream)

        @override
        def __enter__(self) -> t.TextStream:
            self._check_open()
            return self

        @override
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
            /,
        ) -> None:
            _ = exc_type, exc_value, traceback
            self.close()

        def _check_open(self) -> None:
            if self.closed:
                msg = "I/O operation on closed log stream"
                raise ValueError(msg)


__all__: list[str] = ["FlextUtilitiesLoggingStream"]
