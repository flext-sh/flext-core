"""Minimal native text stream consumed by the logging output owner.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import logging
import sys
from typing import TextIO, override

from flext_core import t


class FlextUtilitiesLoggingStream:
    """Own the minimal native stream contract the logging writer consumes."""

    class StreamHandler(logging.StreamHandler[t.TextStream]):
        """Retain native rendering/locking while surfacing the native sink error."""

        include_callsite: bool = True

        @override
        def handleError(self, record: t.LoggingNativeRecord) -> None:
            error = sys.exception()
            if error is not None:
                raise error
            msg = "Logging handler failed without an active exception"
            raise RuntimeError(msg)

    class TextStream(io.TextIOBase, TextIO):
        """Delegate the consumed write lifecycle to a caller-owned stream."""

        def __init__(self, stream: t.TextStream) -> None:
            """Retain the caller-owned stream without taking its lifecycle."""
            super().__init__()
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
        def closed(self) -> bool:
            return self._closed

        @override
        def close(self) -> None:
            """Finish owned work without closing the caller's output stream."""
            if self.closed:
                return
            self.flush()
            self._closed = True

        @override
        def write(self, s: str, /) -> int:
            """Write through to the caller-owned target stream.

            Returns:
                The number of characters written.

            """
            self._check_open()
            return self._target_stream.write(s)

        @override
        def flush(self) -> None:
            """Flush the caller-owned target stream."""
            self._check_open()
            self._target_stream.flush()

        def _check_open(self) -> None:
            if self.closed:
                msg = "I/O operation on closed log stream"
                raise ValueError(msg)


__all__: list[str] = ["FlextUtilitiesLoggingStream"]
