"""Shared fixtures for split decorator unit tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

from tests import tm

if TYPE_CHECKING:
    from collections.abc import Callable


def capture_stdout[T](emit: Callable[[], T], *, contains: str) -> T:
    stream = io.StringIO()
    with redirect_stdout(stream):
        result = emit()
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and contains not in stream.getvalue():
            time.sleep(0.01)
    tm.that(contains in stream.getvalue(), truth=True)
    return result
