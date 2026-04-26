"""Process-local registry of beartype-captured enforcement reports.

Provides a flat ``u.append_violation_report`` / ``u.drain_violation_reports``
/ ``u.clear_violation_reports`` surface used by
``FlextUtilitiesBeartypeEngine.register_violation_capture`` (Task 0.3) to
forward violations into a typed buffer the pytest dispatcher can drain
between tests. Thread-safe so beartype hooks fired from worker threads do
not race with the dispatcher's drain call.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as _me


class FlextUtilitiesRuntimeViolationRegistry:
    """Thread-safe process-local buffer of ``m.Report`` instances."""

    _violation_buffer: ClassVar[list[_me.Report]] = []
    _violation_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def append_violation_report(cls, report: _me.Report) -> None:
        """Buffer ``report`` for later drainage by the dispatcher."""
        with cls._violation_lock:
            cls._violation_buffer.append(report)

    @classmethod
    def drain_violation_reports(cls) -> tuple[_me.Report, ...]:
        """Return every buffered report and reset the buffer atomically.

        Idempotent: a second call returns an empty tuple until new appends
        arrive.
        """
        with cls._violation_lock:
            drained = tuple(cls._violation_buffer)
            cls._violation_buffer.clear()
        return drained

    @classmethod
    def clear_violation_reports(cls) -> None:
        """Drop every buffered report without returning them."""
        with cls._violation_lock:
            cls._violation_buffer.clear()
