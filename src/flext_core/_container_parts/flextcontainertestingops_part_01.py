"""Container testing-only singleton reset operations.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from typing import ClassVar, Self


class FlextContainerTestingOps:
    """Testing-only singleton reset operations for the shared container."""

    _global_instance: Self | None = None

    _global_lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._global_lock:
            cls._global_instance = None


__all__: list[str] = ["FlextContainerTestingOps"]
