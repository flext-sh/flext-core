"""Container testing-only singleton reset operations.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextContainerTestingOps:
    """Testing-only singleton reset operations for the shared container."""

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._global_lock:
            cls._global_instance = None


__all__: list[str] = ["FlextContainerTestingOps"]
