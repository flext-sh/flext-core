"""Context example aligned to current public context API.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_core


def run() -> None:
    """Set and read a value from context.

    Raises:
        RuntimeError: If context set failed; or if context get failed.

    """
    ctx = flext_core.FlextContext()
    if not ctx.set("example", "context").success:
        msg = "context set failed"
        raise RuntimeError(msg)
    value = ctx.get("example")
    if not value.success:
        msg = "context get failed"
        raise RuntimeError(msg)


class Ex06FlextContext:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> None:
        """Run context example."""
        run()


if __name__ == "__main__":
    run()
