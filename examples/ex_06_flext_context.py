"""Context example aligned to current public context API.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextContext


class ExamplesFlextContext:
    """Demonstrate the current public context API."""

    @staticmethod
    def run() -> None:
        """Set and read a value from context.

        Raises:
            RuntimeError: If context set failed; or if context get failed.

        """
        ctx = FlextContext()
        if not ctx.set("example", "context").success:
            msg = "context set failed"
            raise RuntimeError(msg)
        value = ctx.get("example")
        if not value.success:
            msg = "context get failed"
            raise RuntimeError(msg)


if __name__ == "__main__":
    ExamplesFlextContext.run()

__all__ = ["ExamplesFlextContext"]
