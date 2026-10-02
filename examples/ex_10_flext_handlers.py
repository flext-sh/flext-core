"""Handlers example aligned to current stable result contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from examples.protocols import p
from flext_core import r


def run() -> p.Result[str]:
    """Return a successful handler-like response.

    Returns:
        A successful handler-like response.

    """
    return r[str].ok("handler-example")


class Ex10FlextHandlers:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> p.Result[str]:
        """Run handlers example.

        Returns:
            The resulting ``p.Result[str]``.

        """
        return run()


if __name__ == "__main__":
    result = run()
    if not result.success:
        msg = "handler example failed"
        raise RuntimeError(msg)
