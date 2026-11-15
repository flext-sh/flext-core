"""Initializer that publishes ``Outer``, which nests ``Inner``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class Outer:
    """Export whose nested class a part extends."""

    class Inner:
        """Nested class read through the export as ``Outer.Inner``."""

        INNER_VALUE = 7


__all__: list[str] = ["Outer"]
