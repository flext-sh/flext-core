"""Sibling part no access to ``Part`` may load.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class Other:
    """Members of the sibling part."""

    OTHER: ClassVar[int] = 11


__all__: list[str] = ["Other"]
