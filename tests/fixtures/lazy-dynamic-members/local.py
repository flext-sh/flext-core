"""Part whose base is a class of the same module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class Helper:
    """Same-module base, exported by no initializer."""

    HELPER: ClassVar[int] = 1


class Local(Helper):
    """Inherits ``HELPER`` from its same-module base."""

    LOCAL: ClassVar[int] = 2


__all__: list[str] = ["Local"]
