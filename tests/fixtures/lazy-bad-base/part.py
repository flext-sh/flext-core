"""Part whose base a package exports with another metaclass.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .meta import Abstract


class Capability(Abstract):
    """No stand-in can share an ``ABCMeta`` base."""

    VALUE = 7


__all__: list[str] = ["Capability"]
