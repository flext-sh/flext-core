"""Composition part that binds no member of its own, only its base.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .bases.base import Base


class Composed(Base):
    """Every member comes from ``Base``."""


__all__: list[str] = ["Composed"]
