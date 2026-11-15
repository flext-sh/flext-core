"""Part that shares its base through the package export.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .bases import Base


class Routed(Base):
    """Its base is the stand-in the ``bases`` initializer publishes."""

    ROUTED = 2


__all__: list[str] = ["Routed"]
