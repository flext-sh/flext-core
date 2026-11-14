"""Part composed from a deferred base that lives in another initializer.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from .bases.base import Base


class Mixin(Base):
    """Overrides ``VALUE`` so the stand-in MRO must match the real one."""

    VALUE: ClassVar[int] = 7

    @classmethod
    def who(cls) -> str:
        """Name the class the member is read through.

        Returns:
            The receiving class name.

        """
        return cls.__name__
