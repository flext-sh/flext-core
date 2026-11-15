"""Part that declares a member only for type checkers, and its provider.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar


class Declared:
    """Declares ``spawn`` for type checkers; the provider binds it."""

    OWN: ClassVar[int] = 4

    if TYPE_CHECKING:

        @staticmethod
        def spawn() -> str: ...


class Provider:
    """Binds ``spawn`` at run time."""

    @staticmethod
    def spawn() -> str:
        """Name the provider.

        Returns:
            The provider name.

        """
        return "provider"


__all__: list[str] = ["Declared", "Provider"]
