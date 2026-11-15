"""Part whose members a deriving class's metaclass reads while creating it.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class Part:
    """Members that ``ABCMeta`` and Pydantic read from a stand-in base."""

    model_config: ClassVar[dict[str, bool]] = {"frozen": True}

    VALUE: ClassVar[int] = 7

    def who(self) -> str:
        """Implement the abstract method a deriving class declares.

        Returns:
            The receiving class name.

        """
        return type(self).__name__


__all__: list[str] = ["Part"]
