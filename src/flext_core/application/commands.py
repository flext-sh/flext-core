"""Base command classes for CQRS pattern.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Base classes for implementing commands in the CQRS pattern.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class Command(ABC):
    """Base class for all commands.

    Commands represent intent to change the system state.
    They should be immutable and contain all data needed
    to perform the operation.
    """

    @abstractmethod
    def validate(self) -> bool:
        """Validate command data before execution.

        Returns:
            True if command is valid, False otherwise.

        """


__all__ = [
    "Command",
]
