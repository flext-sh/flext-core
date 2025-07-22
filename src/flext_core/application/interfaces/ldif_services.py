"""LDIF Services Interfaces - Abstract domain contracts.

This module defines abstract interfaces for LDIF processing operations
that enterprise applications can depend on without coupling to
concrete LDIF implementations.

Copyright (c) 2025 Flext. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from flext_core.domain.shared_types import ServiceResult


class LDIFEntryProtocol(Protocol):
    """Protocol for LDIF entry representation."""

    dn: str
    changetype: str | None
    attributes: dict[str, Any]


class LDIFProcessorInterface(ABC):
    """Abstract interface for LDIF processing operations.

    This interface defines the contract for LDIF file operations
    without coupling to any specific LDIF implementation.
    Enterprise applications should depend on this interface.
    """

    @abstractmethod
    async def read_entries(
        self,
        file_path: Path,
    ) -> ServiceResult[list[LDIFEntryProtocol]]:
        """Read entries from LDIF file.

        Args:
            file_path: Path to LDIF file

        Returns:
            ServiceResult containing list of entries or error

        """
        ...

    @abstractmethod
    async def write_entries(
        self,
        entries: list[LDIFEntryProtocol],
        file_path: Path,
    ) -> ServiceResult[bool]:
        """Write entries to LDIF file.

        Args:
            entries: List of LDIF entries to write
            file_path: Path where to write LDIF file

        Returns:
            ServiceResult indicating success or failure

        """
        ...

    @abstractmethod
    async def sort_entries_hierarchical(
        self,
        entries: list[LDIFEntryProtocol],
    ) -> ServiceResult[list[LDIFEntryProtocol]]:
        """Sort entries in hierarchical order (parents before children).

        Args:
            entries: List of LDIF entries to sort

        Returns:
            ServiceResult containing sorted entries or error

        """
        ...

    @abstractmethod
    async def validate_entry(
        self,
        entry: LDIFEntryProtocol,
    ) -> ServiceResult[bool]:
        """Validate LDIF entry format and content.

        Args:
            entry: LDIF entry to validate

        Returns:
            ServiceResult indicating validation success or failure

        """
        ...

    @abstractmethod
    async def merge_entries(
        self,
        entries1: list[LDIFEntryProtocol],
        entries2: list[LDIFEntryProtocol],
    ) -> ServiceResult[list[LDIFEntryProtocol]]:
        """Merge two lists of LDIF entries.

        Args:
            entries1: First list of entries
            entries2: Second list of entries

        Returns:
            ServiceResult containing merged entries or error

        """
        ...


class LDIFAdapterInterface(ABC):
    """Abstract interface for LDIF processor adapters.

    This interface allows different LDIF implementations
    to be plugged in via dependency injection.
    """

    @abstractmethod
    def get_ldif_processor(self) -> LDIFProcessorInterface:
        """Get LDIF processor implementation.

        Returns:
            Configured LDIF processor implementation

        """
        ...
