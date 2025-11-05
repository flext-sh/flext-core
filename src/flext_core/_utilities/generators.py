"""Utilities module - FlextUtilitiesGenerators.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import secrets
import string
import uuid
import warnings
from datetime import UTC, datetime

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
_logger = logging.getLogger(__name__)


class FlextUtilitiesGenerators:
    """ID and data generation utilities."""

    @staticmethod
    def _generate_prefixed_id(
        prefix: str,
        *parts: object,
        length: int = FlextConstants.Utilities.SHORT_UUID_LENGTH,
    ) -> str:
        """Factory method for generating prefixed IDs with UUID.

        **INTERNAL METHOD**: This is a private implementation detail used
        by public ID generation methods. Do not call directly - use the
        specific public methods instead (generate_correlation_id,
        generate_batch_id, etc.).

        This method consolidates 12+ similar ID generation methods following
        DRY principle (Don't Repeat Yourself).

        Args:
            prefix: ID prefix (e.g., 'corr', 'batch', 'txn')
            *parts: Optional middle parts (e.g., batch_size, context)
            length: UUID truncation length (SHORT or LONG)

        Returns:
            Formatted ID string: {prefix}_{parts}_{uuid[:length]}

        Example:
            >>> _generate_prefixed_id("corr")
            'corr_a1b2c3d4'
            >>> _generate_prefixed_id("batch", 100)
            'batch_100_a1b2c3d4'

        """
        uuid_part = str(uuid.uuid4())[:length]
        if parts:
            middle = "_".join(str(p) for p in parts)
            return f"{prefix}_{middle}_{uuid_part}"
        return f"{prefix}_{uuid_part}"

    @staticmethod
    def generate_id() -> str:
        """Generate a unique ID using UUID4."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_uuid() -> str:
        """Generate a UUID string."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_timestamp() -> str:
        """Generate ISO format timestamp without microseconds.

        .. deprecated:: 0.9.9
            Use :func:`generate_iso_timestamp` instead. This method is identical
            and will be removed in version 2.0.0.

        """
        warnings.warn(
            "generate_timestamp() is deprecated, use generate_iso_timestamp() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return FlextUtilitiesGenerators.generate_iso_timestamp()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp without microseconds."""
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a correlation ID for tracking."""
        return FlextUtilitiesGenerators._generate_prefixed_id("corr")

    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """Generate a short random ID."""
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def generate_entity_id() -> str:
        """Generate a unique entity ID for domain entities.

        Returns:
            A unique entity identifier suitable for domain entities

        """
        return str(uuid.uuid4())

    @staticmethod
    def create_module_utilities(module_name: str) -> FlextResult[type]:
        """Create utilities for a specific module.

        Args:
            module_name: Name of the module to create utilities for

        Returns:
            FlextResult containing module utilities type or error

        """
        if not module_name:
            return FlextResult[type].fail(
                "Module name must be a non-empty string",
            )

        # For now, return a simple utilities object
        # This can be expanded with actual module-specific functionality
        utilities = type(
            f"{module_name}_utilities",
            (),
            {
                "module_name": module_name,
                "logger": lambda: f"Logger for {module_name}",
                "config": lambda: f"Config for {module_name}",
            },
        )()

        return FlextResult[type].ok(type(utilities))

    @staticmethod
    def generate_correlation_id_with_context(context: str) -> str:
        """Generate a correlation ID with context prefix."""
        return FlextUtilitiesGenerators._generate_prefixed_id(context)

    @staticmethod
    def generate_batch_id(batch_size: int) -> str:
        """Generate a batch ID with size information."""
        return FlextUtilitiesGenerators._generate_prefixed_id("batch", batch_size)

    @staticmethod
    def generate_transaction_id() -> str:
        """Generate a transaction ID for distributed transactions."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "txn", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_saga_id() -> str:
        """Generate a saga ID for distributed transaction patterns."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "saga", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_event_id() -> str:
        """Generate an event ID for domain events."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "evt", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_command_id() -> str:
        """Generate a command ID for CQRS patterns."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "cmd", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_query_id() -> str:
        """Generate a query ID for CQRS patterns."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            "qry", length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_aggregate_id(aggregate_type: str) -> str:
        """Generate an aggregate ID with type prefix."""
        return FlextUtilitiesGenerators._generate_prefixed_id(
            aggregate_type, length=FlextConstants.Utilities.LONG_UUID_LENGTH
        )

    @staticmethod
    def generate_entity_version() -> int:
        """Generate an entity version number using FlextConstants.Context."""
        return (
            int(
                datetime.now(UTC).timestamp()
                * FlextConstants.Context.MILLISECONDS_PER_SECOND
            )
            % FlextConstants.Utilities.VERSION_MODULO
        ) + 1

    @staticmethod
    def ensure_id(obj: FlextTypes.CachedObjectType) -> None:
        """Ensure object has an ID using FlextUtilities and FlextConstants.

        Args:
            obj: Object to ensure ID for

        """
        if hasattr(obj, FlextConstants.Mixins.FIELD_ID):
            id_value = getattr(obj, FlextConstants.Mixins.FIELD_ID, None)
            if not id_value:
                new_id = FlextUtilitiesGenerators.generate_id()
                setattr(obj, FlextConstants.Mixins.FIELD_ID, new_id)


__all__ = ["FlextUtilitiesGenerators"]
