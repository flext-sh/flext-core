"""FLEXT Identification Mixin - Entity ID management using centralized components.

This module provides identification mixins that leverage centralized FLEXT
ecosystem components for consistent ID generation and management patterns.
"""

from __future__ import annotations

from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.utilities import FlextUtilities


class FlextIdentification:
    """Unified identification system using centralized FLEXT components."""

    @staticmethod
    def ensure_id(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        id_field: str = "id",
    ) -> None:
        """Ensure object has an ID using FlextUtilities.Generators."""
        if not hasattr(obj, id_field) or getattr(obj, id_field) is None:
            # Generate ID using FlextUtilities
            entity_id = FlextUtilities.Generators.generate_entity_id()

            # Validate using FlextModels
            validated_id = FlextModels.EntityId(root=entity_id)

            setattr(obj, id_field, validated_id.root)
            obj._id_initialized = True

    @staticmethod
    def set_id(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        id_value: str,
        id_field: str = "id",
    ) -> None:
        """Set object ID with validation."""
        # Validate ID using FlextModels
        validated_id = FlextModels.EntityId(root=id_value)

        setattr(obj, id_field, validated_id.root)
        obj._id_initialized = True

    @staticmethod
    def has_id(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        id_field: str = "id",
    ) -> bool:
        """Check if object has an ID."""
        return hasattr(obj, id_field) and getattr(obj, id_field) is not None

    @staticmethod
    def get_id(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        id_field: str = "id",
    ) -> str | None:
        """Get object ID if present."""
        if FlextIdentification.has_id(obj, id_field):
            return str(getattr(obj, id_field))
        return None

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a correlation ID using FlextUtilities."""
        return FlextUtilities.Generators.generate_correlation_id()

    @staticmethod
    def generate_entity_id() -> str:
        """Generate an entity ID using FlextUtilities."""
        return FlextUtilities.Generators.generate_entity_id()

    class Identifiable:
        """Mixin class providing identification capabilities."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize with ID."""
            super().__init__(*args, **kwargs)
            FlextIdentification.ensure_id(self)

        def ensure_id(self, id_field: str = "id") -> None:
            """Ensure this object has an ID."""
            FlextIdentification.ensure_id(self, id_field)

        def set_id(self, id_value: str, id_field: str = "id") -> None:
            """Set this object's ID."""
            FlextIdentification.set_id(self, id_value, id_field)

        def has_id(self, id_field: str = "id") -> bool:
            """Check if this object has an ID."""
            return FlextIdentification.has_id(self, id_field)

        def get_id(self, id_field: str = "id") -> str | None:
            """Get this object's ID."""
            return FlextIdentification.get_id(self, id_field)
