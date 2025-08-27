"""Pydantic RootModel patterns for FLEXT Core.

This module demonstrates modern Pydantic RootModel patterns to replace
legacy BaseModel usage throughout the FLEXT ecosystem. RootModel provides
cleaner type safety and reduced boilerplate for simple data structures.

Key Benefits:
- Simplified type definitions
- Better serialization/deserialization
- Reduced code duplication
- Cleaner API design
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast, override

from pydantic import Field, RootModel, field_validator

from flext_core.exceptions import FlextExceptions
from flext_core.payload import FlextEvent, FlextPayload
from flext_core.result import FlextResult

# Type aliases for unified approach with FlextProtocols integration - Python 3.13+ syntax
# Note: These reference the FlextProtocols hierarchy for architectural consistency


# Delayed import functions to avoid circular imports
def _generate_timestamp() -> str:
    """Generate timestamp using FlextUtilities to avoid circular imports."""
    from flext_core.utilities import FlextUtilities  # noqa: PLC0415

    return FlextUtilities.Generators.generate_iso_timestamp()


def _get_flext_event_class() -> type[FlextPayload[Mapping[str, object]]]:
    """Get FlextPayload class for event creation with delayed import to avoid circular dependencies."""
    return FlextPayload


# =============================================================================
# TYPE ALIASES USING ROOTMODEL PATTERN
# =============================================================================


# Simple scalar types with validation
class FlextEntityId(RootModel[str]):
    """Entity identifier with validation."""

    root: str = Field(min_length=1, description="Non-empty entity identifier")

    @field_validator("root", mode="before")
    @classmethod
    def validate_id(cls, v: object) -> str:
        """Validate and clean entity ID."""
        # Convert to string if needed
        if isinstance(v, FlextEntityId):
            return v.root
        if not isinstance(v, str):
            v = str(v)
        # Strip whitespace
        v = v.strip()
        if not v:
            msg = "Entity ID cannot be empty"
            # Use the legacy name that's properly typed
            raise FlextExceptions.FlextValidationError(msg)
        return v

    @override
    def __str__(self) -> str:
        """Return string representation."""
        return self.root

    @override
    def __hash__(self) -> int:
        """Return hash value."""
        return hash(self.root)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare with string or other FlextEntityId."""
        if isinstance(other, str):
            return self.root == other
        if isinstance(other, FlextEntityId):
            return self.root == other.root
        return False


class FlextVersion(RootModel[int]):
    """Entity version with validation."""

    root: int = Field(ge=1, description="Version number starting from 1")

    @field_validator("root", mode="before")
    @classmethod
    def validate_version(cls, v: object) -> int:
        """Validate and convert version number."""
        # Convert to int if needed
        if isinstance(v, FlextVersion):
            return v.root
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError as e:
                msg = f"Invalid version number: {v}"
                raise FlextExceptions.FlextValidationError(msg) from e
        if not isinstance(v, int):
            msg = f"Version must be an integer, got {type(v).__name__}"
            raise FlextExceptions.FlextValidationError(msg)
        if v < 1:
            msg = "Version must be >= 1"
            raise FlextExceptions.FlextValidationError(msg)
        return v

    def __int__(self) -> int:
        """Return integer representation."""
        return self.root

    def __add__(self, other: object) -> FlextVersion:
        """Addition operation."""
        if isinstance(other, int):
            return FlextVersion(root=self.root + other)
        if isinstance(other, FlextVersion):
            return FlextVersion(root=self.root + other.root)
        return NotImplemented

    def __sub__(self, other: object) -> FlextVersion:
        """Subtraction operation."""
        if isinstance(other, int):
            result = self.root - other
            if result < 1:
                msg = "Version cannot be less than 1"
                raise ValueError(msg)
            return FlextVersion(root=result)
        if isinstance(other, FlextVersion):
            result = self.root - other.root
            if result < 1:
                msg = "Version cannot be less than 1"
                raise ValueError(msg)
            return FlextVersion(root=result)
        return NotImplemented

    @override
    def __eq__(self, other: object) -> bool:
        """Equality comparison with int or FlextVersion."""
        if isinstance(other, int):
            return self.root == other
        if isinstance(other, FlextVersion):
            return self.root == other.root
        return False

    @override
    def __ne__(self, other: object) -> bool:
        """Inequality comparison."""
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        """Less than comparison."""
        if isinstance(other, int):
            return self.root < other
        if isinstance(other, FlextVersion):
            return self.root < other.root
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, int):
            return self.root <= other
        if isinstance(other, FlextVersion):
            return self.root <= other.root
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Greater than comparison."""
        if isinstance(other, int):
            return self.root > other
        if isinstance(other, FlextVersion):
            return self.root > other.root
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, int):
            return self.root >= other
        if isinstance(other, FlextVersion):
            return self.root >= other.root
        return NotImplemented

    @override
    def __hash__(self) -> int:
        """Hash for dictionary usage."""
        return hash(self.root)

    def increment(self) -> FlextVersion:
        """Return incremented version."""
        return FlextVersion(self.root + 1)


class FlextTimestamp(RootModel[datetime]):
    """UTC timestamp with validation."""

    root: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("root", mode="before")
    @classmethod
    def validate_timestamp(cls, v: object) -> datetime:
        """Validate and convert timestamp."""
        # Convert to datetime if needed
        if isinstance(v, FlextTimestamp):
            return v.root
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError as e:
                msg = f"Invalid timestamp format: {v}"
                # Use the legacy name that's properly typed
                raise FlextExceptions.FlextValidationError(msg) from e
        msg = f"Timestamp must be datetime or ISO string, got {type(v).__name__}"
        # Use the legacy name that's properly typed
        raise FlextExceptions.FlextValidationError(msg)

    @override
    def __str__(self) -> str:
        """Return ISO format timestamp string."""
        return self.root.isoformat()

    def __lt__(self, other: FlextTimestamp) -> bool:
        """Less than comparison."""
        return self.root < other.root

    def __le__(self, other: FlextTimestamp) -> bool:
        """Less than or equal comparison."""
        return self.root <= other.root

    def __gt__(self, other: FlextTimestamp) -> bool:
        """Greater than comparison."""
        return self.root > other.root

    def __ge__(self, other: FlextTimestamp) -> bool:
        """Greater than or equal comparison."""
        return self.root >= other.root

    @override
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, FlextTimestamp):
            return self.root == other.root
        if isinstance(other, datetime):
            return self.root == other
        return False

    @override
    def __hash__(self) -> int:
        """Hash for dictionary usage."""
        return hash(self.root)

    @classmethod
    def now(cls) -> FlextTimestamp:
        """Create current timestamp."""
        return cls(datetime.now(UTC))


# =============================================================================
# COLLECTION TYPES USING ROOTMODEL PATTERN
# =============================================================================


class FlextMetadata(RootModel[dict[str, object]]):
    """Metadata dictionary with validation."""

    root: dict[str, object] = Field(default_factory=dict)

    def get(self, key: str, default: object = None) -> object:
        """Get metadata value."""
        return self.root.get(key, default)

    def set(self, key: str, value: object) -> FlextMetadata:
        """Return new metadata with added key-value pair."""
        new_data = self.root.copy()
        new_data[key] = value
        return FlextMetadata(new_data)


class FlextEventList(RootModel[list[dict[str, object]]]):
    """Domain events list with validation."""

    # root annotation is inherited from RootModel

    def __init__(
        self,
        root: list[dict[str, object]] | None = None,
        **data: object,
    ) -> None:
        """Initialize with optional internal event storage."""
        if root is None:
            root = []
        super().__init__(root, **data)
        # Internal storage for Flext Event objects - this satisfies MyPy
        object.__setattr__(self, "_flext_events", [])

    def add_event(self, event_type: str, data: dict[str, object]) -> FlextEventList:
        """Add event and return new list."""
        new_events = self.root.copy()
        new_events.append(
            {
                "type": event_type,
                "data": data,
                "timestamp": _generate_timestamp(),
            },
        )
        new_list = FlextEventList(new_events)
        # Copy existing FlextEvent objects
        if hasattr(self, "_flext_events"):
            object.__setattr__(
                new_list,
                "_flext_events",
                getattr(self, "_flext_events", []).copy(),
            )
        return new_list

    def clear(self) -> tuple[FlextEventList, list[dict[str, object]]]:
        """Clear events and return new empty list plus cleared events."""
        events = self.root.copy()
        return FlextEventList([]), events

    def __len__(self) -> int:
        """Return length of events list."""
        return len(self.root)

    def __getitem__(self, index: int) -> object:
        """Get event by index."""
        # Check if we have stored FlextEvent objects from legacy add_domain_event
        flext_events = getattr(self, "_flext_events", [])
        if flext_events and 0 <= index < len(flext_events):
            return flext_events[index]

        event_dict = self.root[index]

        # If it's already a FlextEvent object, return it directly
        if hasattr(event_dict, "event_type"):
            return event_dict

        # For legacy compatibility, convert dict to FlextEvent object if needed
        # This handles the case where  FlextEvent properties
        if hasattr(event_dict, "get") and "type" in event_dict:
            # Extract event information from dictionary
            event_type = event_dict.get("type", "")
            event_data = event_dict.get("data", {})
            aggregate_id = event_dict.get(
                "entity_id",
                event_dict.get("aggregate_id", ""),
            )
            version = event_dict.get("version", 1)

            # Create FlextEvent object using delayed import
            if isinstance(event_type, str) and hasattr(event_data, "get"):
                flext_event_cls = _get_flext_event_class()
                # Dynamic method call requires casting
                typed_event_data = cast("dict[str, object]", event_data)
                event_result: FlextResult[FlextEvent] = cast(
                    "FlextResult[FlextEvent]",
                    flext_event_cls.create_event(
                        event_type=event_type,
                        event_data=typed_event_data,
                        aggregate_id=str(aggregate_id) if aggregate_id else None,
                        version=int(version)
                        if isinstance(version, (int, str))
                        else None,
                    ),
                )
                if event_result.is_success:
                    # Return the FlextEvent object directly
                    return event_result.value

        return event_dict

    # Removed custom __iter__ method to use RootModel's default implementation
    # which returns the correct Generator type


# =============================================================================
# CONFIGURATION TYPES USING ROOTMODEL PATTERN
# =============================================================================


class FlextHost(RootModel[str]):
    """Host with validation."""

    root: str = Field(min_length=1, description="Non-empty host")

    @override
    def __str__(self) -> str:
        """Return string representation."""
        return self.root


class FlextPort(RootModel[int]):
    """Port number with validation."""

    root: int = Field(ge=1, le=65535, description="Valid port number")

    def __int__(self) -> int:
        """Return integer representation."""
        return self.root


class FlextConnectionString(RootModel[str]):
    """Connection string with validation."""

    root: str = Field(min_length=1, description="Non-empty connection string")

    @override
    def __str__(self) -> str:
        """Return string representation."""
        return self.root


# =============================================================================
# BUSINESS DOMAIN TYPES USING ROOTMODEL PATTERN
# =============================================================================


class FlextEmailAddress(RootModel[str]):
    """Email address with validation."""

    root: str = Field(
        pattern=r"^[^@]+@[^@]+\.[^@]+$",
        description="Valid email address",
    )

    @override
    def __str__(self) -> str:
        """Return string representation."""
        return self.root

    @property
    def domain(self) -> str:
        """Extract domain from email."""
        return self.root.split("@")[1]


class FlextServiceName(RootModel[str]):
    """Service name with validation."""

    root: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",
        description="Valid service name (alphanumeric, underscore, hyphen)",
    )

    @override
    def __str__(self) -> str:
        """Return string representation."""
        return self.root


class FlextPercentage(RootModel[float]):
    """Percentage value with validation."""

    root: float = Field(ge=0.0, le=100.0, description="Percentage between 0 and 100")

    def __float__(self) -> float:
        """Return float representation."""
        return self.root

    def as_decimal(self) -> float:
        """Return as decimal (0.0 to 1.0)."""
        return self.root / 100.0


# =============================================================================
# RESULT TYPES USING ROOTMODEL PATTERN
# =============================================================================


class FlextErrorCode(RootModel[str]):
    """Error code with validation."""

    root: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="Error code in UPPER_CASE format",
    )

    @override
    def __str__(self) -> str:
        """Return string representation."""
        return self.root


class FlextErrorMessage(RootModel[str]):
    """Error message with validation."""

    root: str = Field(min_length=1, max_length=512, description="Error message")

    @override
    def __str__(self) -> str:
        """Return string representation."""
        return self.root


# =============================================================================
# FACTORY FUNCTIONS FOR CONVENIENT CREATION
# =============================================================================


def create_entity_id(value: str) -> FlextResult[FlextEntityId]:
    """Create validated entity ID."""
    try:
        return FlextResult[FlextEntityId].ok(FlextEntityId(value))
    except Exception as e:
        return FlextResult[FlextEntityId].fail(f"Invalid entity ID: {e}")


def create_version(value: int) -> FlextResult[FlextVersion]:
    """Create validated version."""
    try:
        return FlextResult[FlextVersion].ok(FlextVersion(value))
    except Exception as e:
        return FlextResult[FlextVersion].fail(f"Invalid version: {e}")


def create_email(value: str) -> FlextResult[FlextEmailAddress]:
    """Create validated email address."""
    try:
        return FlextResult[FlextEmailAddress].ok(FlextEmailAddress(value))
    except Exception as e:
        return FlextResult[FlextEmailAddress].fail(f"Invalid email address: {e}")


def create_service_name(value: str) -> FlextResult[FlextServiceName]:
    """Create validated service name."""
    try:
        return FlextResult[FlextServiceName].ok(FlextServiceName(value))
    except Exception as e:
        return FlextResult[FlextServiceName].fail(f"Invalid service name: {e}")


def create_host_port(host: str, port: int) -> FlextResult[tuple[FlextHost, FlextPort]]:
    """Create validated host and port pair."""
    try:
        host_obj = FlextHost(host)
        port_obj = FlextPort(port)
        return FlextResult[tuple[FlextHost, FlextPort]].ok((host_obj, port_obj))
    except Exception as e:
        return FlextResult[tuple[FlextHost, FlextPort]].fail(f"Invalid host/port: {e}")


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================


def from_legacy_dict(data: dict[str, object]) -> FlextMetadata:
    """Convert legacy dict to FlextMetadata."""
    return FlextMetadata(data)


def to_legacy_dict(metadata: FlextMetadata) -> dict[str, object]:
    """Convert FlextMetadata to legacy dict."""
    return metadata.root


# =============================================================================
# TYPE ALIASES FOR BACKWARDS COMPATIBILITY
# =============================================================================

# These type aliases help with migration from legacy patterns
EntityId = FlextEntityId
Version = FlextVersion
Timestamp = FlextTimestamp
Metadata = FlextMetadata
Host = FlextHost
Port = FlextPort
EmailAddress = FlextEmailAddress
ServiceName = FlextServiceName
ErrorCode = FlextErrorCode
ErrorMessage = FlextErrorMessage

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FlextEmailAddress",  # Email root model
    "FlextEntityId",  # Entity ID root model
    "FlextTimestamp",  # Timestamp root model
    # NOTE: FlextRootModels class not yet implemented
    "FlextVersion",
    # Removed conflicting aliases that exist in validation.py with proper validation:
    # Metadata, Port, ServiceName, Timestamp, Version
    "create_email",
    # Factory functions
    "create_entity_id",
    "create_host_port",
    "create_service_name",
    "create_version",
    # Utilities
    "from_legacy_dict",
    "to_legacy_dict",
]
