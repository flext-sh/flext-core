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

from collections.abc import Iterator
from datetime import UTC, datetime

from pydantic import Field, RootModel, field_validator

from flext_core.exceptions import FlextValidationError
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult


# Delayed import function to avoid circular imports
def _get_flext_event_class() -> type[object]:
    """Get FlextEvent class with delayed import to avoid circular dependencies."""
    return FlextEvent


# =============================================================================
# TYPE ALIASES USING ROOTMODEL PATTERN
# =============================================================================


# Simple scalar types with validation
class FlextEntityId(RootModel[str]):
    """Entity identifier with validation."""

    root: str = Field(min_length=1, description="Non-empty entity identifier")

    @field_validator("root", mode="before")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate and clean entity ID."""
        if isinstance(v, str):
            # Strip whitespace
            v = v.strip()
            if not v:
                msg = "Entity ID cannot be empty"
                raise FlextValidationError(msg)
        return v

    def __str__(self) -> str:
        """Return string representation."""
        return self.root

    def __hash__(self) -> int:
        """Return hash value."""
        return hash(self.root)

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

    def __eq__(self, other: object) -> bool:
        """Equality comparison with int or FlextVersion."""
        if isinstance(other, int):
            return self.root == other
        if isinstance(other, FlextVersion):
            return self.root == other.root
        return False

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

    def __hash__(self) -> int:
        """Hash for dictionary usage."""
        return hash(self.root)

    def increment(self) -> FlextVersion:
        """Return incremented version."""
        return FlextVersion(self.root + 1)


class FlextTimestamp(RootModel[datetime]):
    """UTC timestamp with validation."""

    root: datetime = Field(default_factory=lambda: datetime.now(UTC))

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

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, FlextTimestamp):
            return self.root == other.root
        return False

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

    root: list[dict[str, object]] = Field(default_factory=list)

    def add_event(self, event_type: str, data: dict[str, object]) -> FlextEventList:
        """Add event and return new list."""
        new_events = self.root.copy()
        new_events.append(
            {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return FlextEventList(new_events)

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
        if hasattr(self, "_flext_events") and 0 <= index < len(self._flext_events):
            return self._flext_events[index]

        event_dict = self.root[index]

        # If it's already a FlextEvent object, return it directly
        if hasattr(event_dict, "event_type"):
            return event_dict

        # For legacy compatibility, convert dict to FlextEvent object if needed
        # This handles the case where tests expect FlextEvent properties
        if isinstance(event_dict, dict) and "type" in event_dict:
            # Extract event information from dictionary
            event_type = event_dict.get("type", "")
            event_data = event_dict.get("data", {})
            aggregate_id = event_dict.get(
                "entity_id",
                event_dict.get("aggregate_id", ""),
            )
            version = event_dict.get("version", 1)

            # Create FlextEvent object using delayed import
            if isinstance(event_type, str) and isinstance(event_data, dict):
                flext_event_cls = _get_flext_event_class()
                # Type: ignore needed as dynamic import loses type info
                event_result = flext_event_cls.create_event(  # type: ignore[attr-defined]
                    event_type=event_type,
                    event_data=event_data,
                    aggregate_id=str(aggregate_id) if aggregate_id else None,
                    version=int(version) if isinstance(version, (int, str)) else None,
                )
                if event_result.is_success:
                    # Return the FlextEvent object directly for tests
                    return event_result.unwrap()

        return event_dict

    def __iter__(self) -> Iterator[dict[str, object]]:  # type: ignore[override]
        """Iterate over events."""
        return iter(self.root)


# =============================================================================
# CONFIGURATION TYPES USING ROOTMODEL PATTERN
# =============================================================================


class FlextHost(RootModel[str]):
    """Host with validation."""

    root: str = Field(min_length=1, description="Non-empty host")

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

    def __str__(self) -> str:
        """Return string representation."""
        return self.root


class FlextErrorMessage(RootModel[str]):
    """Error message with validation."""

    root: str = Field(min_length=1, max_length=512, description="Error message")

    def __str__(self) -> str:
        """Return string representation."""
        return self.root


# =============================================================================
# FACTORY FUNCTIONS FOR CONVENIENT CREATION
# =============================================================================


def create_entity_id(value: str) -> FlextResult[FlextEntityId]:
    """Create validated entity ID."""
    try:
        return FlextResult.ok(FlextEntityId(value))
    except Exception as e:
        return FlextResult.fail(f"Invalid entity ID: {e}")


def create_version(value: int) -> FlextResult[FlextVersion]:
    """Create validated version."""
    try:
        return FlextResult.ok(FlextVersion(value))
    except Exception as e:
        return FlextResult.fail(f"Invalid version: {e}")


def create_email(value: str) -> FlextResult[FlextEmailAddress]:
    """Create validated email address."""
    try:
        return FlextResult.ok(FlextEmailAddress(value))
    except Exception as e:
        return FlextResult.fail(f"Invalid email address: {e}")


def create_service_name(value: str) -> FlextResult[FlextServiceName]:
    """Create validated service name."""
    try:
        return FlextResult.ok(FlextServiceName(value))
    except Exception as e:
        return FlextResult.fail(f"Invalid service name: {e}")


def create_host_port(host: str, port: int) -> FlextResult[tuple[FlextHost, FlextPort]]:
    """Create validated host and port pair."""
    try:
        host_obj = FlextHost(host)
        port_obj = FlextPort(port)
        return FlextResult.ok((host_obj, port_obj))
    except Exception as e:
        return FlextResult.fail(f"Invalid host/port: {e}")


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
type EntityId = FlextEntityId
type Version = FlextVersion
type Timestamp = FlextTimestamp
type Metadata = FlextMetadata
type Host = FlextHost
type Port = FlextPort
type EmailAddress = FlextEmailAddress
type ServiceName = FlextServiceName
type ErrorCode = FlextErrorCode
type ErrorMessage = FlextErrorMessage

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EmailAddress",
    # Type aliases for migration
    "EntityId",
    "ErrorCode",
    "ErrorMessage",
    "FlextConnectionString",
    "FlextEmailAddress",
    # Core RootModel types
    "FlextEntityId",
    "FlextErrorCode",
    "FlextErrorMessage",
    "FlextEventList",
    "FlextHost",
    "FlextMetadata",
    "FlextPercentage",
    "FlextPort",
    "FlextServiceName",
    "FlextTimestamp",
    "FlextVersion",
    "Host",
    "Metadata",
    "Port",
    "ServiceName",
    "Timestamp",
    "Version",
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
