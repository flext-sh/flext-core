"""Primitive Types - Fundamental Value Objects.

Provides the most basic value objects used throughout FLEXT.
These are simple, immutable types that form the building blocks.
"""

from __future__ import annotations

import uuid
from datetime import UTC
from datetime import datetime
from typing import NewType

from flext_core.foundation.abstractions import AbstractValueObject

# Primitive ID types - strongly typed identifiers
EntityId = NewType("EntityId", str)
UserId = NewType("UserId", str)
TenantId = NewType("TenantId", str)
SessionId = NewType("SessionId", str)


def generate_entity_id() -> EntityId:
    """Generate a new unique entity identifier."""
    return EntityId(str(uuid.uuid4()))


def generate_user_id() -> UserId:
    """Generate a new unique user identifier."""
    return UserId(str(uuid.uuid4()))


def generate_tenant_id() -> TenantId:
    """Generate a new unique tenant identifier."""
    return TenantId(str(uuid.uuid4()))


def generate_session_id() -> SessionId:
    """Generate a new unique session identifier."""
    return SessionId(str(uuid.uuid4()))


class Timestamp(AbstractValueObject):
    """Immutable timestamp value object.

    Represents a point in time with timezone awareness.
    Always uses UTC internally for consistency.
    """

    def __init__(self, value: datetime | None = None) -> None:
        if value is None:
            value = datetime.now(UTC)
        elif value.tzinfo is None:
            # Assume UTC if no timezone provided
            value = value.replace(tzinfo=UTC)
        else:
            # Convert to UTC
            value = value.astimezone(UTC)

        self._value = value

    @property
    def value(self) -> datetime:
        """The datetime value in UTC."""
        return self._value

    def __eq__(self, other: object) -> bool:
        """Timestamps are equal if they represent the same moment in time."""
        if not isinstance(other, Timestamp):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        """Hash based on the datetime value."""
        return hash(self._value)

    def __str__(self) -> str:
        """ISO format string representation."""
        return self._value.isoformat()

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Timestamp({self._value.isoformat()})"

    def __lt__(self, other: Timestamp) -> bool:
        """Support comparison operators."""
        return self._value < other._value

    def __le__(self, other: Timestamp) -> bool:
        """Support comparison operators."""
        return self._value <= other._value

    def __gt__(self, other: Timestamp) -> bool:
        """Support comparison operators."""
        return self._value > other._value

    def __ge__(self, other: Timestamp) -> bool:
        """Support comparison operators."""
        return self._value >= other._value


class Version(AbstractValueObject):
    """Semantic version value object.

    Represents a semantic version number (major.minor.patch).
    Useful for entity versioning and API versioning.
    """

    def __init__(self, major: int, minor: int, patch: int) -> None:
        if major < 0 or minor < 0 or patch < 0:
            msg = "Version numbers must be non-negative"
            raise ValueError(msg)

        self._major = major
        self._minor = minor
        self._patch = patch

    @property
    def major(self) -> int:
        """Major version number."""
        return self._major

    @property
    def minor(self) -> int:
        """Minor version number."""
        return self._minor

    @property
    def patch(self) -> int:
        """Patch version number."""
        return self._patch

    @classmethod
    def from_string(cls, version_str: str) -> Version:
        """Create version from string like '1.2.3'."""
        try:
            parts = version_str.split(".")
            if len(parts) != 3:
                msg = f"Invalid version format: {version_str}"
                raise ValueError(msg)

            major, minor, patch = map(int, parts)
            return cls(major, minor, patch)
        except ValueError as e:
            msg = f"Invalid version format: {version_str}"
            raise ValueError(msg) from e

    def __eq__(self, other: object) -> bool:
        """Versions are equal if all parts match."""
        if not isinstance(other, Version):
            return False
        return (self._major, self._minor, self._patch) == (
            other._major,
            other._minor,
            other._patch,
        )

    def __hash__(self) -> int:
        """Hash based on version components."""
        return hash((self._major, self._minor, self._patch))

    def __str__(self) -> str:
        """Return string representation in major.minor.patch format."""
        return f"{self._major}.{self._minor}.{self._patch}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Version({self._major}, {self._minor}, {self._patch})"

    def __lt__(self, other: Version) -> bool:
        """Version comparison for ordering."""
        return (self._major, self._minor, self._patch) < (
            other._major,
            other._minor,
            other._patch,
        )

    def __le__(self, other: Version) -> bool:
        """Version comparison for ordering."""
        return (self._major, self._minor, self._patch) <= (
            other._major,
            other._minor,
            other._patch,
        )

    def __gt__(self, other: Version) -> bool:
        """Version comparison for ordering."""
        return (self._major, self._minor, self._patch) > (
            other._major,
            other._minor,
            other._patch,
        )

    def __ge__(self, other: Version) -> bool:
        """Version comparison for ordering."""
        return (self._major, self._minor, self._patch) >= (
            other._major,
            other._minor,
            other._patch,
        )


class Email(AbstractValueObject):
    """Email address value object.

    Represents a valid email address with basic validation.
    Immutable and ensures email format correctness.
    """

    def __init__(self, value: str) -> None:
        normalized = value.strip().lower()
        if not self._is_valid_email(normalized):
            msg = f"Invalid email address: {value}"
            raise ValueError(msg)

        self._value = normalized

    @property
    def value(self) -> str:
        """The normalized email address."""
        return self._value

    @property
    def local_part(self) -> str:
        """The local part of the email (before @)."""
        return self._value.split("@")[0]

    @property
    def domain(self) -> str:
        """The domain part of the email (after @)."""
        return self._value.split("@")[1]

    def _is_valid_email(self, email: str) -> bool:
        """Validate email format."""
        if "@" not in email:
            return False

        parts = email.split("@")
        if len(parts) != 2:
            return False

        local, domain = parts
        if not local or not domain:
            return False

        if "." not in domain:
            return False

        # Basic character validation
        return not any(
            char in local for char in [" ", ",", "<", ">", "(", ")", "[", "]"]
        )

    def __eq__(self, other: object) -> bool:
        """Emails are equal if they have the same normalized value."""
        if not isinstance(other, Email):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        """Hash based on normalized email value."""
        return hash(self._value)

    def __str__(self) -> str:
        """Return string representation of the email."""
        return self._value

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Email('{self._value}')"
