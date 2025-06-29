"""Domain Value Objects for Identifiers - with strict validation.

This module implements all domain identifier value objects to eliminate
primitive string usage throughout the domain layer.

ARCHITECTURAL PRINCIPLES:
- All identifiers are value objects with validation
- No primitive strings in domain service interfaces
- Type safety for all business identifiers
- Immutable value objects with business rules
"""

from __future__ import annotations

import re
from typing import Self
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from flx_core.config.domain_config import get_domain_constants
from flx_core.domain.pydantic_base import DomainValueObject


class PipelineIdString(DomainValueObject):
    """Pipeline identifier as validated string value object - with strict validation."""

    value: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")

    @field_validator("value")
    @classmethod
    def validate_pipeline_id_format(cls, v: str) -> str:
        """Validate pipeline ID follows naming conventions."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            msg = "Pipeline ID can only contain alphanumeric characters, underscores, and hyphens"
            raise ValueError(msg)
        return v

    def __str__(self) -> str:
        """Return string representation for compatibility."""
        return self.value

    def __hash__(self) -> int:
        """Generate hash based on pipeline ID value for set/dict operations."""
        return hash(self.value)

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Create from string with validation."""
        return cls(value=value)


class UserIdString(DomainValueObject):
    """User identifier as validated string value object - with strict validation."""

    value: str = Field(min_length=1, max_length=255)

    @field_validator("value")
    @classmethod
    def validate_user_id_format(cls, v: str) -> str:
        """Validate user ID is not empty after stripping."""
        if not v.strip():
            msg = "User ID cannot be empty or whitespace"
            raise ValueError(msg)
        return v.strip()

    def __str__(self) -> str:
        """Return string representation for compatibility."""
        return self.value

    def __hash__(self) -> int:
        """Generate hash based on user ID value for set/dict operations."""
        return hash(self.value)

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Create from string with validation."""
        return cls(value=value)

    @classmethod
    def system_user(cls) -> Self:
        """Create system user identifier."""
        return cls(value="system")


class OptionalUserId(DomainValueObject):
    """Optional user identifier with explicit None handling - with strict validation."""

    value: UserIdString | None = None

    def is_present(self) -> bool:
        """Check if optional user ID contains a value."""
        return self.value is not None

    def is_system(self) -> bool:
        """Check if this represents the system user account."""
        return self.value is not None and self.value.value == "system"

    def get_value_or_system(self) -> UserIdString:
        """Get user ID or return system user."""
        return self.value if self.value is not None else UserIdString.system_user()

    def __str__(self) -> str:
        """Return string representation or 'None' if empty."""
        return str(self.value) if self.value is not None else "None"

    @classmethod
    def from_string(cls, value: str | None) -> Self:
        """Create OptionalUserId from string value or None."""
        if value is None:
            return cls(value=None)
        return cls(value=UserIdString.from_string(value))

    @classmethod
    def none(cls) -> Self:
        """Create empty optional user ID."""
        return cls(value=None)


class ExecutionIdString(DomainValueObject):
    """Execution identifier as validated string value object - with strict validation."""

    value: str = Field(pattern=r"^[a-fA-F0-9-]{36}$")

    @field_validator("value")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        """Validate string matches UUID format for execution tracking."""
        try:
            # Validate by parsing as UUID
            UUID(v)
        except ValueError as e:
            msg = f"Invalid UUID format: {v}"
            raise ValueError(msg) from e
        else:
            return v

    def __str__(self) -> str:
        """Return string representation for execution ID compatibility."""
        return self.value

    def __hash__(self) -> int:
        """Generate hash based on execution ID for set/dict operations."""
        return hash(self.value)

    @classmethod
    def from_uuid(cls, uuid_val: UUID) -> Self:
        """Convert UUID object to ExecutionIdString value object."""
        return cls(value=str(uuid_val))

    @classmethod
    def generate(cls) -> Self:
        """Generate new UUID-based execution identifier."""
        return cls.from_uuid(uuid4())

    def to_uuid(self) -> UUID:
        """Convert string representation to UUID object."""
        return UUID(self.value)


class PipelineDescription(DomainValueObject):
    """Pipeline description with business rules - with strict validation."""

    value: str = Field(max_length=1000, default="")

    @field_validator("value")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate and clean description."""
        # Strip whitespace and normalize line endings
        return v.strip().replace("\r\n", "\n").replace("\r", "\n")

    def __str__(self) -> str:
        """Return string representation for pipeline description."""
        return self.value

    def __hash__(self) -> int:
        """Generate hash based on description text for set/dict operations."""
        return hash(self.value)

    def is_empty(self) -> bool:
        """Check if description is empty."""
        return not self.value.strip()

    def word_count(self) -> int:
        """Get word count for business validation."""
        return len(self.value.split()) if self.value.strip() else 0

    @classmethod
    def empty(cls) -> Self:
        """Create empty description for pipelines without documentation."""
        return cls(value="")


class ActiveFlag(DomainValueObject):
    """Active/inactive status with explicit business meaning - with strict validation."""

    value: bool

    def is_active(self) -> bool:
        """Check if entity is in active state for execution."""
        return self.value

    def can_execute(self) -> bool:
        """Check if entity can be executed (business rule)."""
        return self.value

    def can_be_scheduled(self) -> bool:
        """Check if entity can be scheduled (business rule)."""
        return self.value

    def __str__(self) -> str:
        """Return string representation for active status."""
        return "active" if self.value else "inactive"

    def __hash__(self) -> int:
        """Generate hash based on active flag for set/dict operations."""
        return hash(self.value)

    @classmethod
    def active(cls) -> Self:
        """Create active flag."""
        return cls(value=True)

    @classmethod
    def inactive(cls) -> Self:
        """Create inactive flag."""
        return cls(value=False)


class Environment(DomainValueObject):
    """Environment specification with validation - with strict validation."""

    value: str = Field(pattern=r"^(dev|test|staging|prod)$")

    @field_validator("value")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of: dev, test, staging, prod."""
        valid_envs = {"dev", "test", "staging", "prod"}
        if v not in valid_envs:
            msg = f"Environment must be one of {valid_envs}, got: {v}"
            raise ValueError(msg)
        return v

    def __str__(self) -> str:
        """Return string representation for environment."""
        return self.value

    def __hash__(self) -> int:
        """Generate hash based on environment value for set/dict operations."""
        return hash(self.value)

    def is_production(self) -> bool:
        """Check if this is production environment."""
        return self.value == "prod"

    def is_development(self) -> bool:
        """Check if this is development environment."""
        return self.value == "dev"

    def allows_debugging(self) -> bool:
        """Check if debugging is allowed in this environment."""
        return self.value in {"dev", "test"}

    @classmethod
    def development(cls) -> Self:
        """Create development environment."""
        return cls(value="dev")

    @classmethod
    def production(cls) -> Self:
        """Create production environment."""
        return cls(value="prod")

    @classmethod
    def testing(cls) -> Self:
        """Create testing environment."""
        return cls(value="test")

    @classmethod
    def staging(cls) -> Self:
        """Create staging environment."""
        return cls(value="staging")


class PaginationLimit(DomainValueObject):
    """Pagination limit with business rules - with strict validation."""

    value: int = Field(ge=1, le=1000, default=100)

    @field_validator("value")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        """Validate pagination limit is between 1 and maximum allowed."""
        constants = get_domain_constants()
        if v < 1:
            msg = "Pagination limit must be at least 1"
            raise ValueError(msg)
        if v > constants.MAX_PIPELINE_LIMIT:
            msg = f"Pagination limit cannot exceed {constants.MAX_PIPELINE_LIMIT}"
            raise ValueError(msg)
        return v

    def __str__(self) -> str:
        """Return string representation for pagination limit."""
        return str(self.value)

    def __hash__(self) -> int:
        """Generate hash based on limit value for set/dict operations."""
        return hash(self.value)

    def is_small_page(self) -> bool:
        """Check if this is a small page size."""
        constants = get_domain_constants()
        return self.value <= constants.DEFAULT_PAGINATION_LIMIT

    def is_large_page(self) -> bool:
        """Check if this is a large page size."""
        constants = get_domain_constants()
        return self.value >= constants.MAX_EXECUTION_LIMIT

    @classmethod
    def default(cls) -> Self:
        """Create default pagination limit."""
        return cls(value=100)

    @classmethod
    def small(cls) -> Self:
        """Create small pagination limit."""
        return cls(value=20)

    @classmethod
    def large(cls) -> Self:
        """Create large pagination limit."""
        return cls(value=500)


class PaginationOffset(DomainValueObject):
    """Pagination offset with validation - with strict validation."""

    value: int = Field(ge=0, default=0)

    @field_validator("value")
    @classmethod
    def validate_offset(cls, v: int) -> int:
        """Validate pagination offset is non-negative for database queries."""
        if v < 0:
            msg = "Pagination offset cannot be negative"
            raise ValueError(msg)
        return v

    def __str__(self) -> str:
        """Return string representation for pagination offset."""
        return str(self.value)

    def __hash__(self) -> int:
        """Generate hash based on offset value for set/dict operations."""
        return hash(self.value)

    def is_first_page(self) -> bool:
        """Check if this is the first page."""
        return self.value == 0

    def get_page_number(self, limit: PaginationLimit) -> int:
        """Calculate 1-based page number from zero-based offset and page size."""
        return (self.value // limit.value) + 1

    @classmethod
    def first_page(cls) -> Self:
        """Create first page offset."""
        return cls(value=0)

    @classmethod
    def for_page(cls, page_number: int, limit: PaginationLimit) -> Self:
        """Create offset for specific page (1-based)."""
        if page_number < 1:
            msg = "Page number must be at least 1"
            raise ValueError(msg)
        return cls(value=(page_number - 1) * limit.value)


# Export all value objects for domain usage
__all__ = [
    "ActiveFlag",
    "Environment",
    "ExecutionIdString",
    "OptionalUserId",
    "PaginationLimit",
    "PaginationOffset",
    "PipelineDescription",
    "PipelineIdString",
    "UserIdString",
]
