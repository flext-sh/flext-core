"""Shared domain models for FLEXT ecosystem.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

These models are used across multiple FLEXT modules to ensure consistency.
All modules should import these models instead of creating duplicates.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any

# Import UUID directly since it's used at runtime
from pydantic import Field
from pydantic import field_validator

from flext_core.domain.pydantic_base import DomainBaseModel
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainValueObject

if TYPE_CHECKING:
    from uuid import UUID


class EntityStatus(StrEnum):
    """Status for entities."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DELETED = "deleted"


class OperationStatus(StrEnum):
    """Status for async operations."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineExecutionStatus(StrEnum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LogLevel(StrEnum):
    """Standard log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Authentication Models
class AuthToken(DomainValueObject):
    """Authentication token value object."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds
    refresh_token: str | None = None
    scope: str | None = None


class UserInfo(DomainEntity):
    """Basic user information for API responses."""

    username: str | None = None
    full_name: str | None = None
    is_active: bool = True
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)


# Plugin Models
class PluginType(StrEnum):
    """Plugin types."""

    EXTRACTOR = "extractor"
    LOADER = "loader"
    TRANSFORM = "transform"
    ORCHESTRATOR = "orchestrator"
    UTILITY = "utility"


class PluginMetadata(DomainBaseModel):
    """Plugin metadata for registration and discovery."""

    name: str
    version: str | None = None
    author: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] | None = None


# Health & Monitoring Models
class HealthStatus(StrEnum):
    """Health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(DomainBaseModel):
    """Health status of a system component."""

    name: str
    status: HealthStatus | None = None
    checks: dict[str, Any] = Field(default_factory=dict)
    last_check: datetime | None = None


class SystemHealth(DomainBaseModel):
    """System health."""

    status: HealthStatus
    components: list[ComponentHealth] = Field(default_factory=list)
    version: str | None = None
    uptime: float | None = None  # seconds


# Error Models
class ErrorDetail(DomainBaseModel):
    """Detailed error information."""

    code: str
    message: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(DomainBaseModel):
    """Detailed error information."""

    success: bool = False
    error: ErrorDetail
    correlation_id: str | None = Field(default=None, alias="request_id")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Configuration Models
class DatabaseConfig(DomainValueObject):
    """Enhanced database connection configuration to eliminate duplications."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(
        default=5432,
        description="Database port",
        ge=1,
        le=65535,
        repr=False,
    )
    database: str = Field(default="flext", description="Database name")
    username: str = Field(default="user", description="Database username")
    password: str = Field(
        default="password",
        description="Database password",
        repr=False,
    )

    # Connection pool settings (consolidated from multiple projects)
    pool_size: int = Field(default=20, description="Connection pool size", ge=1, le=100)
    max_overflow: int = Field(
        default=40,
        description="Maximum overflow connections",
        ge=0,
        le=100,
    )
    pool_min_size: int = Field(default=1, description="Minimum pool size", ge=1)
    pool_max_size: int = Field(default=10, description="Maximum pool size", ge=1)

    # Connection behavior
    pool_timeout: int = Field(default=30, description="Pool timeout seconds", ge=1)
    pool_recycle: int = Field(default=3600, description="Pool recycle seconds", ge=60)
    connect_timeout: int = Field(default=30, description="Connection timeout", ge=1)

    @field_validator("pool_max_size")
    @classmethod
    def validate_pool_max(cls, v: int, info: Any) -> int:
        """Validate pool_max_size >= pool_min_size."""
        if hasattr(info, "data") and "pool_min_size" in info.data:
            pool_min = info.data.get("pool_min_size", 1)
            if v < pool_min:
                msg = f"pool_max_size ({v}) must be >= pool_min_size ({pool_min})"
                raise ValueError(msg)
        return v

    def get_url(self) -> str:
        """Get SQLAlchemy database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(DomainValueObject):
    """Enhanced Redis connection configuration to eliminate duplications."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port", ge=1, le=65535)
    db: int = Field(default=0, description="Redis database number", ge=0, le=15)
    password: str | None = Field(default=None, description="Redis password", repr=False)

    # Connection settings (consolidated from multiple projects)
    decode_responses: bool = Field(
        default=True,
        description="Decode responses to strings",
    )
    socket_timeout: int = Field(default=5, description="Socket timeout seconds", ge=1)
    socket_keepalive: bool = Field(default=True, description="Enable socket keepalive")

    # Pool settings (from duplicated configs)
    pool_size: int = Field(default=20, description="Connection pool size", ge=1, le=100)
    max_connections: int = Field(
        default=50,
        description="Maximum connections",
        ge=1,
        le=1000,
    )
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")

    # Key management (from auth configs)
    key_prefix: str = Field(default="flext:", description="Key prefix for namespacing")

    def get_url(self) -> str:
        """Get Redis connection URL."""
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.db}"


class HTTPConnectionConfig(DomainValueObject):
    """HTTP connection configuration to eliminate Oracle OIC/WMS duplications."""

    base_url: str = Field(..., description="Base URL for API endpoints")
    timeout: int = Field(
        default=30,
        description="Request timeout seconds",
        ge=1,
        le=300,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts",
        ge=0,
        le=10,
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries",
        ge=0.1,
        le=60.0,
    )
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    # Connection pooling
    pool_connections: int = Field(
        default=10,
        description="Pool connections",
        ge=1,
        le=100,
    )
    pool_maxsize: int = Field(default=20, description="Pool max size", ge=1, le=200)

    # Headers and auth
    user_agent: str = Field(default="FLEXT-Client/1.0", description="User-Agent header")
    additional_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers",
    )


class SecurityConfig(DomainValueObject):
    """Security configuration to eliminate auth/API duplications."""

    secret_key: str = Field(
        default="change-this-secret-in-production",
        description="Secret key for signing",
        min_length=32,
        repr=False,
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration",
        ge=1,
        le=1440,
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration",
        ge=1,
        le=365,
    )

    # Password policies
    min_password_length: int = Field(
        default=8,
        description="Minimum password length",
        ge=6,
    )
    require_special_chars: bool = Field(
        default=True,
        description="Require special characters",
    )
    password_history_count: int = Field(
        default=5,
        description="Password history",
        ge=0,
        le=20,
    )


# LDAP Models
class LDAPScope(StrEnum):
    """LDAP search scope."""

    BASE = "base"
    ONE = "one"
    SUB = "sub"


class LDAPEntry(DomainBaseModel):
    """LDAP entry representation."""

    dn: str
    attributes: dict[str, list[str]]

    @field_validator("dn")
    @classmethod
    def validate_dn(cls, v: str) -> str:
        """Validate DN.

        Arguments:
            v: The DN    to validate.

        Raises:
            ValueError: If the DN is empty.

        Returns:
            The validated DN.

        """
        if not v or not v.strip():
            msg = "DN cannot be empty"
            raise ValueError(msg)
        return v.strip()


# Pipeline Models (extend from domain)
class PipelineConfig(DomainBaseModel):
    """Pipeline configuration for API."""

    name: str
    description: str | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    schedule: str | None = None  # cron expression
    timeout: int = 3600  # seconds
    retries: int = 3
    is_active: bool = True


class PipelineRunStatus(DomainBaseModel):
    """Pipeline run status."""

    run_id: UUID
    pipeline_id: UUID | None = None
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


# Data Models
class DataSchema(DomainBaseModel):
    """Schema definition for data validation."""

    fields: dict[str, dict[str, Any]]
    required: list[str] = Field(default_factory=list)
    unique: list[str] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=list)


class DataRecord(DomainBaseModel):
    """Data record."""

    id: UUID | None = None
    data: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
