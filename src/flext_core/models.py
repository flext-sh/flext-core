"""FLEXT Core Models - Domain Layer General Models and Data Structures.

Comprehensive domain model system providing general-purpose models and data structures
across the entire 32-project FLEXT ecosystem with Pydantic 2.x, Python 3.13 type safety,
and Domain-Driven Design patterns for enterprise applications.

Module Role in Architecture:
    Domain Layer → General Models → Universal Data Structures and Semantic Types

    This module provides general-purpose models used across the ecosystem:
    - Domain entity models with identity, versioning, and lifecycle management
    - Value object models with attribute-based equality and immutability
    - Operation tracking models for async workflows and process monitoring
    - Service discovery models for microservices architecture
    - Data integration models for Singer, Meltano, DBT pipeline orchestration
    - Application settings models with environment variable integration

Domain Model Patterns:
    Semantic Enums: Universal status types and data formats across projects
    Base Model Hierarchy: FlextBaseModel → Immutable/Mutable → Domain Models
    Domain-Driven Design: Entity, ValueObject, AggregateRoot with rich behaviors
    Operation Tracking: Async operation status, progress, retry mechanisms
    Service Discovery: Microservice registration, health checks, load balancing
    Settings Integration: Environment-aware configuration with secure defaults

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Base models, domain entities, value objects, operation tracking
    🚧 Integration: Event Sourcing capabilities, domain event handling (GAP 2)
    📋 TODO Integration: Plugin architecture models, cross-language serialization
        (GAP 3)

Model Categories by Purpose:
    Core Domain Models:
        - FlextDomainEntity: Identity, versioning, lifecycle, domain events
        - FlextDomainValueObject: Immutable attribute-based equality

    Operation & Process Models:
        - FlextOperationModel: Async operation tracking with progress and retry

    Integration & Communication:
        - FlextSingerStreamModel: Singer stream configuration with schema validation
        - FlextServiceModel: Service discovery with health checks and metadata

    Configuration & Settings:
        - FlextDatabaseModel, FlextOracleModel: Enhanced database configurations
        - FlextBaseSettings: Environment variable integration with validation

Ecosystem Usage Patterns:
    # FlexCore service entity tracking across 32 projects
    service_entity = FlextDomainEntity(
        id="flext-tap-oracle-wms-001",
        status=FlextEntityStatus.ACTIVE,
        metadata={"version": "1.2.0", "environment": "production"}
    )
    service_entity.add_domain_event({
        "event_type": "ServiceStarted",
        "timestamp": datetime.now(UTC).isoformat()
    })

    # Cross-project operation tracking (Meltano pipeline execution)
    pipeline_operation = FlextOperationModel(
        operation_id="meltano-oracle-to-postgres-20250802",
        operation_type="ETL_PIPELINE",
        status=FlextOperationStatus.RUNNING,
        max_retries=3
    )
    pipeline_operation.update_progress(45.2, "Processing inventory records")

    # Singer stream configuration across 15 Singer projects
    inventory_stream = FlextSingerStreamModel(
        stream_name="oracle_inventory",
        tap_name="flext-tap-oracle-wms",
        target_name="flext-target-postgres",
        replication_method="INCREMENTAL",
        replication_key="last_updated",
        batch_size=10000
    )

    # Service discovery for microservices architecture
    api_service = FlextServiceModel(
        service_name="flext-api",
        service_id="flext-api-prod-01",
        host="api.flext.enterprise.com",
        port=8080,
        version="2.1.0",
        environment="production",
        health_check_url="/health"
    )

Domain Modeling Philosophy:
    Rich Domain Models: Models contain business logic and enforce invariants
    Semantic Validation: Business rule validation beyond basic type checking
    Event-Driven Architecture: Domain events for cross-service communication
    Immutability Where Appropriate: Value objects frozen, entities mutable
    Type Safety: Comprehensive type annotations with runtime validation
    Cross-Language Compatibility: Serialization patterns for Go service integration

Quality Standards:
    - Domain-driven design patterns with rich business behaviors
    - Comprehensive semantic validation beyond basic field validation
    - Event sourcing readiness with domain event collection
    - Type-safe factory functions for common model creation scenarios
    - Cross-service serialization compatibility for microservices architecture

See Also:
    docs/TODO.md: Event Sourcing implementation (GAP 2), Plugin architecture (GAP 3)
    config_models.py: Configuration-specific models and TypedDict definitions
    examples/06_flext_entity_valueobject_ddd_patterns.py: Domain modeling patterns

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import NotRequired, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.result import FlextResult

# =============================================================================
# CORE SEMANTIC ENUMS - Domain value objects with type safety
# =============================================================================


class FlextEntityStatus(StrEnum):
    """Universal entity status across all FLEXT projects."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DELETED = "deleted"
    SUSPENDED = "suspended"


class FlextOperationStatus(StrEnum):
    """Universal operation status for async operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class FlextDataFormat(StrEnum):
    """Universal data formats across FLEXT ecosystem."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    LDIF = "ldif"
    YAML = "yaml"
    PARQUET = "parquet"


class FlextConnectionType(StrEnum):
    """Universal connection types."""

    DATABASE = "database"
    REDIS = "redis"
    LDAP = "ldap"
    ORACLE = "oracle"
    REST_API = "rest_api"
    GRPC = "grpc"
    FILE = "file"


# =============================================================================
# CORE DOMAIN TYPEDDICT DEFINITIONS - Type-safe dictionaries
# =============================================================================


class FlextEntityDict(TypedDict):
    """TypedDict for core entity structure."""

    id: str
    created_at: str
    updated_at: str
    version: int
    status: FlextEntityStatus
    metadata: NotRequired[dict[str, object]]


class FlextValueObjectDict(TypedDict):
    """TypedDict for value object structure."""

    value: str | int | float | bool | None
    type: str
    metadata: NotRequired[dict[str, object]]


class FlextOperationDict(TypedDict):
    """TypedDict for operation tracking."""

    operation_id: str
    operation_type: str
    status: FlextOperationStatus
    started_at: str
    completed_at: NotRequired[str | None]
    error_message: NotRequired[str | None]
    metadata: NotRequired[dict[str, object]]


class FlextConnectionDict(TypedDict):
    """TypedDict for connection configuration."""

    connection_id: str
    connection_type: FlextConnectionType
    host: str
    port: int
    credentials: dict[str, str]
    options: NotRequired[dict[str, object]]


# =============================================================================
# CORE SEMANTIC BASE MODELS - Foundation for all FLEXT models
# =============================================================================


class FlextBaseModel(BaseModel):
    """Base semantic model for all FLEXT Pydantic models.

    Provides consistent configuration, validation, and behavior across
    the entire FLEXT ecosystem with Python 3.13 type safety.
    """

    model_config = ConfigDict(
        # Type safety and validation
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        # Serialization
        arbitrary_types_allowed=True,
        validate_default=True,
        # Python 3.13 compatibility
        frozen=False,  # Allow mutation for entities
    )

    # Metadata for all models
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Model metadata",
    )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation."""
        return self.model_dump()

    def to_typed_dict(self) -> dict[str, object]:
        """Convert to typed dictionary representation."""
        return self.model_dump(exclude_unset=True)

    def validate_semantic_rules(self) -> FlextResult[None]:
        """Validate semantic business rules - override in subclasses."""
        return FlextResult.ok(None)


class FlextImmutableModel(FlextBaseModel):
    """Immutable base model for value objects and configuration."""

    model_config = ConfigDict(
        # Type safety and validation
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        # Serialization
        arbitrary_types_allowed=True,
        validate_default=True,
        # Immutable
        frozen=True,
    )


class FlextMutableModel(FlextBaseModel):
    """Mutable base model for entities and services."""

    model_config = ConfigDict(
        # Type safety and validation
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        # Serialization
        arbitrary_types_allowed=True,
        validate_default=True,
        # Mutable
        frozen=False,
    )


# =============================================================================
# DOMAIN ENTITY MODELS - Core entity patterns with DDD
# =============================================================================


class FlextDomainEntity(FlextMutableModel):
    """Base domain entity with identity, versioning, and lifecycle management."""

    # Core identity
    id: str = Field(description="Unique entity identifier")

    # Lifecycle management
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: int = Field(default=1, description="Entity version for optimistic locking")

    # Status management
    status: FlextEntityStatus = Field(default=FlextEntityStatus.ACTIVE)

    # Domain events (placeholder for event sourcing)
    domain_events: list[dict[str, object]] = Field(default_factory=list, exclude=True)

    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on entity ID."""
        if not isinstance(other, FlextDomainEntity):
            return False
        return self.id == other.id

    def increment_version(self) -> None:
        """Increment version and update timestamp."""
        self.version += 1
        self.updated_at = datetime.now(UTC)

    def add_domain_event(self, event: dict[str, object]) -> None:
        """Add domain event for event sourcing."""
        self.domain_events.append(event)

    def clear_domain_events(self) -> list[dict[str, object]]:
        """Clear and return domain events."""
        events = self.domain_events.copy()
        self.domain_events.clear()
        return events

    def to_entity_dict(self) -> FlextEntityDict:
        """Convert to FlextEntityDict."""
        return FlextEntityDict(
            id=self.id,
            created_at=self.created_at.isoformat(),
            updated_at=self.updated_at.isoformat(),
            version=self.version,
            status=self.status,
            metadata=self.metadata,
        )


class FlextDomainValueObject(FlextImmutableModel):
    """Base domain value object with attribute-based equality."""

    def __hash__(self) -> int:
        """Hash based on all attributes (with safe handling for unhashable types)."""

        def make_hashable(item: object) -> object:
            """Convert unhashable types to hashable equivalents."""
            if isinstance(item, dict):
                return tuple(sorted(item.items()))
            if isinstance(item, list):
                return tuple(item)
            if isinstance(item, set):
                return frozenset(item)
            return item

        dumped = self.model_dump()
        hashable_items = []
        for key, value in sorted(dumped.items()):
            hashable_items.append((key, make_hashable(value)))
        return hash(tuple(hashable_items))

    def __eq__(self, other: object) -> bool:
        """Equality based on all attributes."""
        if not isinstance(other, type(self)):
            return False
        return self.model_dump() == other.model_dump()


# =============================================================================
# CONFIGURATION MODELS - Enhanced with Python 3.13 patterns
# =============================================================================


class FlextDatabaseModel(FlextImmutableModel):
    """Enhanced database configuration with semantic validation."""

    # Connection details
    host: str = Field("localhost", description="Database host address")
    port: int = Field(5432, description="Database port", ge=1, le=65535)
    username: str = Field("postgres", description="Database username")
    password: SecretStr = Field(SecretStr("password"), description="Database password")
    database: str = Field("flext", description="Database name")

    # Connection pool configuration
    pool_min: int = Field(1, description="Minimum pool connections", ge=1)
    pool_max: int = Field(10, description="Maximum pool connections", ge=1)
    pool_timeout: int = Field(30, description="Pool timeout in seconds", ge=1)

    # SSL and security
    ssl_enabled: bool = Field(default=False, description="Enable SSL connection")
    ssl_cert_path: str | None = Field(None, description="SSL certificate path")

    # Advanced options
    encoding: str = Field("UTF-8", description="Character encoding")
    autocommit: bool = Field(default=False, description="Enable autocommit mode")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host is not empty."""
        if not v or not v.strip():
            msg = "Database host cannot be empty"
            raise ValueError(msg)
        return v.strip()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username is not empty."""
        if not v or not v.strip():
            msg = "Database username cannot be empty"
            raise ValueError(msg)
        return v.strip()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        password = self.password.get_secret_value()
        return f"postgresql://{self.username}:{password}@{self.host}:{self.port}/{self.database}"

    def validate_semantic_rules(self) -> FlextResult[None]:
        """Validate database-specific business rules."""
        if self.pool_min > self.pool_max:
            return FlextResult.fail("pool_min cannot be greater than pool_max")
        return FlextResult.ok(None)


class FlextOracleModel(FlextImmutableModel):
    """Enhanced Oracle configuration with semantic validation."""

    # Connection details
    host: str = Field("localhost", description="Oracle host address")
    port: int = Field(1521, description="Oracle port", ge=1, le=65535)
    username: str = Field("oracle", description="Oracle username")
    password: SecretStr = Field(SecretStr("oracle"), description="Oracle password")

    # Oracle connection identifiers
    service_name: str | None = Field(None, description="Oracle service name")
    sid: str | None = Field(None, description="Oracle system identifier")

    # Connection pool configuration
    pool_min: int = Field(1, description="Minimum pool connections", ge=1)
    pool_max: int = Field(10, description="Maximum pool connections", ge=1)
    pool_increment: int = Field(1, description="Pool increment", ge=1)
    timeout: int = Field(30, description="Connection timeout", ge=1)

    # Oracle-specific options
    encoding: str = Field("UTF-8", description="Character encoding")
    ssl_enabled: bool = Field(default=False, description="Enable SSL connection")
    protocol: str = Field("tcp", description="Connection protocol")

    @field_validator("service_name", "sid")
    @classmethod
    def validate_connection_identifier(cls, v: str | None) -> str | None:
        """Validate connection identifier format."""
        return v

    @classmethod
    def model_validate(cls, obj: dict[str, object]) -> FlextOracleModel:  # type: ignore[override]
        """Validate that either SID or service_name is provided."""
        instance = super().model_validate(obj)
        if not instance.service_name and not instance.sid:
            msg = "Either service_name or sid must be provided"
            raise ValueError(msg)
        return instance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def connection_string(self) -> str:
        """Generate Oracle connection string."""
        if self.service_name:
            return f"{self.host}:{self.port}/{self.service_name}"
        if self.sid:
            return f"{self.host}:{self.port}:{self.sid}"
        return f"{self.host}:{self.port}"

    def validate_semantic_rules(self) -> FlextResult[None]:
        """Validate Oracle-specific business rules."""
        if not self.service_name and not self.sid:
            return FlextResult.fail("Either service_name or sid must be provided")
        return FlextResult.ok(None)


# =============================================================================
# OPERATION TRACKING MODELS - For async operations and workflows
# =============================================================================


class FlextOperationModel(FlextMutableModel):
    """Model for tracking async operations and workflows."""

    # Operation identity
    operation_id: str = Field(description="Unique operation identifier")
    operation_type: str = Field(description="Type of operation")

    # Status tracking
    status: FlextOperationStatus = Field(default=FlextOperationStatus.PENDING)

    # Timing information
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = Field(None, description="Operation completion time")

    # Error handling
    error_message: str | None = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Progress tracking
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    progress_message: str | None = Field(None, description="Progress description")

    def mark_started(self) -> None:
        """Mark operation as started."""
        self.status = FlextOperationStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def mark_completed(self) -> None:
        """Mark operation as completed successfully."""
        self.status = FlextOperationStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        self.progress_percentage = 100.0

    def mark_failed(self, error_message: str) -> None:
        """Mark operation as failed."""
        self.status = FlextOperationStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error_message = error_message

    def increment_retry(self) -> bool:
        """Increment retry count. Returns True if retries available."""
        self.retry_count += 1
        if self.retry_count <= self.max_retries:
            self.status = FlextOperationStatus.RETRYING
            return True
        return False

    def update_progress(self, percentage: float, message: str | None = None) -> None:
        """Update operation progress."""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        if message:
            self.progress_message = message

    def to_operation_dict(self) -> FlextOperationDict:
        """Convert to FlextOperationDict."""
        return FlextOperationDict(
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            status=self.status,
            started_at=self.started_at.isoformat(),
            completed_at=self.completed_at.isoformat() if self.completed_at else None,
            error_message=self.error_message,
            metadata=self.metadata,
        )


# =============================================================================
# DATA INTEGRATION MODELS - For Singer, Meltano, DBT
# =============================================================================


class FlextSingerStreamModel(FlextImmutableModel):
    """Model for Singer stream configuration."""

    # Stream identity
    stream_name: str = Field(description="Singer stream name")
    tap_name: str = Field(description="Source tap name")
    target_name: str | None = Field(None, description="Target name")

    # Schema configuration
    schema_definition: dict[str, object] = Field(
        default_factory=dict,
        description="Stream schema",
    )
    key_properties: list[str] = Field(default_factory=list, description="Primary keys")
    replication_method: str = Field("FULL_TABLE", description="Replication method")
    replication_key: str | None = Field(None, description="Replication key field")

    # Processing configuration
    batch_size: int = Field(1000, description="Batch processing size", ge=1)
    max_records: int | None = Field(None, description="Maximum records to process")

    # Metadata
    description: str | None = Field(None, description="Stream description")
    tags: list[str] = Field(default_factory=list, description="Stream tags")

    @field_validator("stream_name")
    @classmethod
    def validate_stream_name(cls, v: str) -> str:
        """Validate stream name format."""
        if not v or not v.strip():
            msg = "Singer stream name cannot be empty"
            raise ValueError(msg)
        return v.strip()

    def validate_semantic_rules(self) -> FlextResult[None]:
        """Validate Singer stream business rules."""
        if self.replication_method == "INCREMENTAL" and not self.replication_key:
            return FlextResult.fail("INCREMENTAL replication requires replication_key")
        return FlextResult.ok(None)


# =============================================================================
# SERVICE DISCOVERY MODELS - For microservices architecture
# =============================================================================


class FlextServiceModel(FlextMutableModel):
    """Model for service registration and discovery."""

    # Service identity
    service_name: str = Field(description="Service name")
    service_id: str = Field(description="Unique service instance ID")

    # Network configuration
    host: str = Field(description="Service host address")
    port: int = Field(description="Service port", ge=1, le=65535)
    protocol: str = Field("http", description="Service protocol")

    # Service metadata
    version: str = Field(description="Service version")
    environment: str = Field("development", description="Deployment environment")

    # Health check configuration
    health_check_url: str | None = Field(None, description="Health check endpoint")
    health_check_interval: int = Field(
        30,
        description="Health check interval (seconds)",
        ge=1,
    )

    # Service discovery
    tags: list[str] = Field(default_factory=list, description="Service tags")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def service_url(self) -> str:
        """Generate full service URL."""
        return f"{self.protocol}://{self.host}:{self.port}"

    def validate_semantic_rules(self) -> FlextResult[None]:
        """Validate service configuration business rules."""
        if self.health_check_url and not self.health_check_url.startswith("/"):
            return FlextResult.fail("health_check_url must start with '/'")
        return FlextResult.ok(None)


# =============================================================================
# SETTINGS CLASSES WITH ENVIRONMENT INTEGRATION
# =============================================================================


class FlextBaseSettings(BaseSettings):
    """Enhanced base settings with comprehensive environment integration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        env_nested_delimiter="__",  # Support nested environment variables
    )

    # Common application settings
    app_name: str = Field("FLEXT Application", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    environment: str = Field("development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")

    def validate_semantic_rules(self) -> FlextResult[None]:
        """Validate application settings business rules."""
        valid_environments = {"development", "staging", "production", "test"}
        if self.environment not in valid_environments:
            return FlextResult.fail(f"Environment must be one of: {valid_environments}")
        return FlextResult.ok(None)


# =============================================================================
# FACTORY FUNCTIONS - Type-safe model creation
# =============================================================================


def create_database_model(**kwargs: object) -> FlextDatabaseModel:
    """Factory function to create database model with validation."""
    # Create safe config data with defaults
    config_data = {
        "host": "localhost",
        "port": 5432,
        "username": "postgres",
        "password": SecretStr("password"),
        "database": "flext",
    }

    # Override with provided kwargs if they are valid
    if isinstance(kwargs, dict):
        for key, value in kwargs.items():
            if key in config_data:
                config_data[key] = value

    return FlextDatabaseModel(**config_data)  # type: ignore[arg-type]


def create_oracle_model(**kwargs: object) -> FlextOracleModel:
    """Factory function to create Oracle model with validation."""
    # Create safe config data with defaults
    config_data = {
        "host": "localhost",
        "port": 1521,
        "username": "oracle",
        "password": SecretStr("oracle"),
        "service_name": "ORCL",
    }

    # Override with provided kwargs if they are valid
    if isinstance(kwargs, dict):
        for key, value in kwargs.items():
            if key in config_data:
                config_data[key] = value

    return FlextOracleModel(**config_data)  # type: ignore[arg-type]


def create_operation_model(
    operation_id: str,
    operation_type: str,
    **kwargs: object,
) -> FlextOperationModel:
    """Factory function to create operation tracking model."""
    # Create safe config data with defaults
    config_data = {
        "operation_id": operation_id,
        "operation_type": operation_type,
        "metadata": {},
    }

    # Override with provided kwargs if they are valid
    if isinstance(kwargs, dict):
        for key, value in kwargs.items():
            if hasattr(FlextOperationModel, key):
                config_data[key] = value  # type: ignore[assignment]  # noqa: PERF403

    return FlextOperationModel(**config_data)  # type: ignore[arg-type]


def create_singer_stream_model(
    stream_name: str,
    tap_name: str,
    **kwargs: object,
) -> FlextSingerStreamModel:
    """Factory function to create Singer stream model."""
    # Create safe config data with defaults
    config_data = {
        "stream_name": stream_name,
        "tap_name": tap_name,
        "schema_definition": {},
        "key_properties": [],
        "batch_size": 1000,
        "tags": [],
    }

    # Override with provided kwargs if they are valid
    if isinstance(kwargs, dict):
        config_data.update(
            {
                key: value
                for key, value in kwargs.items()
                if hasattr(FlextSingerStreamModel, key)
            },
        )

    return FlextSingerStreamModel(**config_data)  # type: ignore[arg-type]


def create_service_model(
    service_name: str,
    host: str,
    port: int,
    **kwargs: object,
) -> FlextServiceModel:
    """Factory function to create service model."""
    # Create safe config data with defaults
    config_data = {
        "service_name": service_name,
        "service_id": f"{service_name}-{host}-{port}",
        "host": host,
        "port": port,
        "version": "1.0.0",
        "tags": [],
    }

    # Override with provided kwargs if they are valid
    if isinstance(kwargs, dict):
        config_data.update(
            {
                key: value
                for key, value in kwargs.items()
                if hasattr(FlextServiceModel, key)
            },
        )

    return FlextServiceModel(**config_data)  # type: ignore[arg-type]


# =============================================================================
# MODEL VALIDATION UTILITIES
# =============================================================================


def validate_all_models(*models: FlextBaseModel) -> FlextResult[None]:
    """Validate multiple models and their semantic rules."""
    for model in models:
        # Pydantic validation
        try:
            model.model_validate(model.model_dump())
        except Exception as e:
            return FlextResult.fail(f"Model validation failed: {e}")

        # Semantic validation
        semantic_result = model.validate_semantic_rules()
        if semantic_result.is_failure:
            return semantic_result

    return FlextResult.ok(None)


def model_to_dict_safe(model: FlextBaseModel) -> dict[str, object]:
    """Safely convert model to dictionary with error handling."""
    try:
        return model.to_dict()
    except Exception:
        return {}


# =============================================================================
# EXPORTS - Comprehensive model system
# =============================================================================


__all__ = [
    # Base models
    "FlextBaseModel",
    # Settings
    "FlextBaseSettings",
    "FlextConnectionDict",
    "FlextConnectionType",
    "FlextDataFormat",
    # Configuration models
    "FlextDatabaseModel",
    # Domain models
    "FlextDomainEntity",
    "FlextDomainValueObject",
    # TypedDict definitions
    "FlextEntityDict",
    # Core enums
    "FlextEntityStatus",
    "FlextImmutableModel",
    "FlextMutableModel",
    "FlextOperationDict",
    # Operation models
    "FlextOperationModel",
    "FlextOperationStatus",
    "FlextOracleModel",
    # Service models
    "FlextServiceModel",
    # Data integration models
    "FlextSingerStreamModel",
    "FlextValueObjectDict",
    # Factory functions
    "create_database_model",
    "create_operation_model",
    "create_oracle_model",
    "create_service_model",
    "create_singer_stream_model",
    "model_to_dict_safe",
    # Utilities
    "validate_all_models",
]

# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# Rebuild all Pydantic models to resolve forward references after import
FlextBaseModel.model_rebuild()
FlextImmutableModel.model_rebuild()
FlextMutableModel.model_rebuild()
FlextDomainEntity.model_rebuild()
FlextDomainValueObject.model_rebuild()
FlextDatabaseModel.model_rebuild()
FlextOracleModel.model_rebuild()
FlextOperationModel.model_rebuild()
FlextSingerStreamModel.model_rebuild()
FlextServiceModel.model_rebuild()
