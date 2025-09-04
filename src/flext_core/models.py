"""Consolidated model system using Pydantic with RootModel validation.

Provides FlextModels class with nested entity types, value objects, and factory methods
for creating type-safe domain models with efficient validation.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import ClassVar, Generic, Self
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T


class FlextModels:
    """Consolidated FLEXT model system providing all domain modeling functionality.

    This is the complete model system for the FLEXT ecosystem, providing a unified
    approach to domain modeling using Pydantic BaseModel and RootModel patterns.
    All model types are organized as nested classes within this single container
    for consistent configuration and easy access.

    """

    # =============================================================================
    # BASE MODEL CONFIGURATION
    # =============================================================================

    class Config(BaseModel):
        """Advanced configuration base class for all FLEXT models.

        Provides enterprise-grade configuration management with environment support,
        validation, performance tracking, and security features. This is the single
        configuration class used throughout the FLEXT ecosystem.
        """

        model_config = ConfigDict(
            # Validation settings
            validate_assignment=True,
            validate_default=True,
            use_enum_values=True,
            # JSON settings
            arbitrary_types_allowed=True,
            extra="forbid",
            # Serialization settings
            ser_json_bytes="base64",
            ser_json_timedelta="iso8601",
            # Performance settings
            revalidate_instances="always",
            # String settings
            str_strip_whitespace=True,
            str_to_upper=False,
            str_to_lower=False,
            # Security settings
            hide_input_in_errors=True,
        )

        # Core configuration metadata
        config_version: str = Field(
            default="1.0.0",
            description="Configuration schema version",
            pattern=r"^\d+\.\d+\.\d+$",
        )
        config_environment: str = Field(
            default="development",
            description="Configuration environment",
        )
        config_source: str = Field(
            default="default",
            description="Configuration source",
        )
        config_priority: int = Field(
            default=5,
            ge=1,
            le=10,
            description="Configuration priority",
        )

        # Performance and monitoring
        validation_count: int = Field(
            default=0,
            ge=0,
            description="Number of validations performed",
        )
        last_validated: datetime | None = Field(
            default=None,
            description="Last validation timestamp",
        )
        validation_cache_hits: int = Field(
            default=0,
            ge=0,
            description="Validation cache hit count",
        )

        # Security and secrets management
        enable_secret_masking: bool = Field(
            default=True,
            description="Enable secret field masking in logs",
        )
        secret_fields: list[str] = Field(
            default_factory=list,
            description="Fields to mask as secrets",
        )

        # Environment-specific settings
        debug_mode: bool = Field(default=False, description="Enable debug mode")
        enable_metrics: bool = Field(
            default=True,
            description="Enable performance metrics",
        )
        enable_audit_logging: bool = Field(
            default=True,
            description="Enable audit logging",
        )

        @field_validator("config_environment")
        @classmethod
        def validate_environment(cls, v: str) -> str:
            """Validate configuration environment."""
            valid_environments = [
                "development",
                "staging",
                "production",
                "test",
                "local",
            ]
            if v not in valid_environments:
                msg = f"Invalid environment '{v}'. Valid options: {valid_environments}"
                raise ValueError(msg)
            return v

        @field_validator("config_source")
        @classmethod
        def validate_source(cls, v: str) -> str:
            """Validate configuration source."""
            valid_sources = ["default", "file", "environment", "cli", "api"]
            if v not in valid_sources:
                msg = f"Invalid source '{v}'. Valid options: {valid_sources}"
                raise ValueError(msg)
            return v

        @field_validator("config_priority")
        @classmethod
        def validate_priority(cls, v: int) -> int:
            """Validate configuration priority."""
            if (
                v < FlextConstants.Config.MIN_PRIORITY
                or v > FlextConstants.Config.MAX_PRIORITY
            ):
                msg = f"Configuration priority must be between {FlextConstants.Config.MIN_PRIORITY} and {FlextConstants.Config.MAX_PRIORITY}"
                raise ValueError(msg)
            return v

        @model_validator(mode="after")
        def validate_production_priority(self) -> Self:
            """Validate production environment priority constraints."""
            if (
                self.config_environment == "production"
                and self.config_priority > FlextConstants.Config.PRODUCTION_MAX_PRIORITY
            ):
                msg = f"Production configurations should have priority <= {FlextConstants.Config.PRODUCTION_MAX_PRIORITY}"
                raise ValueError(msg)
            return self

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate business-specific configuration rules (validation tracking only)."""
            try:
                # Update validation tracking only - field validation is handled by Pydantic
                self.validation_count += 1
                self.last_validated = datetime.now(UTC)
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Business rule validation failed: {e}")

        def get_config_info(self) -> dict[str, object]:
            """Get configuration information for debugging and monitoring."""
            return {
                "version": self.config_version,
                "environment": self.config_environment,
                "source": self.config_source,
                "priority": self.config_priority,
                "validation_count": self.validation_count,
                "last_validated": self.last_validated.isoformat()
                if self.last_validated
                else None,
                "cache_hits": self.validation_cache_hits,
            }

        def is_production(self) -> bool:
            """Check if configuration is for production environment."""
            return self.config_environment == "production"

        def is_development(self) -> bool:
            """Check if configuration is for development environment."""
            return self.config_environment in {"development", "local"}

        def is_test(self) -> bool:
            """Check if configuration is for test environment."""
            return self.config_environment == "test"

        def is_staging(self) -> bool:
            """Check if configuration is for staging environment."""
            return self.config_environment == "staging"

        def is_local(self) -> bool:
            """Check if configuration is for local environment."""
            return self.config_environment == "local"

        def get_environment_config(self) -> dict[str, object]:
            """Get environment-specific configuration settings."""
            base_config = {
                "debug_mode": self.debug_mode,
                "enable_metrics": self.enable_metrics,
                "enable_audit_logging": self.enable_audit_logging,
            }

            if self.is_production():
                return {
                    **base_config,
                    "debug_mode": False,
                    "enable_metrics": True,
                    "enable_audit_logging": True,
                    "validation_strictness": "high",
                }
            if self.is_development() or self.is_local():
                return {
                    **base_config,
                    "debug_mode": True,
                    "enable_metrics": False,
                    "enable_audit_logging": True,
                    "validation_strictness": "normal",
                }
            if self.is_test():
                return {
                    **base_config,
                    "debug_mode": False,
                    "enable_metrics": False,
                    "enable_audit_logging": False,
                    "validation_strictness": "loose",
                }
            # staging
            return {
                **base_config,
                "debug_mode": False,
                "enable_metrics": True,
                "enable_audit_logging": True,
                "validation_strictness": "high",
            }

        def mask_secrets(self, data: dict[str, object]) -> dict[str, object]:
            """Mask secret fields in configuration data."""
            if not self.enable_secret_masking:
                return data

            masked_data = data.copy()
            for field in self.secret_fields:
                if field in masked_data:
                    masked_data[field] = "***MASKED***"

            return masked_data

        def get_config_summary(self) -> dict[str, object]:
            """Get configuration summary for monitoring and debugging."""
            return {
                "version": self.config_version,
                "environment": self.config_environment,
                "source": self.config_source,
                "priority": self.config_priority,
                "validation_count": self.validation_count,
                "last_validated": self.last_validated.isoformat()
                if self.last_validated
                else None,
                "cache_hits": self.validation_cache_hits,
                "debug_mode": self.debug_mode,
                "metrics_enabled": self.enable_metrics,
                "audit_enabled": self.enable_audit_logging,
            }

        def increment_validation_count(self) -> None:
            """Increment validation count and update timestamp."""
            self.validation_count += 1
            self.last_validated = datetime.now(UTC)

        def increment_cache_hits(self) -> None:
            """Increment validation cache hit count."""
            self.validation_cache_hits += 1

        def reset_performance_counters(self) -> None:
            """Reset performance tracking counters."""
            self.validation_count = 0
            self.validation_cache_hits = 0
            self.last_validated = None

    class DatabaseConfig(Config):
        """Database configuration with connection pooling and security validation.

        Provides comprehensive database configuration with connection pooling,
        security validation, and environment-specific optimizations.
        """

        # Connection settings
        host: str = Field(..., description="Database host")
        port: int = Field(default=5432, ge=1, le=65535, description="Database port")
        database: str = Field(..., description="Database name")
        username: str = Field(..., description="Database username")
        password: str = Field(..., description="Database password")

        # Override secret fields for database config
        secret_fields: list[str] = Field(
            default_factory=lambda: ["password", "ssl_key"],
            description="Database secret fields to mask",
        )

        # Connection pooling
        pool_size: int = Field(
            default=10,
            ge=1,
            le=100,
            description="Connection pool size",
        )
        max_overflow: int = Field(
            default=20,
            ge=0,
            le=100,
            description="Maximum pool overflow",
        )
        pool_timeout: int = Field(
            default=30,
            ge=1,
            le=300,
            description="Pool timeout in seconds",
        )
        pool_recycle: int = Field(
            default=3600,
            ge=300,
            le=86400,
            description="Pool recycle time in seconds",
        )

        # Security settings
        ssl_mode: str = Field(default="prefer", description="SSL connection mode")
        ssl_cert: str | None = Field(default=None, description="SSL certificate path")
        ssl_key: str | None = Field(default=None, description="SSL key path")
        ssl_ca: str | None = Field(default=None, description="SSL CA certificate path")

        # Performance settings
        connect_timeout: int = Field(
            default=10,
            ge=1,
            le=60,
            description="Connection timeout in seconds",
        )
        query_timeout: int = Field(
            default=30,
            ge=1,
            le=300,
            description="Query timeout in seconds",
        )
        enable_prepared_statements: bool = Field(
            default=True,
            description="Enable prepared statements",
        )

        @field_validator("host")
        @classmethod
        def validate_host(cls, v: str) -> str:
            """Validate database host."""
            if not v or not v.strip():
                msg = "Database host cannot be empty"
                raise ValueError(msg)
            return v.strip().lower()

        @field_validator("database")
        @classmethod
        def validate_database(cls, v: str) -> str:
            """Validate database name."""
            if not v or not v.strip():
                msg = "Database name cannot be empty"
                raise ValueError(msg)
            # Remove invalid characters
            cleaned = v.strip().replace(" ", "_").replace("-", "_")
            if not cleaned.replace("_", "").isalnum():
                msg = "Database name must contain only alphanumeric characters and underscores"
                raise ValueError(msg)
            return cleaned

        @field_validator("ssl_mode")
        @classmethod
        def validate_ssl_mode(cls, v: str) -> str:
            """Validate SSL mode."""
            valid_modes = [
                "disable",
                "allow",
                "prefer",
                "require",
                "verify-ca",
                "verify-full",
            ]
            if v not in valid_modes:
                msg = f"Invalid SSL mode '{v}'. Valid options: {valid_modes}"
                raise ValueError(msg)
            return v

        def get_connection_string(self) -> str:
            """Get database connection string."""
            ssl_params = ""
            if self.ssl_mode != "disable":
                ssl_params = f"?sslmode={self.ssl_mode}"
                if self.ssl_cert:
                    ssl_params += f"&sslcert={self.ssl_cert}"
                if self.ssl_key:
                    ssl_params += f"&sslkey={self.ssl_key}"
                if self.ssl_ca:
                    ssl_params += f"&sslrootcert={self.ssl_ca}"

            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}{ssl_params}"

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate database-specific business rules."""
            try:
                # Call parent validation
                parent_result = super().validate_business_rules()
                if parent_result.is_failure:
                    return parent_result

                # Database-specific validation
                if self.pool_size > self.max_overflow:
                    return FlextResult[None].fail(
                        "Pool size cannot be greater than max overflow",
                    )

                if self.is_production() and self.ssl_mode in {"disable", "allow"}:
                    return FlextResult[None].fail(
                        "Production databases must use secure SSL modes",
                    )

                if (
                    self.is_production()
                    and self.pool_size < FlextConstants.Config.MIN_POOL_SIZE
                ):
                    return FlextResult[None].fail(
                        f"Production databases should have pool size >= {FlextConstants.Config.MIN_POOL_SIZE}",
                    )

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Database validation failed: {e}")

    class SecurityConfig(Config):
        """Security configuration with encryption and authentication settings.

        Provides comprehensive security configuration including encryption,
        authentication, authorization, and audit logging.

        """

        # Encryption settings
        secret_key: str = Field(..., description="Application secret key")
        jwt_secret: str = Field(..., description="JWT signing secret")
        encryption_key: str = Field(..., description="Data encryption key")

        # Override secret fields for security config
        secret_fields: list[str] = Field(
            default_factory=lambda: ["secret_key", "jwt_secret", "encryption_key"],
            description="Security secret fields to mask",
        )

        # Authentication settings
        session_timeout: int = Field(
            default=3600,
            ge=300,
            le=86400,
            description="Session timeout in seconds",
        )
        jwt_expiry: int = Field(
            default=3600,
            ge=300,
            le=86400,
            description="JWT expiry in seconds",
        )
        refresh_token_expiry: int = Field(
            default=604800,
            ge=3600,
            le=2592000,
            description="Refresh token expiry in seconds",
        )

        # Password policy
        min_password_length: int = Field(
            default=8,
            ge=6,
            le=128,
            description="Minimum password length",
        )
        require_uppercase: bool = Field(
            default=True,
            description="Require uppercase letters",
        )
        require_lowercase: bool = Field(
            default=True,
            description="Require lowercase letters",
        )
        require_numbers: bool = Field(default=True, description="Require numbers")
        require_special_chars: bool = Field(
            default=True,
            description="Require special characters",
        )

        # Rate limiting
        rate_limit_requests: int = Field(
            default=100,
            ge=1,
            le=10000,
            description="Rate limit requests per window",
        )
        rate_limit_window: int = Field(
            default=3600,
            ge=60,
            le=86400,
            description="Rate limit window in seconds",
        )

        # Security headers
        enable_cors: bool = Field(default=True, description="Enable CORS")
        cors_origins: list[str] = Field(
            default_factory=list,
            description="Allowed CORS origins",
        )
        enable_csrf: bool = Field(default=True, description="Enable CSRF protection")

        @field_validator("secret_key")
        @classmethod
        def validate_secret_key(cls, v: str) -> str:
            """Validate secret key strength."""
            min_length = FlextConstants.Config.MIN_SECRET_LENGTH
            if len(v) < min_length:
                msg = f"Secret key must be at least {min_length} characters"
                raise ValueError(msg)
            if not any(c.isupper() for c in v):
                msg = "Secret key must contain uppercase letters"
                raise ValueError(msg)
            if not any(c.islower() for c in v):
                msg = "Secret key must contain lowercase letters"
                raise ValueError(msg)
            if not any(c.isdigit() for c in v):
                msg = "Secret key must contain numbers"
                raise ValueError(msg)
            return v

        @field_validator("jwt_secret")
        @classmethod
        def validate_jwt_secret(cls, v: str) -> str:
            """Validate JWT secret strength."""
            min_length = FlextConstants.Config.MIN_SECRET_LENGTH
            if len(v) < min_length:
                msg = f"JWT secret must be at least {min_length} characters"
                raise ValueError(msg)
            return v

        @field_validator("encryption_key")
        @classmethod
        def validate_encryption_key(cls, v: str) -> str:
            """Validate encryption key strength."""
            min_length = FlextConstants.Config.MIN_SECRET_LENGTH
            if len(v) < min_length:
                msg = f"Encryption key must be at least {min_length} characters"
                raise ValueError(msg)
            return v

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate security-specific business rules."""
            try:
                # Call parent validation
                parent_result = super().validate_business_rules()
                if parent_result.is_failure:
                    return parent_result

                # Security-specific validation
                if self.is_production():
                    if (
                        len(self.secret_key)
                        < FlextConstants.Config.PRODUCTION_SECRET_LENGTH
                    ):
                        return FlextResult[None].fail(
                            f"Production secret key must be at least {FlextConstants.Config.PRODUCTION_SECRET_LENGTH} characters",
                        )

                    if not self.enable_csrf:
                        return FlextResult[None].fail(
                            "CSRF protection must be enabled in production",
                        )

                    if not self.cors_origins:
                        return FlextResult[None].fail(
                            "CORS origins must be specified in production",
                        )

                if self.session_timeout > self.jwt_expiry:
                    return FlextResult[None].fail(
                        "Session timeout cannot be greater than JWT expiry",
                    )

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Security validation failed: {e}")

    class LoggingConfig(Config):
        """Logging configuration with structured logging and performance monitoring.

        Provides comprehensive logging configuration including structured logging,
        performance monitoring, and log rotation.
        """

        # Basic logging settings
        log_level: str = Field(default="INFO", description="Log level")
        log_format: str = Field(default="json", description="Log format (json, text)")
        log_file: str | None = Field(default=None, description="Log file path")

        # Log rotation
        max_file_size: int = Field(
            default=10485760,
            ge=1048576,
            le=1073741824,
            description="Max file size in bytes",
        )
        backup_count: int = Field(
            default=5,
            ge=1,
            le=100,
            description="Number of backup files",
        )
        rotation_when: str = Field(default="midnight", description="Rotation schedule")

        # Performance monitoring
        enable_performance_logging: bool = Field(
            default=True,
            description="Enable performance logging",
        )
        slow_query_threshold: float = Field(
            default=1.0,
            ge=FlextConstants.Config.MIN_ROTATION_SIZE,
            le=60.0,
            description="Slow query threshold in seconds",
        )
        enable_metrics: bool = Field(
            default=True,
            description="Enable metrics collection",
        )

        # Error tracking
        enable_error_tracking: bool = Field(
            default=True,
            description="Enable error tracking",
        )
        error_sample_rate: float = Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Error sampling rate",
        )

        # Audit logging
        enable_audit_logging: bool = Field(
            default=True,
            description="Enable audit logging",
        )
        audit_events: list[str] = Field(
            default_factory=lambda: ["login", "logout", "create", "update", "delete"],
            description="Events to audit",
        )

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            """Validate log level."""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"]
            if v.upper() not in valid_levels:
                msg = f"Invalid log level '{v}'. Valid options: {valid_levels}"
                raise ValueError(msg)
            return v.upper()

        @field_validator("log_format")
        @classmethod
        def validate_log_format(cls, v: str) -> str:
            """Validate log format."""
            valid_formats = ["json", "text", "structured"]
            if v.lower() not in valid_formats:
                msg = f"Invalid log format '{v}'. Valid options: {valid_formats}"
                raise ValueError(msg)
            return v.lower()

        @field_validator("rotation_when")
        @classmethod
        def validate_rotation_when(cls, v: str) -> str:
            """Validate rotation schedule."""
            valid_schedules = [
                "midnight",
                "S",
                "M",
                "H",
                "D",
                "W0",
                "W1",
                "W2",
                "W3",
                "W4",
                "W5",
                "W6",
            ]
            if v not in valid_schedules:
                msg = (
                    f"Invalid rotation schedule '{v}'. Valid options: {valid_schedules}"
                )
                raise ValueError(msg)
            return v

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate logging-specific business rules."""
            try:
                # Call parent validation
                parent_result = super().validate_business_rules()
                if parent_result.is_failure:
                    return parent_result

                # Logging-specific validation
                if self.is_production():
                    if self.log_level in {"DEBUG", "TRACE"}:
                        return FlextResult[None].fail(
                            "Production should not use DEBUG or TRACE log levels",
                        )

                    if not self.log_file:
                        return FlextResult[None].fail(
                            "Production must specify a log file",
                        )

                if (
                    self.error_sample_rate > FlextConstants.Config.MIN_ROTATION_SIZE
                    and self.is_production()
                ):
                    return FlextResult[None].fail(
                        f"Production error sampling rate should be <= {FlextConstants.Config.MIN_ROTATION_SIZE}",
                    )

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Logging validation failed: {e}")

    # =============================================================================
    # DOMAIN MODEL CLASSES
    # =============================================================================

    class Entity(Config, ABC):
        """Mutable entities with identity, versioning and domain events.

        Entities have identity that persists across state changes and support
        domain events, versioning, and lifecycle management.
        """

        # Core identity fields
        id: str = Field(..., description="Unique entity identifier")
        version: int = Field(
            default=1,
            description="Entity version for optimistic locking",
        )

        # Metadata fields
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        created_by: str | None = Field(
            default=None,
            description="User who created entity",
        )
        updated_by: str | None = Field(
            default=None,
            description="User who last updated entity",
        )

        # Domain events (not persisted)
        domain_events: list[FlextTypes.Core.JsonObject] = Field(
            default_factory=list,
            exclude=True,
            description="Domain events raised by entity",
        )

        def __eq__(self, other: object) -> bool:
            """Entities are equal if they have same type and ID."""
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id

        def __hash__(self) -> int:
            """Hash based on entity type and ID."""
            return hash((self.__class__, self.id))

        @abstractmethod
        def validate_business_rules(self) -> FlextResult[None]:
            """Validate entity-specific business rules."""

        def add_domain_event(self, event: FlextTypes.Core.JsonObject) -> None:
            """Add domain event to entity."""
            self.domain_events.append(event)

        def clear_domain_events(self) -> list[FlextTypes.Core.JsonObject]:
            """Clear and return all domain events."""
            events = self.domain_events.copy()
            self.domain_events.clear()
            return events

        def increment_version(self) -> None:
            """Increment entity version and update timestamp."""
            self.version += 1
            self.updated_at = datetime.now(UTC)

    class Value(Config, ABC):
        """Immutable value objects with structural equality.

        Value objects are compared by value rather than identity and are
        immutable once created. They encapsulate business logic and validation.
        """

        # Inherit Config settings and add frozen for immutability
        model_config = ConfigDict(
            # Immutability settings - Value objects must be frozen
            frozen=True,
            # Validation settings
            validate_assignment=True,
            validate_default=True,
            use_enum_values=True,
            # JSON settings
            arbitrary_types_allowed=True,
            extra="forbid",
            # Serialization settings
            ser_json_bytes="base64",
            ser_json_timedelta="iso8601",
            # Performance settings
            revalidate_instances="always",
            # String settings
            str_strip_whitespace=True,
            str_to_upper=False,
            str_to_lower=False,
        )

        def __eq__(self, other: object) -> bool:
            """Value objects are equal if all fields are equal."""
            if not isinstance(other, self.__class__):
                return False
            return self.model_dump() == other.model_dump()

        def __hash__(self) -> int:
            """Hash based on all field values."""

            def _make_hashable(obj: object) -> object:
                """Convert non-hashable types to hashable equivalents."""
                if isinstance(obj, dict):
                    return tuple(
                        sorted(
                            (_make_hashable(k), _make_hashable(v))
                            for k, v in obj.items()
                        ),
                    )
                if isinstance(obj, list):
                    return tuple(_make_hashable(item) for item in obj)
                if isinstance(obj, set):
                    return tuple(
                        sorted(
                            (_make_hashable(item) for item in obj),
                            key=str,
                        ),
                    )
                return obj

            model_data = self.model_dump()
            hashable_items = tuple(
                sorted((k, _make_hashable(v)) for k, v in model_data.items()),
            )
            return hash(hashable_items)

        @abstractmethod
        def validate_business_rules(self) -> FlextResult[None]:
            """Validate value object business rules."""

    class AggregateRoot(Entity):
        """Aggregate root managing consistency boundary and domain events.

        Aggregate roots are the entry point for commands and coordinate
        changes across multiple entities within a consistency boundary.
        """

        # Aggregate metadata
        aggregate_type: ClassVar[str] = Field(
            default="",
            description="Type of aggregate",
        )
        aggregate_version: int = Field(
            default=1,
            description="Aggregate schema version",
        )

        def apply_domain_event(
            self,
            event: FlextTypes.Core.JsonObject,
        ) -> FlextResult[None]:
            """Apply domain event to aggregate state."""
            try:
                # Add event to uncommitted events
                self.add_domain_event(event)

                # Apply event to state - safely handle event_type
                event_type = event.get("event_type")
                if event_type and isinstance(event_type, str):
                    handler_name = f"_apply_{event_type.lower()}"
                    if hasattr(self, handler_name):
                        handler = getattr(self, handler_name)
                        handler(event)

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to apply event: {e}")

    # =============================================================================
    # PAYLOAD CLASSES FOR MESSAGING
    # =============================================================================

    class Payload(Config, Generic[T]):
        """Generic type-safe payload container for structured data transport and messaging.

        This class provides a standardized message format for inter-service communication
        within the FLEXT ecosystem. It includes efficient metadata for message
        routing, correlation tracking, expiration handling, and retry management.
        The generic type parameter ensures type safety for payload data.

        """

        # Message metadata
        message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
        correlation_id: str = Field(
            default_factory=lambda: f"corr_{uuid.uuid4().hex[:8]}",
        )
        causation_id: str | None = Field(
            default=None,
            description="ID of causing message",
        )

        # Message timing
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        expires_at: datetime | None = Field(
            default=None,
            description="Message expiration time",
        )

        # Message routing
        source_service: str = Field(..., description="Service that created message")
        target_service: str | None = Field(default=None, description="Target service")
        message_type: str = Field(..., description="Type of message")

        # Actual payload data
        data: T = Field(..., description="Message payload data")

        # Message metadata
        headers: FlextTypes.Core.JsonObject = Field(default_factory=dict)
        priority: int = Field(
            default=5,
            ge=1,
            le=10,
            description="Message priority (1-10)",
        )
        retry_count: int = Field(
            default=0,
            ge=0,
            description="Number of processing attempts",
        )

        @computed_field
        def is_expired(self) -> bool:
            """Check if message has expired."""
            if self.expires_at is None:
                return False
            return datetime.now(UTC) > self.expires_at

        @computed_field
        def age_seconds(self) -> float:
            """Get message age in seconds."""
            return (datetime.now(UTC) - self.timestamp).total_seconds()

    class Message(Payload[FlextTypes.Core.JsonObject]):
        """Structured message container with JSON payload for general-purpose communication.

        This class specializes the generic Payload for JSON-based message transport,
        providing a standardized format for general-purpose inter-service communication.
        It inherits all payload functionality while constraining the data typ

        """

    class Event(Payload[FlextTypes.Core.JsonObject]):
        """Domain event message with structured payload for event sourcing and CQRS patterns.

        This class extends the generic Payload to provide specialized functionality
        for domain events in event-driven architectures. It includes additional
        metadata for aggregate identification, event versioning, and sequence tracking
        to support event sourcing, CQRS, and distributed event processing patterns.

        Key Features:
            - **Aggregate Correlation**: Links events to their originating aggregate
            - **Event Versioning**: Schema evolution support for event structures
            - **Sequence Tracking**: Maintains event ordering within aggregates
            - **Domain Semantics**: Rich metadata for business event processing
            - **Event Sourcing**: Full support for event store patterns
            - **CQRS Integration**: Seamless integration with command/query separation

        Event Sourcing Support:
            Events serve as the source of truth for aggregate state:

            - Each event represents a state change in the domain
            - Events are immutable once created and stored
            - Aggregate state can be rebuilt by replaying events
            - Event sequence ensures consistent state reconstruction
            - Version tracking supports schema evolution

        Aggregate Boundary:
            Events maintain clear aggregate boundaries:

            - **Aggregate ID**: Identifies the specific aggregate instance
            - **Aggregate Type**: Identifies the aggregate class/category
            - **Sequence Number**: Orders events within the aggregate
            - **Event Version**: Supports event schema evolution

        Event Ordering:
            Sequence numbers ensure proper event ordering:

            - Starts at 1 for each aggregate instance
            - Increments for each new event in the aggregate
            - Enables detection of missing or out-of-order events
            - Supports concurrent event processing validation

        Schema Evolution:
            Event versioning supports backward compatibility:

            - Events can evolve their structure over time
            - Version field tracks schema changes
            - Old events remain processable by newer code
            - Event upcasting supported through version detection

        Threading Considerations:
            - Events are immutable after creation
            - Thread-safe for read operations
            - Sequence number assignment requires coordination
            - Aggregate-level synchronization for event ordering

        Performance Characteristics:
            - Efficient event creation with minimal overhead
            - Optimized for append-only event stores
            - Fast aggregate ID indexing support
            - Minimal memory footprint for metadata

        Example Usage::

            # User registration event
            user_registered = FlextModels.Event(
                data={
                    "user_id": "user_123",
                    "email": "john@example.com",
                    "registration_date": "2025-01-15T10:30:00Z",
                },
                message_type="UserRegistered",
                source_service="registration_service",
                aggregate_id="user_123",
                aggregate_type="User",
                sequence_number=1,  # First event for this user
                event_version=1,
            )

            # Order item added event
            item_added = FlextModels.Event(
                data={"item_id": "item_456", "quantity": 2, "unit_price": "29.99"},
                message_type="OrderItemAdded",
                source_service="order_service",
                aggregate_id="order_789",
                aggregate_type="Order",
                sequence_number=3,  # Third event for this order
                event_version=2,  # Updated event schema
            )

        Event Store Integration:
            Events are designed for event store compatibility:

            - Aggregate ID for partitioning and indexing
            - Sequence number for ordering guarantees
            - Version for schema evolution support
            - Timestamp for temporal queries
            - All metadata required for event sourcing

        CQRS Patterns:
            Events serve as the bridge between command and query sides:

            - Commands generate events through aggregates
            - Query models subscribe to events for updates
            - Event handlers maintain read model consistency
            - Event metadata enables distributed processing

        Common Event Types:
            - **Entity Lifecycle**: Created, Updated, Deleted events
            - **State Transitions**: Status changes and workflow events
            - **Business Actions**: User actions and business process events
            - **Integration Events**: Cross-service communication events

        Validation:
            The Event class validates aggregate-specific constraints:

            - Aggregate ID cannot be empty or whitespace
            - Sequence number must be positive
            - Event version must be positive
            - All inherited payload validations apply

        Note:
            This class is specifically designed for domain events in event-driven
            architectures. For general messaging, use the Message class instead.

        """

        # Event-specific fields
        event_version: int = Field(default=1, description="Event schema version")
        aggregate_id: str = Field(..., description="ID of aggregate that raised event")
        aggregate_type: str = Field(..., description="Type of aggregate")
        sequence_number: int = Field(
            default=1,
            ge=1,
            description="Event sequence in aggregate",
        )

        @field_validator("aggregate_id")
        @classmethod
        def validate_aggregate_id(cls, v: str) -> str:
            """Validate aggregate ID is not empty and properly formatted.

            Ensures that the aggregate ID meets the requirements for event sourcing
            and aggregate identification. The ID must be non-empty, non-whitespace,
            and properly trimmed.

            Args:
                v: The aggregate ID string to validate.

            Returns:
                The validated and trimmed aggregate ID.

            Raises:
                ValueError: If the aggregate ID is empty, None, or only whitespace.

            Validation Rules:
                - Must not be None or empty string
                - Must not be only whitespace characters
                - Leading and trailing whitespace is automatically trimmed
                - Must remain non-empty after trimming

            Examples::

                # Valid aggregate IDs
                "user_123"     -> "user_123"
                "  order_456  " -> "order_456"  # Trimmed
                "product-789"  -> "product-789"

                # Invalid aggregate IDs (raise ValueError)
                ""             # Empty string
                "   "          # Only whitespace
                None           # None value

            """
            if not v or not v.strip():
                msg = "Aggregate ID cannot be empty"
                raise ValueError(msg)
            return v.strip()

        @property
        def event_type(self) -> str:
            """Alias for message_type to maintain backward compatibility.

            Returns:
                The event type (message_type).

            """
            return self.message_type

    # =============================================================================
    # ROOTMODEL CLASSES FOR PRIMITIVE VALIDATION
    # =============================================================================

    class EntityId(RootModel[str]):
        """Entity identifier with validation."""

        root: str = Field(
            min_length=1,
            max_length=255,
            description="Non-empty entity identifier",
        )

        @field_validator("root")
        @classmethod
        def validate_not_empty(cls, v: str) -> str:
            """Ensure ID is not empty or whitespace."""
            if not v or not v.strip():
                msg = "Entity ID cannot be empty"
                raise ValueError(msg)
            return v.strip()

    class Version(RootModel[int]):
        """Version number with validation."""

        root: int = Field(ge=1, description="Version number starting from 1")

    class Timestamp(RootModel[datetime]):
        """Timestamp with timezone handling."""

        root: datetime

        @field_validator("root")
        @classmethod
        def ensure_utc(cls, v: datetime) -> datetime:
            """Ensure timestamp is in UTC."""
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v.astimezone(UTC)

    class EmailAddress(RootModel[str]):
        """Email address with validation."""

        root: str = Field(
            pattern=r"^[^@]+@[^@]+\.[^@]+$",
            description="Valid email address",
        )

        @field_validator("root")
        @classmethod
        def validate_email(cls, v: str) -> str:
            """Additional email validation."""
            v = v.strip().lower()
            email_parts = v.split("@")
            expected_email_parts = 2  # local@domain
            if "@" not in v or len(email_parts) != expected_email_parts:
                msg = "Invalid email format"
                raise ValueError(msg)
            local, domain = v.split("@")
            if not local or not domain or "." not in domain:
                msg = "Invalid email format"
                raise ValueError(msg)
            return v

    class Port(RootModel[int]):
        """Network port with validation."""

        root: int = Field(ge=1, le=65535, description="Valid network port (1-65535)")

    class Host(RootModel[str]):
        """Hostname or IP address with validation."""

        root: str = Field(
            min_length=1,
            max_length=255,
            description="Valid hostname or IP",
        )

        @field_validator("root")
        @classmethod
        def validate_host(cls, v: str) -> str:
            """Basic hostname validation."""
            v = v.strip().lower()
            if not v or " " in v:
                msg = "Invalid hostname format"
                raise ValueError(msg)
            return v

    class Url(RootModel[str]):
        """URL with validation."""

        root: str = Field(description="Valid URL")

        @field_validator("root")
        @classmethod
        def validate_url(cls, v: str) -> str:
            """Validate URL format."""
            v = v.strip()
            if not v:
                msg = "URL cannot be empty"
                raise ValueError(msg)

            def _raise_url_error(
                error_msg: str,
                cause: Exception | None = None,
            ) -> None:
                """Abstract raise for URL validation errors."""
                if cause:
                    raise ValueError(error_msg) from cause
                raise ValueError(error_msg)

            try:
                parsed = urlparse(v)
                if not parsed.scheme or not parsed.netloc:
                    _raise_url_error("Invalid URL format")
                return v
            except Exception as e:
                _raise_url_error(f"Invalid URL: {e}", e)
                return v  # This line should never be reached due to the exception

    class JsonData(RootModel[FlextTypes.Core.JsonObject]):
        """JSON data with validation."""

        root: FlextTypes.Core.JsonObject

        @field_validator("root")
        @classmethod
        def validate_json(
            cls,
            v: FlextTypes.Core.JsonObject,
        ) -> FlextTypes.Core.JsonObject:
            """Ensure valid JSON serializable data."""
            try:
                # Test JSON serialization
                json.dumps(v)
                return v
            except (TypeError, ValueError) as e:
                msg = f"Data is not JSON serializable: {e}"
                raise ValueError(msg) from e

    class Metadata(RootModel[dict[str, str]]):
        """String-only metadata with validation."""

        root: dict[str, str] = Field(default_factory=dict)

        @field_validator("root")
        @classmethod
        def validate_string_values(cls, v: dict[str, str]) -> dict[str, str]:
            """Ensure all values are strings."""
            # Type validation is already handled by Pydantic typing
            return v


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FlextModels",
]
