"""Domain models using Pydantic with validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import ClassVar, Generic, Literal, Self
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T


class FlextModels:
    """Consolidated FLEXT model system providing all domain modeling functionality."""

    # =============================================================================
    # BASE MODEL CONFIGURATION
    # =============================================================================

    class Config(BaseModel):
        """Advanced configuration base class for all FLEXT models."""

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
        app_name: str = Field(default="flext-app", description="Application name")
        environment: str = Field(
            default="development",
            description="Application environment",
        )
        debug: bool = Field(
            default=False,
            description="Debug mode flag",
        )
        log_level: str = Field(
            default="INFO",
            description="Logging level",
        )
        max_workers: int = Field(
            default=4,
            ge=1,
            le=100,
            description="Maximum number of workers",
        )
        timeout_seconds: int = Field(
            default=30,
            ge=1,
            le=3600,
            description="Operation timeout in seconds",
        )
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
        secret_fields: FlextTypes.Core.StringList = Field(
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
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if v not in valid_environments:
                msg = f"Invalid environment '{v}'. Valid options: {valid_environments}"
                raise ValueError(msg)
            return v

        @field_validator("config_source")
        @classmethod
        def validate_source(cls, v: str) -> str:
            """Validate source."""
            valid_sources = [
                source.value for source in FlextConstants.Config.ConfigSource
            ]
            if v not in valid_sources:
                msg = f"Invalid source '{v}'. Valid options: {valid_sources}"
                raise ValueError(msg)
            return v

        @field_validator("config_priority")
        @classmethod
        def validate_priority(cls, v: int) -> int:
            """Validate priority."""
            if (
                v < FlextConstants.Config.MIN_PRIORITY
                or v > FlextConstants.Config.MAX_PRIORITY
            ):
                msg = f"Configuration priority must be between {FlextConstants.Config.MIN_PRIORITY} and {FlextConstants.Config.MAX_PRIORITY}"
                raise ValueError(msg)
            return v

        @model_validator(mode="after")
        def validate_production_priority(self) -> Self:
            """Validate production priority."""
            if (
                self.config_environment == "production"
                and self.config_priority > FlextConstants.Config.PRODUCTION_MAX_PRIORITY
            ):
                msg = f"Production configurations should have priority <= {FlextConstants.Config.PRODUCTION_MAX_PRIORITY}"
                raise ValueError(msg)
            return self

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate business rules."""
            try:
                # Update validation tracking only - field validation is handled by Pydantic
                self.validation_count += 1
                self.last_validated = datetime.now(UTC)
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Business rule validation failed: {e}")

        def get_config_info(self) -> FlextTypes.Core.Dict:
            """Get configuration information."""
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
            """Check if production environment."""
            return self.config_environment == "production"

        def is_development(self) -> bool:
            """Check if development environment."""
            return self.config_environment in {"development", "local"}

        def is_test(self) -> bool:
            """Check if test environment."""
            return self.config_environment == "test"

        def is_staging(self) -> bool:
            """Check if staging environment."""
            return self.config_environment == "staging"

        def is_local(self) -> bool:
            """Check if local environment."""
            return self.config_environment == "local"

        def get_environment_config(self) -> FlextTypes.Core.Dict:
            """Get environment-specific settings."""
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

        def mask_secrets(self, data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            """Mask secret fields."""
            if not self.enable_secret_masking:
                return data

            masked_data = data.copy()
            for field in self.secret_fields:
                if field in masked_data:
                    masked_data[field] = "***MASKED***"

            return masked_data

        def get_config_summary(self) -> FlextTypes.Core.Dict:
            """Get configuration summary."""
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
            """Increment validation count."""
            self.validation_count += 1
            self.last_validated = datetime.now(UTC)

        def increment_cache_hits(self) -> None:
            """Increment cache hits."""
            self.validation_cache_hits += 1

        def reset_performance_counters(self) -> None:
            """Reset performance counters."""
            self.validation_count = 0
            self.validation_cache_hits = 0
            self.last_validated = None

    class DatabaseConfig(Config):
        """Database configuration with pooling."""

        # Connection settings
        host: str = Field(..., description="Database host")
        port: int = Field(default=5432, ge=1, le=65535, description="Database port")
        database: str = Field(..., description="Database name")
        username: str = Field(..., description="Database username")
        password: str = Field(..., description="Database password")

        # Override secret fields for database config
        secret_fields: FlextTypes.Core.StringList = Field(
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
            """Validate business rules."""
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
        """Security configuration."""

        # Encryption settings
        secret_key: str = Field(..., description="Application secret key")
        jwt_secret: str = Field(..., description="JWT signing secret")
        encryption_key: str = Field(..., description="Data encryption key")

        # Override secret fields for security config
        secret_fields: FlextTypes.Core.StringList = Field(
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
        cors_origins: FlextTypes.Core.StringList = Field(
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
            """Validate JWT secret."""
            min_length = FlextConstants.Config.MIN_SECRET_LENGTH
            if len(v) < min_length:
                msg = f"JWT secret must be at least {min_length} characters"
                raise ValueError(msg)
            return v

        @field_validator("encryption_key")
        @classmethod
        def validate_encryption_key(cls, v: str) -> str:
            """Validate encryption key."""
            min_length = FlextConstants.Config.MIN_SECRET_LENGTH
            if len(v) < min_length:
                msg = f"Encryption key must be at least {min_length} characters"
                raise ValueError(msg)
            return v

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate business rules."""
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
        """Logging configuration."""

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
        audit_events: FlextTypes.Core.StringList = Field(
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
            """Validate business rules."""
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
        """Mutable entities with identity."""

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
            """Check equality by ID."""
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id

        def __hash__(self) -> int:
            """Hash by type and ID."""
            return hash((self.__class__, self.id))

        @abstractmethod
        def validate_business_rules(self) -> FlextResult[None]:
            """Validate business rules."""

        def add_domain_event(self, event: FlextTypes.Core.JsonObject) -> None:
            """Add domain event."""
            self.domain_events.append(event)

        def clear_domain_events(self) -> list[FlextTypes.Core.JsonObject]:
            """Clear domain events."""
            events = self.domain_events.copy()
            self.domain_events.clear()
            return events

        def increment_version(self) -> None:
            """Increment version."""
            self.version += 1
            self.updated_at = datetime.now(UTC)

    class Value(Config, ABC):
        """Immutable value objects."""

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
                """Convert to hashable."""
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
            """Validate business rules."""

    class AggregateRoot(Entity):
        """Aggregate root for domain events."""

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
            """Apply domain event."""
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
    # HTTP & API-RELATED MODELS
    # =============================================================================

    class Http:
        """Models for HTTP requests, responses, and configurations."""

        class HttpRequestConfig(BaseModel):
            """HTTP Request configuration using Pydantic V2."""

            config_type: Literal["http_request"] = "http_request"
            method: str = "GET"
            url: str
            headers: FlextTypes.Core.Headers = Field(default_factory=dict)
            params: FlextTypes.Core.Dict = Field(default_factory=dict)
            timeout: float = 30.0

        class HttpErrorConfig(BaseModel):
            """HTTP Error configuration using Pydantic V2."""

            config_type: Literal["http_error"] = "http_error"
            message: str
            status_code: int = 500
            url: str | None = None
            method: str | None = None
            headers: FlextTypes.Core.Headers | None = None
            context: FlextTypes.Core.Dict = Field(default_factory=dict)

        class ValidationConfig(BaseModel):
            """Validation configuration using Pydantic V2."""

            config_type: Literal["validation"] = "validation"
            message: str = "Validation error"
            field: str | None = None
            value: object = None
            url: str | None = None
            method: str | None = None
            headers: FlextTypes.Core.Headers | None = None
            context: FlextTypes.Core.Dict = Field(default_factory=dict)

    # =============================================================================
    # PAYLOAD CLASSES FOR MESSAGING
    # =============================================================================

    class Payload(Config, Generic[T]):
        """Type-safe payload container."""

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

        class _CallableBool:
            def __init__(self, *, value: bool) -> None:
                self._value = value

            def __bool__(self) -> bool:  # Truthiness support
                return self._value

            def __call__(self) -> bool:  # Call compatibility
                return self._value

            def __repr__(self) -> str:
                return str(self._value)

        class _CallableFloat:
            def __init__(self, value: float) -> None:
                self._value = value

            def __call__(self) -> float:  # Call compatibility
                return self._value

            def __float__(self) -> float:
                return self._value

            def __repr__(self) -> str:
                return str(self._value)

        @property
        def is_expired(self) -> _CallableBool:
            """Boolean-like value that is also callable for compatibility."""
            value = (
                False
                if self.expires_at is None
                else bool(datetime.now(UTC) > self.expires_at)
            )
            return FlextModels.Payload._CallableBool(value=value)

        @property
        def age_seconds(self) -> _CallableFloat:
            """Float-like value that is also callable for compatibility."""
            value = (datetime.now(UTC) - self.timestamp).total_seconds()
            return FlextModels.Payload._CallableFloat(value)

    class Message(Payload[FlextTypes.Core.JsonObject]):
        """Message container with JSON payload."""

    class Event(Payload[FlextTypes.Core.JsonObject]):
        """Domain event message."""

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
            """Validate aggregate ID."""
            if not v or not v.strip():
                msg = "Aggregate ID cannot be empty"
                raise ValueError(msg)
            return v.strip()

        @property
        def event_type(self) -> str:
            """Alias for message_type."""
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
            """Validate not empty."""
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
            """Ensure UTC."""
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
            """Validate email."""
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

    # =============================================================================
    # SYSTEM CONFIGS - Unified Pydantic models for subsystem configuration
    # =============================================================================

    class SystemConfigs:
        """Unified configuration models for subsystems using Pydantic v2.

        These models are the single validation point for configuration parameters
        across subsystems. They centralize normalization and business rules and
        should be used internally, with dicts exposed only at integration borders
        via `model_dump()`.
        """

        class BaseSystemConfig(BaseModel):
            """Base system config with common fields and validation.

            Provides normalized environment/log level validation using
            FlextConstants StrEnums and shared defaults.
            """

            model_config = ConfigDict(
                validate_assignment=True,
                extra="ignore",
                use_enum_values=True,
                str_strip_whitespace=True,
            )

            environment: str = Field(
                default=FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                description="Runtime environment",
            )
            log_level: str = Field(
                default=FlextConstants.Config.LogLevel.INFO.value,
                description="Logging level",
            )
            validation_level: str | None = Field(
                default=None,
                description="Validation strictness level",
            )

            @field_validator("environment")
            @classmethod
            def _validate_environment(cls, v: str) -> str:
                mapping = {
                    "dev": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                    "prod": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
                    "stage": FlextConstants.Config.ConfigEnvironment.STAGING.value,
                    "stg": FlextConstants.Config.ConfigEnvironment.STAGING.value,
                }
                normalized = mapping.get(v.lower(), v)
                valid = {e.value for e in FlextConstants.Config.ConfigEnvironment}
                if normalized not in valid:
                    msg = f"Invalid environment '{v}'. Valid: {sorted(valid)}"
                    raise ValueError(msg)
                return normalized

            @field_validator("log_level")
            @classmethod
            def _validate_log_level(cls, v: str) -> str:
                upper = v.upper()
                valid = {e.value for e in FlextConstants.Config.LogLevel}
                if upper not in valid:
                    msg = f"Invalid log_level '{v}'. Valid: {sorted(valid)}"
                    raise ValueError(msg)
                return upper

            @field_validator("validation_level")
            @classmethod
            def _validate_validation_level(cls, v: str | None) -> str | None:
                if v is None:
                    return None
                valid = {e.value for e in FlextConstants.Config.ValidationLevel}
                if v not in valid:
                    msg = f"Invalid validation_level '{v}'. Valid: {sorted(valid)}"
                    raise ValueError(msg)
                return v

        class CommandsConfig(BaseSystemConfig):
            """Configuration for Commands subsystem."""

            enable_handler_discovery: bool = Field(default=True)
            enable_middleware_pipeline: bool = Field(default=True)
            enable_performance_monitoring: bool = Field(default=False)
            max_concurrent_commands: int = Field(default=100, ge=1, le=1000)
            command_timeout_seconds: int = Field(default=30, ge=1, le=300)

            @model_validator(mode="after")
            def validate_production_settings(self) -> Self:
                """Adjust settings for production."""
                if (
                    self.environment
                    == FlextConstants.Config.ConfigEnvironment.PRODUCTION
                    and self.enable_performance_monitoring is False
                ):
                    self.enable_performance_monitoring = True
                return self

        class CommandModel(BaseModel):
            """Base model for CQRS commands."""

            model_config = ConfigDict(
                frozen=True,  # Commands are immutable
                validate_assignment=True,
                extra="ignore",
                use_enum_values=True,
            )

            command_id: str = Field(
                default_factory=lambda: f"cmd_{uuid.uuid4().hex[:12]}",
                description="Unique command identifier",
            )
            command_type: str = Field(..., description="Type of command")
            correlation_id: str = Field(
                default_factory=lambda: f"corr_{uuid.uuid4().hex[:8]}",
                description="Correlation ID for tracking",
            )
            created_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="Command creation timestamp",
            )
            user_id: str | None = Field(
                default=None, description="User who issued command"
            )
            metadata: FlextTypes.Core.Headers = Field(
                default_factory=dict, description="Command metadata"
            )

            def validate_command(self) -> FlextResult[None]:
                """Validate command before execution."""
                return FlextResult[None].ok(None)

        class QueryModel(BaseModel):
            """Base model for CQRS queries."""

            model_config = ConfigDict(
                frozen=True,  # Queries are immutable
                validate_assignment=True,
                extra="ignore",
                use_enum_values=True,
            )

            query_id: str = Field(
                default_factory=lambda: f"qry_{uuid.uuid4().hex[:12]}",
                description="Unique query identifier",
            )
            query_type: str = Field(..., description="Type of query")
            created_at: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="Query creation timestamp",
            )
            user_id: str | None = Field(
                default=None, description="User who issued query"
            )
            filters: FlextTypes.Core.Dict = Field(
                default_factory=dict, description="Query filters"
            )
            pagination: FlextTypes.Core.CounterDict | None = Field(
                default=None, description="Pagination params"
            )

            def validate_query(self) -> FlextResult[None]:
                """Validate query before execution."""
                return FlextResult[None].ok(None)

        class HandlerConfig(BaseModel):
            """Configuration for command/query handlers."""

            model_config = ConfigDict(
                validate_assignment=True,
                extra="ignore",
            )

            handler_id: str = Field(..., description="Unique handler identifier")
            handler_name: str = Field(..., description="Human-readable handler name")
            handler_type: str = Field(
                default="command",
                pattern="^(command|query|event)$",
                description="Type of handler",
            )
            can_retry: bool = Field(
                default=True, description="Whether handler can be retried"
            )
            max_retries: int = Field(
                default=3, ge=0, le=10, description="Max retry attempts"
            )
            timeout_seconds: int = Field(
                default=30, ge=1, le=300, description="Handler timeout"
            )
            priority: int = Field(
                default=5, ge=1, le=10, description="Handler priority"
            )

        class BusConfig(BaseModel):
            """Configuration for command/query bus."""

            model_config = ConfigDict(
                validate_assignment=True,
                extra="ignore",
            )

            bus_id: str = Field(
                default_factory=lambda: f"bus_{uuid.uuid4().hex[:8]}",
                description="Bus instance identifier",
            )
            enable_middleware: bool = Field(
                default=True, description="Enable middleware pipeline"
            )
            enable_async: bool = Field(
                default=False, description="Enable async processing"
            )
            max_queue_size: int = Field(
                default=1000, ge=1, le=10000, description="Max queue size"
            )
            worker_threads: int = Field(
                default=4, ge=1, le=32, description="Worker thread count"
            )
            enable_metrics: bool = Field(
                default=True, description="Enable metrics collection"
            )
            enable_tracing: bool = Field(
                default=True, description="Enable distributed tracing"
            )

        class MiddlewareConfig(BaseModel):
            """Configuration for middleware components."""

            model_config = ConfigDict(
                validate_assignment=True,
                extra="ignore",
            )

            middleware_id: str = Field(..., description="Middleware identifier")
            middleware_type: str = Field(..., description="Type of middleware")
            enabled: bool = Field(
                default=True, description="Whether middleware is enabled"
            )
            order: int = Field(
                default=0, description="Execution order (lower = earlier)"
            )
            config: FlextTypes.Core.Dict = Field(
                default_factory=dict, description="Middleware config"
            )

        class DomainServicesConfig(BaseSystemConfig):
            """Configuration for Domain Services."""

            service_level: str = Field(
                default="standard", pattern="^(basic|standard|premium)$"
            )
            enable_caching: bool = Field(default=False)
            cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
            max_retry_attempts: int = Field(default=3, ge=0, le=10)

            @field_validator("cache_ttl_seconds")
            @classmethod
            def validate_cache_when_enabled(cls, v: int, info: object) -> int:
                """Validate TTL when cache is enabled."""
                data = getattr(info, "data", {}) if hasattr(info, "data") else {}
                if data.get("enable_caching") and v == 0:
                    msg = "cache_ttl_seconds must be > 0 when cache is enabled"
                    raise ValueError(msg)
                return v

        class CoreConfig(BaseSystemConfig):
            """Configuration for Core system."""

            enable_debug_mode: bool = Field(default=False)
            enable_profiling: bool = Field(default=False)
            max_recursion_depth: int = Field(default=100, ge=10, le=1000)

        class ServicesConfig(BaseSystemConfig):
            """Configuration for Services."""

            service_name: str = Field(default="flext-service")
            enable_health_checks: bool = Field(default=True)
            health_check_interval_seconds: int = Field(default=30, ge=5, le=300)

        class ValidationSystemConfig(BaseSystemConfig):
            """Configuration for Validation System."""

            enable_detailed_errors: bool = Field(default=True)
            max_validation_errors: int = Field(default=10, ge=1, le=100)
            validation_timeout_ms: int = Field(default=1000, ge=100, le=10000)

        class TypeAdaptersConfig(BaseSystemConfig):
            """Configuration for Type Adapters."""

            strict_mode: bool = Field(default=False)
            coerce_types: bool = Field(default=True)
            validate_assignment: bool = Field(default=True)

        class ProtocolsConfig(BaseSystemConfig):
            """Configuration for Protocols."""

            enable_protocol_validation: bool = Field(default=True)
            protocol_timeout_seconds: int = Field(default=30, ge=1, le=300)

        class MixinsConfig(BaseSystemConfig):
            """Configuration for Mixins."""

            enable_serialization: bool = Field(default=True)
            enable_timestamps: bool = Field(default=True)
            enable_versioning: bool = Field(default=False)

        class GuardsConfig(BaseSystemConfig):
            """Configuration for Guards."""

            enable_type_guards: bool = Field(default=True)
            enable_permission_guards: bool = Field(default=True)
            guard_timeout_seconds: int = Field(default=5, ge=1, le=60)

        class ProcessorsConfig(BaseSystemConfig):
            """Configuration for Processors."""

            enable_pipeline_processing: bool = Field(default=True)
            max_pipeline_depth: int = Field(default=10, ge=1, le=100)
            enable_async_processing: bool = Field(default=False)

        class ContextConfig(BaseSystemConfig):
            """Configuration for Context."""

            enable_context_propagation: bool = Field(default=True)
            context_timeout_seconds: int = Field(default=60, ge=1, le=600)

        class DelegationConfig(BaseSystemConfig):
            """Configuration for Delegation."""

            enable_delegation: bool = Field(default=True)
            max_delegation_depth: int = Field(default=5, ge=1, le=20)

        class FieldsConfig(BaseSystemConfig):
            """Configuration for Fields."""

            enable_field_validation: bool = Field(default=True)
            enable_field_transformation: bool = Field(default=True)
            max_field_size: int = Field(default=1048576, ge=1024, le=104857600)

        class ContainerConfig(BaseSystemConfig):
            """Configuration for Dependency Injection Container."""

            max_services: int = Field(
                default=1000,
                ge=1,
                le=10000,
                description="Maximum number of services that can be registered",
            )
            service_timeout: int = Field(
                default=30000,
                ge=1000,
                le=300000,
                description="Service resolution timeout in milliseconds",
            )
            enable_auto_wire: bool = Field(
                default=True,
                description="Enable automatic dependency wiring",
            )
            enable_factory_cache: bool = Field(
                default=True,
                description="Cache factory-created services for singleton behavior",
            )
            enable_lazy_loading: bool = Field(
                default=True,
                description="Enable lazy service loading",
            )
            enable_service_validation: bool = Field(
                default=True,
                description="Validate services on registration",
            )
            config_source: str = Field(
                default=FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
                description="Configuration source",
            )

            @field_validator("config_source")
            @classmethod
            def _validate_config_source(cls, v: str) -> str:
                valid = {e.value for e in FlextConstants.Config.ConfigSource}
                if v not in valid:
                    msg = f"Invalid config_source '{v}'. Valid: {sorted(valid)}"
                    raise ValueError(msg)
                return v

            @model_validator(mode="after")
            def validate_production_settings(self) -> Self:
                """Adjust settings for production."""
                if (
                    self.environment
                    == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
                ):
                    # Production should have higher service timeout - use object.__setattr__ to avoid recursion
                    object.__setattr__(
                        self, "service_timeout", max(self.service_timeout, 60000)
                    )
                    # Production should validate services
                    if not self.enable_service_validation:
                        object.__setattr__(self, "enable_service_validation", True)
                return self

        class HandlersConfig(BaseSystemConfig):
            """Configuration for Handlers subsystem."""

            timeout: int = Field(
                default=30000,
                ge=1000,
                le=300000,
                description="Handler timeout in milliseconds",
            )
            max_retries: int = Field(
                default=3, ge=0, le=10, description="Maximum retry attempts"
            )
            enable_metrics: bool = Field(
                default=True, description="Enable metrics collection"
            )
            enable_tracing: bool = Field(
                default=False, description="Enable execution tracing"
            )
            slow_handler_threshold_ms: int = Field(
                default=1000,
                ge=100,
                le=30000,
                description="Threshold for slow handler warnings",
            )

        class BasicHandlerConfig(BaseSystemConfig):
            """Configuration for BasicHandler."""

            handler_name: str | None = Field(
                default=None, description="Handler identifier"
            )
            timeout: int = Field(
                default=30000,
                ge=1000,
                le=300000,
                description="Execution timeout in milliseconds",
            )
            max_retries: int = Field(
                default=3, ge=0, le=10, description="Maximum retry attempts"
            )
            enable_metrics: bool = Field(
                default=True, description="Enable metrics collection"
            )

        class ValidatingHandlerConfig(BasicHandlerConfig):
            """Configuration for ValidatingHandler."""

            enable_validation: bool = Field(
                default=True, description="Enable request validation"
            )
            validation_strict: bool = Field(
                default=False, description="Use strict validation mode"
            )
            max_validation_errors: int = Field(
                default=10,
                ge=1,
                le=100,
                description="Maximum validation errors to collect",
            )

        class AuthorizingHandlerConfig(BasicHandlerConfig):
            """Configuration for AuthorizingHandler."""

            enable_authorization: bool = Field(
                default=True, description="Enable authorization checks"
            )
            authorization_mode: str = Field(
                default="role_based",
                pattern="^(role_based|permission_based|custom)$",
                description="Authorization mode",
            )
            cache_authorization: bool = Field(
                default=True, description="Cache authorization results"
            )

        class MetricsHandlerConfig(BasicHandlerConfig):
            """Configuration for MetricsHandler."""

            collect_request_sizes: bool = Field(
                default=True, description="Collect request size metrics"
            )
            collect_error_types: bool = Field(
                default=True, description="Collect error type metrics"
            )
            metrics_buffer_size: int = Field(
                default=1000,
                ge=100,
                le=10000,
                description="Size of metrics buffer",
            )

        class EventHandlerConfig(BaseSystemConfig):
            """Configuration for EventHandler."""

            handler_name: str | None = Field(
                default=None, description="Event handler identifier"
            )
            enable_event_replay: bool = Field(
                default=False, description="Enable event replay capability"
            )
            event_buffer_size: int = Field(
                default=1000,
                ge=100,
                le=10000,
                description="Event buffer size",
            )

        class HandlerChainConfig(BaseSystemConfig):
            """Configuration for HandlerChain."""

            chain_name: str | None = Field(default=None, description="Chain identifier")
            stop_on_error: bool = Field(
                default=True, description="Stop chain execution on error"
            )
            parallel_execution: bool = Field(
                default=False, description="Execute handlers in parallel"
            )
            max_handlers: int = Field(
                default=10, ge=1, le=50, description="Maximum handlers in chain"
            )

        class PipelineConfig(BaseSystemConfig):
            """Configuration for Pipeline."""

            pipeline_name: str | None = Field(
                default=None, description="Pipeline identifier"
            )
            enable_stage_metrics: bool = Field(
                default=True, description="Collect per-stage metrics"
            )
            max_stages: int = Field(
                default=10, ge=1, le=50, description="Maximum pipeline stages"
            )
            stage_timeout_ms: int = Field(
                default=5000,
                ge=100,
                le=60000,
                description="Per-stage timeout in milliseconds",
            )

        class HandlerRegistryConfig(BaseSystemConfig):
            """Configuration for HandlerRegistry."""

            enable_auto_discovery: bool = Field(
                default=False, description="Enable automatic handler discovery"
            )
            enable_lazy_loading: bool = Field(
                default=True, description="Enable lazy handler loading"
            )
            max_handlers: int = Field(
                default=100,
                ge=10,
                le=1000,
                description="Maximum registered handlers",
            )

    class Metadata(RootModel[FlextTypes.Core.Headers]):
        """String-only metadata with validation."""

        root: FlextTypes.Core.Headers = Field(default_factory=dict)

        @field_validator("root")
        @classmethod
        def validate_string_values(
            cls, v: FlextTypes.Core.Headers
        ) -> FlextTypes.Core.Headers:
            """Validate string values."""
            # Type validation is already handled by Pydantic typing
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
            """Validate hostname."""
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
            """Validate URL."""
            v = v.strip()
            if not v:
                msg = "URL cannot be empty"
                raise ValueError(msg)

            def _raise_url_error(
                error_msg: str,
                cause: Exception | None = None,
            ) -> None:
                """Raise URL error."""
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
            """Validate JSON serializable."""
            try:
                # Test JSON serialization
                json.dumps(v)
                return v
            except (TypeError, ValueError) as e:
                msg = f"Data is not JSON serializable: {e}"
                raise ValueError(msg) from e


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FlextModels",
]
