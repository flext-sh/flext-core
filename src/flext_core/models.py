"""FlextModels - SIMPLE domain models for FLEXT ecosystem.

NO over-engineering. Just what's actually needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar, cast

from pydantic import BaseModel, Field

from flext_core.result import FlextResult
from flext_core.utilities import FlextUtilities

T = TypeVar("T")


class FlextModels:
    """Simple, useful domain models. No DDD bullshit."""

    class TimestampedModel(BaseModel):
        """Base model with timestamps - SIMPLE."""

        created_at: datetime = Field(default_factory=datetime.now)
        updated_at: datetime = Field(default_factory=datetime.now)

    class Entity(BaseModel):
        """Entity with ID - that's it."""

        id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        version: int = Field(default=1)
        domain_events: list[FlextModels.Event] = Field(default_factory=list)

        def add_domain_event(self, event: FlextModels.Event) -> None:
            """Add domain event."""
            self.domain_events.append(event)

        def clear_domain_events(self) -> list[FlextModels.Event]:
            """Clear and return domain events."""
            events = self.domain_events.copy()
            self.domain_events.clear()
            return events

        def increment_version(self) -> None:
            """Increment version."""
            self.version += 1

    class Value(BaseModel):
        """Immutable value object."""

        class Config:
            """Configuration for Value objects."""

            frozen = True

    class Payload(BaseModel, Generic[T]):
        """Generic payload wrapper - actually useful."""

        data: T = Field(..., description="Payload data")
        metadata: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=datetime.now)
        message_type: str = Field(default="", description="Message type identifier")
        source_service: str = Field(default="", description="Source service name")
        correlation_id: str = Field(
            default="", description="Correlation ID for tracking"
        )
        message_id: str = Field(default="", description="Unique message identifier")

        def extract(self) -> T:
            """Extract payload data."""
            return self.data

    class Event(BaseModel):
        """Simple event for notifications."""

        event_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        event_type: str = Field(...)
        payload: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=datetime.now)

    class Command(BaseModel):
        """Simple command."""

        command_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        command_type: str = Field(...)
        payload: dict[str, object] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=datetime.now)
        correlation_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_id
        )
        user_id: str | None = Field(
            default=None, description="User ID associated with the command"
        )

        def validate_command(self) -> FlextResult[bool]:
            """Validate command data."""
            if not self.command_type:
                return FlextResult[bool].fail("Command type is required")
            return FlextResult[bool].ok(data=True)

    class Query(BaseModel):
        """Simple query."""

        query_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        query_type: str = Field(...)
        filters: dict[str, object] = Field(default_factory=dict)
        pagination: dict[str, int] = Field(
            default_factory=lambda: {"page": 1, "size": 10}
        )
        user_id: str | None = Field(
            default=None, description="User ID associated with the query"
        )

        def validate_query(self) -> FlextResult[bool]:
            """Validate query data."""
            if not self.query_type:
                return FlextResult[bool].fail("Query type is required")
            return FlextResult[bool].ok(data=True)

    class EmailAddress(Value):
        """Email address value object for examples."""

        value: str

        @classmethod
        def create(cls, email: str) -> FlextResult[FlextModels.EmailAddress]:
            """Create email address with validation."""
            if "@" not in email or "." not in email:
                return FlextResult["FlextModels.EmailAddress"].fail(
                    "Invalid email format"
                )
            return FlextResult["FlextModels.EmailAddress"].ok(cls(value=email))

        def domain(self) -> str:
            """Get email domain."""
            return self.value.split("@")[1] if "@" in self.value else ""

    # AggregateRoot for compatibility - but SIMPLE
    class AggregateRoot(Entity):
        """Aggregate root - just an entity with version."""

        version: int = Field(default=1)
        aggregate_type: str = Field(default="")

    # =========================================================================
    # CONFIGURATION CLASSES - Simple configuration models
    # =========================================================================

    class SystemConfigs:
        """System-wide configuration classes."""

        class ContainerConfig(BaseModel):
            """Container configuration."""

            max_services: int = Field(
                default=100, description="Maximum number of services"
            )
            enable_caching: bool = Field(
                default=True, description="Enable service caching"
            )
            cache_ttl: int = Field(
                default=300, description="Cache time-to-live in seconds"
            )
            enable_monitoring: bool = Field(
                default=False, description="Enable monitoring"
            )

        class DatabaseConfig(BaseModel):
            """Database configuration."""

            host: str = Field(default="localhost", description="Database host")
            port: int = Field(default=5432, description="Database port")
            name: str = Field(default="flext_db", description="Database name")
            user: str = Field(default="flext_user", description="Database user")
            password: str = Field(default="", description="Database password")
            ssl_mode: str = Field(default="prefer", description="SSL mode")
            connection_timeout: int = Field(
                default=30, description="Connection timeout"
            )
            max_connections: int = Field(default=20, description="Maximum connections")

        class SecurityConfig(BaseModel):
            """Security configuration."""

            enable_encryption: bool = Field(
                default=True, description="Enable encryption"
            )
            encryption_key: str = Field(default="", description="Encryption key")
            enable_audit: bool = Field(
                default=False, description="Enable audit logging"
            )
            session_timeout: int = Field(
                default=3600, description="Session timeout in seconds"
            )
            password_policy: dict[str, object] = Field(
                default_factory=lambda: cast("dict[str, object]", {
                    "min_length": 8,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_digits": True,
                    "require_special": False,
                })
            )

        class LoggingConfig(BaseModel):
            """Logging configuration."""

            level: str = Field(default="INFO", description="Log level")
            format: str = Field(
                default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                description="Log format",
            )
            file_path: str = Field(default="", description="Log file path")
            max_file_size: int = Field(
                default=10485760, description="Max log file size in bytes"
            )
            backup_count: int = Field(default=5, description="Number of backup files")
            enable_console: bool = Field(
                default=True, description="Enable console logging"
            )

        class MiddlewareConfig(BaseModel):
            """Middleware configuration."""

            middleware_type: str = Field(..., description="Type of middleware")
            middleware_id: str = Field(default="", description="Unique middleware ID")
            order: int = Field(default=0, description="Execution order")
            enabled: bool = Field(
                default=True, description="Whether middleware is enabled"
            )
            config: dict[str, object] = Field(
                default_factory=dict, description="Middleware-specific configuration"
            )

    # =========================================================================
    # SIMPLE CONFIG CLASSES - Direct access versions
    # =========================================================================

    # Alias for backward compatibility and direct access
    DatabaseConfig = SystemConfigs.DatabaseConfig
    SecurityConfig = SystemConfigs.SecurityConfig
    LoggingConfig = SystemConfigs.LoggingConfig
    MiddlewareConfig = SystemConfigs.MiddlewareConfig

    # =========================================================================
    # EXAMPLE-SPECIFIC CLASSES - For examples and demos
    # =========================================================================

    class Config(BaseModel):
        """Simple configuration class for examples."""

        name: str = Field(default="", description="Configuration name")
        enabled: bool = Field(default=True, description="Whether enabled")
        settings: dict[str, object] = Field(
            default_factory=dict, description="Additional settings"
        )

    class Message(BaseModel):
        """Simple message class for examples."""

        message_id: str = Field(default_factory=FlextUtilities.Generators.generate_id)
        content: str = Field(...)
        message_type: str = Field(default="info")
        priority: str = Field(default="normal")
        target_service: str = Field(default="")
        headers: dict[str, str] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=datetime.now)
        source_service: str = Field(default="")
        aggregate_id: str = Field(default="")
        aggregate_type: str = Field(default="")

    # Simple factory methods - no over-engineering
    @staticmethod
    def create_entity(**data: object) -> FlextResult[FlextModels.Entity]:
        """Create an entity."""
        try:
            # Convert object values to appropriate types
            id_value = ""

            for key, value in data.items():
                if key == "id":
                    if isinstance(value, str):
                        id_value = value
                    elif value is not None:
                        id_value = str(value)

            entity = FlextModels.Entity(id=id_value)
            return FlextResult[FlextModels.Entity].ok(entity)
        except Exception as e:
            return FlextResult[FlextModels.Entity].fail(str(e))

    @staticmethod
    def create_event(event_type: str, payload: dict[str, object]) -> Event:
        """Create an event."""
        return FlextModels.Event(event_type=event_type, payload=payload)

    @staticmethod
    def create_command(command_type: str, payload: dict[str, object]) -> Command:
        """Create a command."""
        return FlextModels.Command(command_type=command_type, payload=payload)

    @staticmethod
    def create_query(
        query_type: str, filters: dict[str, object] | None = None
    ) -> Query:
        """Create a query."""
        return FlextModels.Query(query_type=query_type, filters=filters or {})

    class Http:
        """HTTP-related models for API configuration."""

        class HttpRequestConfig(BaseModel):
            """Configuration for HTTP requests."""

            config_type: str = Field(default="http_request")
            url: str = Field(min_length=1)
            method: str = Field(default="GET")
            timeout: int = Field(default=30)
            retries: int = Field(default=3)
            headers: dict[str, str] = Field(default_factory=dict)

        class HttpErrorConfig(BaseModel):
            """Configuration for HTTP error handling."""

            config_type: str = Field(default="http_error")
            status_code: int = Field(ge=100, le=599)
            message: str = Field(min_length=1)
            url: str | None = Field(default=None)
            method: str | None = Field(default=None)
            headers: dict[str, str] | None = Field(default=None)
            context: dict[str, object] = Field(default_factory=dict)
            details: dict[str, object] = Field(default_factory=dict)

        class ValidationConfig(BaseModel):
            """Configuration for validation."""

            config_type: str = Field(default="validation")
            strict_mode: bool = Field(default=False)
            validate_schema: bool = Field(default=True)
            custom_validators: list[str] = Field(default_factory=list)
            field: str | None = Field(default=None)
            value: object | None = Field(default=None)
            url: str | None = Field(default=None)
            method: str | None = Field(default=None)
            headers: dict[str, str] | None = Field(default=None)
            context: dict[str, object] = Field(default_factory=dict)
            message: str = Field(default="Validation error")
