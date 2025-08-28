"""FLEXT Models - Complete consolidated model system using Pydantic RootModel.

CONSOLIDAÇÃO COMPLETA seguindo solicitação do usuário:
- Apenas UMA classe FlextModels com toda funcionalidade
- Baseado em Pydantic RootModel para validação de primitivos
- Todas as outras classes antigas movidas para legacy.py como facades
- Arquitetura hierárquica limpa seguindo padrão FLEXT

Architecture Overview:
    FlextModels - Single consolidated class containing:
        - Nested classes for different model types (Entity, Value, etc.)
        - RootModel classes for primitive validation
        - Factory methods for creating instances
        - Validation utilities and helpers

Examples:
    Using consolidated FlextModels:
        user = FlextModels.Entity(id="user_123", name="John")
        email = FlextModels.EmailAddress(root="test@example.com")
        port = FlextModels.Port(root=8080)

    Factory methods:
        user = FlextModels.create_entity({"id": "user_123", "name": "John"})
        value = FlextModels.create_value_object({"address": "test@example.com"})

Note:
    This is the ONLY model module following complete consolidation.
    All legacy classes are now facades in legacy.py pointing to FlextModels.

"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import urlparse

if TYPE_CHECKING:
    from typing import TypeAlias

from dateutil import parser
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    computed_field,
    field_validator,
)

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextModels:
    """CONSOLIDAÇÃO COMPLETA - Única classe de modelo com toda funcionalidade do sistema.

    Seguindo padrão Pydantic RootModel para validação primitiva e arquitetura
    hierárquica FLEXT com nested classes organizadas por funcionalidade.

    Architecture:
        - Base: Classes base com configuração comum
        - Domain: Entities, ValueObjects, AggregateRoots
        - Validation: RootModel classes para primitivos
        - Factory: Métodos de criação e utilities
        - Payload: Mensagens e eventos estruturados
    """

    # Type aliases for better readability in factory methods

    # =============================================================================
    # BASE MODEL CONFIGURATION
    # =============================================================================

    class BaseConfig(BaseModel):
        """Base configuration for all FLEXT models."""

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
        )

    # =============================================================================
    # DOMAIN MODEL CLASSES
    # =============================================================================

    class Entity(BaseConfig, ABC):
        """Mutable entities with identity, versioning and domain events.

        Entities have identity that persists across state changes and support
        domain events, versioning, and lifecycle management.
        """

        # Core identity fields
        id: str = Field(..., description="Unique entity identifier")
        version: int = Field(
            default=1, description="Entity version for optimistic locking"
        )

        # Metadata fields
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        created_by: str | None = Field(
            default=None, description="User who created entity"
        )
        updated_by: str | None = Field(
            default=None, description="User who last updated entity"
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

    class Value(BaseConfig, ABC):
        """Immutable value objects with structural equality.

        Value objects are compared by value rather than identity and are
        immutable once created. They encapsulate business logic and validation.
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
            # Value object specific
            frozen=True,  # Make immutable
        )

        def __eq__(self, other: object) -> bool:
            """Value objects are equal if all fields are equal."""
            if not isinstance(other, self.__class__):
                return False
            return self.model_dump() == other.model_dump()

        def __hash__(self) -> int:
            """Hash based on all field values."""
            return hash(tuple(sorted(self.model_dump().items())))

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
            default="", description="Type of aggregate"
        )
        aggregate_version: int = Field(
            default=1, description="Aggregate schema version"
        )

        def apply_domain_event(
            self, event: FlextTypes.Core.JsonObject
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

    class Payload[T](BaseConfig):
        """Generic type-safe payload container for structured data transport.

        Provides standardized message format with metadata, correlation IDs,
        and type-safe payload handling for inter-service communication.
        """

        # Message metadata
        message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
        correlation_id: str = Field(
            default_factory=lambda: f"corr_{uuid.uuid4().hex[:8]}"
        )
        causation_id: str | None = Field(
            default=None, description="ID of causing message"
        )

        # Message timing
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        expires_at: datetime | None = Field(
            default=None, description="Message expiration time"
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
            default=5, ge=1, le=10, description="Message priority (1-10)"
        )
        retry_count: int = Field(
            default=0, ge=0, description="Number of processing attempts"
        )

        @property
        @computed_field
        def is_expired(self) -> bool:
            """Check if message has expired."""
            if self.expires_at is None:
                return False
            return datetime.now(UTC) > self.expires_at

        @property
        @computed_field
        def age_seconds(self) -> float:
            """Get message age in seconds."""
            return (datetime.now(UTC) - self.timestamp).total_seconds()

    class Message(Payload[FlextTypes.Core.JsonObject]):
        """Structured message with JSON payload."""

    class Event(Payload[FlextTypes.Core.JsonObject]):
        """Domain event with structured payload."""

        # Event-specific fields
        event_version: int = Field(default=1, description="Event schema version")
        aggregate_id: str = Field(..., description="ID of aggregate that raised event")
        aggregate_type: str = Field(..., description="Type of aggregate")
        sequence_number: int = Field(
            default=1, ge=1, description="Event sequence in aggregate"
        )

        @field_validator("aggregate_id")
        @classmethod
        def validate_aggregate_id(cls, v: str) -> str:
            """Validate aggregate ID is not empty."""
            if not v or not v.strip():
                msg = "Aggregate ID cannot be empty"
                raise ValueError(msg)
            return v.strip()

    # =============================================================================
    # ROOTMODEL CLASSES FOR PRIMITIVE VALIDATION
    # =============================================================================

    class EntityId(RootModel[str]):
        """Entity identifier with validation."""

        root: str = Field(
            min_length=1, max_length=255, description="Non-empty entity identifier"
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
            pattern=r"^[^@]+@[^@]+\.[^@]+$", description="Valid email address"
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
            min_length=1, max_length=255, description="Valid hostname or IP"
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
                error_msg: str, cause: Exception | None = None
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
            cls, v: FlextTypes.Core.JsonObject
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
            for key, value in v.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    msg = "All metadata keys and values must be strings"
                    raise TypeError(msg)
            return v

    # =============================================================================
    # FACTORY METHODS AND UTILITIES
    # =============================================================================

    @classmethod
    def create_entity(
        cls,
        data: FlextTypes.Core.JsonObject,
        entity_class: type[FlextModels.Entity] | None = None,
    ) -> FlextResult[FlextModels.Entity]:
        """Create entity instance with validation."""
        try:
            if entity_class is None:
                entity_class = cls.Entity

            # Convert data to proper types for entity creation
            entity_data = dict(data)

            # Ensure required fields
            if "id" not in entity_data:
                entity_data["id"] = f"entity_{uuid.uuid4().hex[:12]}"

            entity = entity_class(**entity_data)

            # Validate business rules
            validation_result = entity.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[cls.Entity].fail(
                    f"Business rule validation failed: {validation_result.error}"
                )

            return FlextResult[cls.Entity].ok(entity)

        except ValidationError as e:
            return FlextResult[cls.Entity].fail(f"Entity validation failed: {e}")
        except Exception as e:
            return FlextResult[cls.Entity].fail(f"Entity creation failed: {e}")

    @classmethod
    def create_value_object(
        cls,
        data: FlextTypes.Core.JsonObject,
        value_class: type[Value] | None = None,
    ) -> FlextResult[Value]:
        """Create value object instance with validation."""
        try:
            if value_class is None:
                value_class = cls.Value

            # Convert data to proper types for value object creation
            value_data = dict(data)
            value_obj = value_class(**value_data)

            # Validate business rules
            validation_result = value_obj.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[Value].fail(
                    f"Business rule validation failed: {validation_result.error}"
                )

            return FlextResult[Value].ok(value_obj)

        except ValidationError as e:
            return FlextResult[Value].fail(f"Value object validation failed: {e}")
        except Exception as e:
            return FlextResult[Value].fail(f"Value object creation failed: {e}")

    @classmethod
    def create_payload[T](
        cls,
        data: T,
        message_type: str,
        source_service: str,
        target_service: str | None = None,
        correlation_id: str | None = None,
    ) -> FlextResult[Payload[T]]:
        """Create payload instance with proper metadata."""
        try:
            payload_data = {
                "data": data,
                "message_type": message_type,
                "source_service": source_service,
            }

            if target_service:
                payload_data["target_service"] = target_service
            if correlation_id:
                payload_data["correlation_id"] = correlation_id

            payload = cls.Payload[T](**payload_data)
            return FlextResult[Payload[T]].ok(payload)

        except ValidationError as e:
            return FlextResult[Payload[T]].fail(f"Payload validation failed: {e}")
        except Exception as e:
            return FlextResult[Payload[T]].fail(f"Payload creation failed: {e}")

    @classmethod
    def create_domain_event(
        cls,
        event_type: str,
        aggregate_id: str,
        aggregate_type: str,
        data: FlextTypes.Core.JsonObject,
        source_service: str,
        sequence_number: int = 1,
    ) -> FlextResult[Event]:
        """Create domain event with proper structure."""
        try:
            event_data = {
                "data": data,
                "message_type": event_type,
                "source_service": source_service,
                "aggregate_id": aggregate_id,
                "aggregate_type": aggregate_type,
                "sequence_number": sequence_number,
            }

            event = cls.Event(**event_data)
            return FlextResult[cls.Event].ok(event)

        except ValidationError as e:
            return FlextResult[cls.Event].fail(f"Event validation failed: {e}")
        except Exception as e:
            return FlextResult[cls.Event].fail(f"Event creation failed: {e}")

    @classmethod
    def validate_json_serializable(
        cls, data: FlextTypes.Core.JsonValue
    ) -> FlextResult[FlextTypes.Core.JsonValue]:
        """Validate that data is JSON serializable."""
        try:
            json.dumps(data, default=str)
            return FlextResult[FlextTypes.Core.JsonValue].ok(data)
        except (TypeError, ValueError) as e:
            return FlextResult[FlextTypes.Core.JsonValue].fail(
                f"Data is not JSON serializable: {e}"
            )

    @classmethod
    def safe_parse_datetime(cls, value: str | datetime) -> FlextResult[datetime]:
        """Safely parse datetime from string or return existing datetime."""
        if isinstance(value, datetime):
            # Ensure UTC timezone
            if value.tzinfo is None:
                return FlextResult[datetime].ok(value.replace(tzinfo=UTC))
            return FlextResult[datetime].ok(value.astimezone(UTC))

        try:
            parsed = parser.parse(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return FlextResult[datetime].ok(parsed.astimezone(UTC))
        except Exception as e:
            return FlextResult[datetime].fail(f"Failed to parse datetime: {e}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FlextModels",
]
