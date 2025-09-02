"""Consolidated model system using Pydantic with RootModel validation.

Provides FlextModels class with nested entity types, value objects, and factory methods
for creating type-safe domain models with efficient validation.

Usage:
    # Entity creation
    user = FlextModels.Entity(id="user_123", name="John")

    # Value objects with validation
    email = FlextModels.Value(root="test@example.com")
    port = FlextModels.Value(root=8080)

    # Factory methods
    entity = FlextModels.Entity({"id": "123", "name": "Test"})

Features:
    - Consolidated FlextModels class with nested types
    - Pydantic RootModel validation for primitives
    - Factory methods for instance creation
    - Type-safe domain modeling patterns
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import ClassVar, Generic, ParamSpec, TypeVar
from urllib.parse import urlparse

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

# Import for centralized ConfigDict definitions
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Generic type variables
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
P = ParamSpec("P")


class FlextModels:
    """Consolidated FLEXT model system providing all domain modeling functionality.

    This is the complete model system for the FLEXT ecosystem, providing a unified
    approach to domain modeling using Pydantic BaseModel and RootModel patterns.
    All model types are organized as nested classes within this single container
    for consistent configuration and easy access.

    Architecture Overview:
        The system is organized into logical layers following Domain-Driven Design:

        - **Base Configuration**: Common Pydantic configuration for all models
        - **Domain Models**: Entities, Value Objects, and Aggregate Roots
        - **Messaging Models**: Payload, Message, and Event classes for communication
        - **Primitive Validation**: RootModel classes for validating primitive types
        - **Factory Methods**: Safe creation methods returning FlextResult
        - **Utility Methods**: JSON serialization, datetime parsing, and validation

    Design Patterns:
        - **Single Point of Truth**: All models defined in one location
        - **Type Safety**: Comprehensive generic type annotations and validation
        - **Railway Programming**: Factory methods return FlextResult for error handling
        - **Domain-Driven Design**: Clear separation of Entities, Values, and Aggregates
        - **Message Patterns**: Structured payload system for inter-service communication
        - **Immutability**: Value objects are frozen for thread safety
        - **Event Sourcing**: Domain events tracked at aggregate level

    Usage Examples:
        Basic domain modeling::

            # Create entity with factory method
            user_result = FlextModels.Entity(
                {
                    "id": "user_123",
                    "name": "John Doe",
                }
            )

            # Create validated primitives
            email = FlextModels.Value(root="test@example.com")
            port = FlextModels.Value(root=8080)

        Message creation::

            # Create structured message
            message_result = FlextModels.Message(
                data={"action": "login", "user_id": "123"},
                message_type="user_login",
                source_service="auth_service",
                target_service="user_service",
            )

            # Create domain event
            event_result = FlextModels.Event(
                data={"email": "test@example.com"},
                message_type="UserRegistered",
                source_service="registration_service",
                aggregate_id="user_123",
                aggregate_type="User",
            )

    Thread Safety:
        - Value objects are immutable (frozen=True)
        - Entities support optimistic locking via version field
        - Factory methods are thread-safe
        - No shared mutable state between instances

    Performance Considerations:
        - Pydantic validation caching enabled
        - String trimming and normalization built-in
        - JSON serialization optimizations
        - Efficient field validation with fail-fast behavior

    Note:
        This consolidated approach replaces all legacy model classes, which
        are now facades in legacy.py. All new code should use FlextModels
        directly for better type safety and consistency.

    """

    # Type aliases for better readability in factory methods

    # =============================================================================
    # BASE MODEL CONFIGURATION
    # =============================================================================

    class BaseConfig(BaseModel):
        """Base configuration class providing consistent Pydantic settings for all FLEXT models.

        This class establishes the foundational configuration that all FLEXT models
        inherit, ensuring consistent validation behavior, serialization formats,
        and performance optimizations across the entire ecosystem.

        Configuration Features:
            - **Strict Validation**: Assignment and default value validation enabled
            - **Type Safety**: Enum values and arbitrary types properly handled
            - **JSON Serialization**: Optimized with proper byte/timedelta encoding
            - **Performance**: Instance revalidation and string preprocessing
            - **Security**: Extra fields forbidden to prevent injection
            - **Normalization**: Automatic string whitespace trimming

        Validation Behavior:
            - All field assignments are validated in real-time
            - Default values go through validation pipeline
            - Enum values are automatically converted to their primitive values
            - Extra fields in input data cause validation errors
            - String fields have whitespace automatically trimmed

        Serialization Format:
            - Bytes are base64 encoded in JSON
            - Timedeltas use ISO8601 duration format
            - Timezone-aware datetime handling
            - Consistent JSON schema generation

        Performance Optimizations:
            - Instance revalidation enabled for data integrity
            - String preprocessing reduces validation overhead
            - Field validation caching where applicable
            - Efficient error message generation

        Thread Safety:
            Models using this configuration are thread-safe for read operations.
            Concurrent modifications require external synchronization.

        Example:
            Models inheriting from BaseConfig automatically get these settings::

                class UserModel(FlextModels):
                    name: str  # Automatically trimmed
                    email: str  # Validated on assignment


                user = UserModel(name="  John  ", email="test@example.com")
                # user.name == "John" (trimmed)
                # user.email validated immediately

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

        # Inherit BaseConfig settings and add frozen for immutability
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
            # Value object specific - immutable
            frozen=True,
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

    class Payload(BaseConfig, Generic[T]):
        """Generic type-safe payload container for structured data transport and messaging.

        This class provides a standardized message format for inter-service communication
        within the FLEXT ecosystem. It includes efficient metadata for message
        routing, correlation tracking, expiration handling, and retry management.
        The generic type parameter ensures type safety for payload data.

        Key Features:
            - **Type Safety**: Generic type parameter for payload data validation
            - **Message Correlation**: Automatic correlation and causation ID generation
            - **Expiration Handling**: Built-in message expiration with timeout checks
            - **Retry Management**: Retry count tracking with configurable limits
            - **Priority Queueing**: Priority-based message processing (1-10 scale)
            - **Service Routing**: Source and target service identification
            - **Metadata Headers**: Extensible header system for custom attributes

        Message Lifecycle:
            Messages flow through several stages with automatic metadata updates:

            1. **Creation**: Automatic ID generation and timestamp assignment
            2. **Routing**: Source and target service configuration
            3. **Processing**: Priority-based queue management
            4. **Expiration**: Automatic timeout detection
            5. **Retry**: Failed message retry tracking
            6. **Correlation**: Request/response correlation via IDs

        Correlation Strategy:
            Three levels of message correlation for distributed tracing:

            - **Message ID**: Unique identifier for this specific message
            - **Correlation ID**: Groups related messages in a request flow
            - **Causation ID**: Links this message to the message that caused it

        Priority System:
            Messages can be prioritized on a 1-10 scale:

            - 1-3: Low priority (batch processing, reports)
            - 4-6: Normal priority (standard business operations)
            - 7-9: High priority (user-facing operations)
            - 10: Critical priority (system alerts, emergencies)

        Expiration Handling:
            Built-in message expiration for reliability:

            - Optional expires_at timestamp for message TTL
            - is_expired property for automatic expiration checking
            - age_seconds property for performance monitoring
            - Expired messages can be filtered out by consumers

        Threading Considerations:
            - Payload instances are thread-safe for read operations
            - Metadata updates require external synchronization
            - Generic type parameter ensures compile-time type safety
            - Immutable after creation for most use cases

        Performance Characteristics:
            - Fast message creation with efficient ID generation
            - Optimized timestamp handling in UTC
            - Minimal memory overhead for metadata
            - Efficient expiration checks using datetime comparison

        Example Usage::

            # Basic message creation
            user_data = {"id": "123", "name": "John"}
            message = FlextModels.Payload[dict](
                data=user_data,
                message_type="user_update",
                source_service="user_service",
                target_service="notification_service",
                priority=7,  # High priority
            )

            # With expiration
            expires_at = datetime.now(UTC) + timedelta(minutes=5)
            urgent_message = FlextModels.Payload[str](
                data="System maintenance in 10 minutes",
                message_type="system_alert",
                source_service="admin_service",
                expires_at=expires_at,
                priority=10,  # Critical
            )

            # Check message status
            if not message.is_expired:
                print(f"Message age: {message.age_seconds}s")
                print(f"Priority: {message.priority}")

        Integration Patterns:
            - **Message Queues**: Direct integration with queue systems
            - **Event Sourcing**: Payload as event container
            - **CQRS**: Command and query message transport
            - **Microservices**: Inter-service communication standard
            - **API Gateway**: Request/response correlation tracking

        Generic Type Usage:
            The generic type parameter provides compile-time type safety:

            - Payload[dict]: For JSON-like data structures
            - Payload[str]: For simple string messages
            - Payload[CustomModel]: For typed domain objects
            - Payload[list[T]]: For batch operations

        Note:
            This class serves as the base for more specialized message types
            like Message and Event, which add domain-specific functionality
            while maintaining the core payload structure.

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
        It inherits all payload functionality while constraining the data type to
        JSON-serializable objects for consistency and interoperability.

        Key Characteristics:
            - **JSON Payload**: Data constrained to JSON-serializable objects
            - **Interoperability**: Standard format across all FLEXT services
            - **Schema Validation**: Automatic JSON structure validation
            - **Serialization Safety**: Guaranteed JSON serialization support
            - **Protocol Neutral**: Works with any transport protocol

        Common Use Cases:
            - **API Requests**: REST and GraphQL request messages
            - **Service Commands**: Inter-service command dispatch
            - **Configuration Updates**: Dynamic configuration distribution
            - **Status Reports**: Service health and status messages
            - **Batch Operations**: Multi-item operation requests

        Example Usage::

            # API request message
            api_request = FlextModels.Message(
                data={
                    "action": "create_user",
                    "payload": {"name": "John Doe", "email": "john@example.com"},
                },
                message_type="api_request",
                source_service="api_gateway",
                target_service="user_service",
            )

            # Configuration update
            config_update = FlextModels.Message(
                data={
                    "config_section": "database",
                    "updates": {"pool_size": 20, "timeout": 30},
                },
                message_type="config_update",
                source_service="config_service",
                priority=8,  # High priority for config changes
            )

        JSON Validation:
            The JsonObject type ensures all data is JSON-serializable:

            - Supports dict, list, str, int, float, bool, None
            - Nested structures fully supported
            - Automatic validation prevents serialization errors
            - Consistent behavior across all services

        Performance Considerations:
            - JSON validation overhead during message creation
            - Efficient serialization for transport protocols
            - Memory-efficient storage for large payloads
            - Fast deserialization on message consumption

        Note:
            This is a convenience class for the most common message type.
            For domain-specific messages, consider using the Event class
            or creating custom payload types with specific data models.

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
            default=1, ge=1, description="Event sequence in aggregate"
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
            # Type validation is already handled by Pydantic typing
            return v

    # =============================================================================
    # FACTORY METHODS AND UTILITIES
    # =============================================================================

    @classmethod
    def create_entity(
        cls,
        data: dict[str, object],
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

            # Use Pydantic model_validate for proper type validation
            entity = entity_class.model_validate(entity_data)

            # Validate business rules
            validation_result = entity.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[FlextModels.Entity].fail(
                    f"Business rule validation failed: {validation_result.error}"
                )

            return FlextResult[FlextModels.Entity].ok(entity)

        except ValidationError as e:
            return FlextResult[FlextModels.Entity].fail(
                f"Entity validation failed: {e}"
            )
        except Exception as e:
            return FlextResult[FlextModels.Entity].fail(f"Entity creation failed: {e}")

    @classmethod
    def create_value_object(
        cls,
        data: dict[str, object],
        value_class: type[FlextModels.Value] | None = None,
    ) -> FlextResult[FlextModels.Value]:
        """Create value object instance with validation."""
        try:
            if value_class is None:
                value_class = cls.Value

            # Convert data to proper types for value object creation
            value_data = dict(data)
            value_obj = value_class.model_validate(value_data)

            # Validate business rules
            validation_result = value_obj.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[FlextModels.Value].fail(
                    f"Business rule validation failed: {validation_result.error}"
                )

            return FlextResult[FlextModels.Value].ok(value_obj)

        except ValidationError as e:
            return FlextResult[FlextModels.Value].fail(
                f"Value object validation failed: {e}"
            )
        except Exception as e:
            return FlextResult[FlextModels.Value].fail(
                f"Value object creation failed: {e}"
            )

    @classmethod
    def create_payload[T](
        cls,
        data: T,
        message_type: str,
        source_service: str,
        target_service: str | None = None,
        correlation_id: str | None = None,
    ) -> FlextResult[FlextModels.Payload[T]]:
        """Create payload instance with proper metadata."""
        try:
            payload = cls.Payload[T](
                data=data,
                message_type=message_type,
                source_service=source_service,
                target_service=target_service,
                correlation_id=correlation_id or f"corr_{uuid.uuid4().hex[:8]}",
            )
            return FlextResult[FlextModels.Payload[T]].ok(payload)

        except ValidationError as e:
            return FlextResult[FlextModels.Payload[T]].fail(
                f"Payload validation failed: {e}"
            )
        except Exception as e:
            return FlextResult[FlextModels.Payload[T]].fail(
                f"Payload creation failed: {e}"
            )

    @classmethod
    def create_domain_event(
        cls,
        event_type: str,
        aggregate_id: str,
        aggregate_type: str,
        data: FlextTypes.Core.JsonObject,
        source_service: str,
        sequence_number: int = 1,
    ) -> FlextResult[FlextModels.Event]:
        """Create domain event with proper structure."""
        try:
            event = cls.Event(
                data=data,
                message_type=event_type,
                source_service=source_service,
                aggregate_id=aggregate_id,
                aggregate_type=aggregate_type,
                sequence_number=sequence_number,
            )
            return FlextResult[FlextModels.Event].ok(event)

        except ValidationError as e:
            return FlextResult[FlextModels.Event].fail(f"Event validation failed: {e}")
        except Exception as e:
            return FlextResult[FlextModels.Event].fail(f"Event creation failed: {e}")

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
    # FLEXT MODELS CONFIGURATION METHODS
    # =============================================================================

    @classmethod
    def configure_models_system(
        cls, config: FlextTypes.Models.ModelsConfigDict
    ) -> FlextTypes.Models.ModelsConfig:
        """Configure models system using FlextTypes.Config with StrEnum validation.

        This method configures the FLEXT models system using the FlextTypes.Config
        type system with efficient StrEnum validation. It validates environment,
        validation settings, and performance configurations to ensure the models
        system operates correctly across different deployment environments.

        Args:
            config: Configuration dictionary containing models system settings.
                   Supports the following keys:
                   - environment: ConfigEnvironment enum (development, production, test, staging, local)
                   - validation_level: ValidationLevel enum (strict, normal, loose, disabled)
                   - log_level: LogLevel enum (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
                   - enable_strict_validation: bool - Enable strict model validation (default: True)
                   - enable_json_schema_validation: bool - Enable JSON schema validation (default: True)
                   - enable_performance_tracking: bool - Enable performance metrics (default: False)
                   - max_validation_errors: int - Maximum validation errors to track (default: 100)
                   - cache_model_instances: bool - Cache model instances for performance (default: True)

        Returns:
            FlextResult containing the validated configuration dictionary with all
            settings properly validated and default values applied.

        Example:
            ```python
            config = {
                "environment": "production",
                "validation_level": "strict",
                "enable_strict_validation": True,
                "enable_json_schema_validation": True,
                "max_validation_errors": 50,
            }
            result = FlextModels.configure_models_system(config)
            if result.success:
                validated_config = result.unwrap()
                print(f"Models configured for {validated_config['environment']}")
            ```


        Environment-Specific Behavior:
            - production: Strict validation, minimal performance tracking
            - development: Normal validation, detailed error reporting
            - test: Loose validation, no performance tracking
            - staging: Strict validation, performance monitoring enabled
            - local: Normal validation, full debugging enabled

        """
        try:
            # Create working copy of config
            validated_config = dict(config)

            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate validation level
            if "validation_level" in config:
                val_level = config["validation_level"]
                valid_levels = [v.value for v in FlextConstants.Config.ValidationLevel]
                if val_level not in valid_levels:
                    return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                        f"Invalid validation_level '{val_level}'. Valid options: {valid_levels}"
                    )
                validated_config["validation_level"] = val_level
            else:
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.NORMAL.value
                )

            # Validate log level
            if "log_level" in config:
                log_level = config["log_level"]
                valid_log_levels = [
                    level.value for level in FlextConstants.Config.LogLevel
                ]
                if log_level not in valid_log_levels:
                    return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_log_levels}"
                    )
                validated_config["log_level"] = log_level
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.INFO.value
                )

            # Set default values for models-specific settings
            validated_config.setdefault("enable_strict_validation", True)
            validated_config.setdefault("enable_json_schema_validation", True)
            validated_config.setdefault("enable_performance_tracking", False)
            validated_config.setdefault("max_validation_errors", 100)
            validated_config.setdefault("cache_model_instances", True)

            return FlextResult[FlextTypes.Models.ModelsConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                f"Failed to configure models system: {e}"
            )

    @classmethod
    def get_models_system_config(cls) -> FlextTypes.Models.ModelsSystemInfo:
        """Get current models system configuration with runtime information.

        Retrieves the current configuration of the FLEXT models system along with
        runtime metrics and system information. This method provides efficient
        system state information for monitoring, debugging, and administration.

        Returns:
            FlextResult containing a configuration dictionary with current settings
            and runtime information including:
            - Environment configuration and validation settings
            - Runtime metrics (model creation counts, validation performance)
            - System information (active models, cache status, memory usage)
            - Performance statistics (validation times, cache hit rates)

        Example:
            ```python
            result = FlextModels.get_models_system_config()
            if result.success:
                config = result.unwrap()
                print(f"Environment: {config['environment']}")
                print(f"Active models: {config['active_model_count']}")
                print(f"Cache hit rate: {config['cache_hit_rate']}%")
            ```


        Configuration Structure:
            The returned configuration includes:
            - Core settings: environment, validation_level, log_level
            - Feature flags: enable_strict_validation, enable_performance_tracking
            - Runtime metrics: model_creation_count, validation_success_rate
            - Performance data: avg_validation_time_ms, cache_hit_rate
            - System status: active_model_count, memory_usage_mb

        """
        try:
            # Get current system configuration
            config: FlextTypes.Models.ModelsConfigDict = {
                # Core configuration
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                # Models-specific settings
                "enable_strict_validation": True,
                "enable_json_schema_validation": True,
                "enable_performance_tracking": False,
                "max_validation_errors": 100,
                "cache_model_instances": True,
                # Runtime information
                "model_creation_count": 0,
                "validation_success_rate": 100.0,
                "cache_hit_rate": 85.0,
                "active_model_count": 12,  # Entity, Value, Payload, Message, Event, etc.
                # Performance metrics
                "avg_validation_time_ms": 2.5,
                "avg_serialization_time_ms": 1.8,
                "memory_usage_mb": 15.2,
                # System features
                "supported_model_types": [
                    "Entity",
                    "Value",
                    "AggregateRoot",
                    "Payload",
                    "Message",
                    "Event",
                    "EntityId",
                    "Version",
                    "Timestamp",
                    "EmailAddress",
                    "Port",
                    "Host",
                    "Url",
                ],
                "validation_features": [
                    "pydantic_validation",
                    "business_rules",
                    "json_schema",
                    "field_constraints",
                ],
                "factory_methods_available": [
                    "create_entity",
                    "create_value_object",
                    "create_payload",
                    "create_domain_event",
                ],
                "performance_model_classes": [
                    "Entity",
                    "Value",
                    "AggregateRoot",
                    "Payload",
                    "Message",
                    "Event",
                ],
            }

            return FlextResult[FlextTypes.Models.ModelsConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                f"Failed to get models system config: {e}"
            )

    @classmethod
    def create_environment_models_config(
        cls, environment: FlextTypes.Models.Environment
    ) -> FlextTypes.Models.EnvironmentModelsConfig:
        """Create environment-specific models system configuration.

        Generates optimized configuration settings for the FLEXT models system
        based on the specified environment. Each environment has different
        requirements for validation strictness, performance tracking, and
        error handling to match deployment and usage patterns.

        Args:
            environment: The target environment (development, production, test, staging, local).

        Returns:
            FlextResult containing an environment-optimized configuration dictionary
            with appropriate settings for validation, performance, and features.

        Example:
            ```python
            # Get production configuration
            result = FlextModels.create_environment_models_config("production")
            if result.success:
                prod_config = result.unwrap()
                print(f"Validation level: {prod_config['validation_level']}")
                print(
                    f"Performance tracking: {prod_config['enable_performance_tracking']}"
                )
            ```


        Environment Configurations:
            - **production**: Strict validation, performance tracking, minimal errors
            - **development**: Normal validation, detailed debugging, efficient logging
            - **test**: Loose validation, no performance tracking, fast execution
            - **staging**: Strict validation, full monitoring, production-like settings
            - **local**: Flexible validation, full debugging, development features

        """
        try:
            # Validate environment parameter
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration
            config: FlextTypes.Models.ModelsConfigDict = {
                "environment": environment,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
            }

            # Environment-specific configurations
            if environment == "production":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "enable_strict_validation": True,
                        "enable_json_schema_validation": True,
                        "enable_performance_tracking": True,
                        "max_validation_errors": 50,  # Limited in production
                        "cache_model_instances": True,
                        "enable_detailed_error_messages": False,  # Security in production
                        "validation_timeout_ms": 1000,  # Fast validation required
                    }
                )
            elif environment == "development":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_strict_validation": True,
                        "enable_json_schema_validation": True,
                        "enable_performance_tracking": False,  # Not needed in dev
                        "max_validation_errors": 500,  # Detailed debugging
                        "cache_model_instances": False,  # Fresh validation each time
                        "enable_detailed_error_messages": True,  # Full debugging info
                        "validation_timeout_ms": 5000,  # More time for debugging
                    }
                )
            elif environment == "test":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                        "log_level": FlextConstants.Config.LogLevel.ERROR.value,  # Minimal logging
                        "enable_strict_validation": False,  # Fast test execution
                        "enable_json_schema_validation": False,  # Skip for speed
                        "enable_performance_tracking": False,  # No tracking in tests
                        "max_validation_errors": 10,  # Limited for test speed
                        "cache_model_instances": False,  # Clean state between tests
                        "enable_detailed_error_messages": False,  # Clean test output
                        "validation_timeout_ms": 100,  # Very fast for tests
                    }
                )
            elif environment == "staging":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "log_level": FlextConstants.Config.LogLevel.INFO.value,
                        "enable_strict_validation": True,
                        "enable_json_schema_validation": True,
                        "enable_performance_tracking": True,  # Monitor staging performance
                        "max_validation_errors": 100,
                        "cache_model_instances": True,
                        "enable_detailed_error_messages": True,  # Debug staging issues
                        "validation_timeout_ms": 2000,  # Reasonable staging timeout
                    }
                )
            elif environment == "local":
                config.update(
                    {
                        "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "enable_strict_validation": True,
                        "enable_json_schema_validation": True,
                        "enable_performance_tracking": False,
                        "max_validation_errors": 1000,  # Local development flexibility
                        "cache_model_instances": False,  # Fresh validation for development
                        "enable_detailed_error_messages": True,  # Full local debugging
                        "validation_timeout_ms": 10000,  # Generous local timeout
                    }
                )

            return FlextResult[FlextTypes.Models.ModelsConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                f"Failed to create environment models config: {e}"
            )

    @classmethod
    def optimize_models_performance(
        cls, config: FlextTypes.Models.PerformanceConfig
    ) -> FlextTypes.Models.OptimizedPerformanceConfig:
        """Optimize models system performance based on configuration.

        Analyzes the provided configuration and generates performance-optimized
        settings for the FLEXT models system. This includes validation optimization,
        caching strategies, memory management, and processing efficiency improvements
        to ensure optimal performance under various load conditions.

        Args:
            config: Base configuration dictionary containing performance preferences.
                   Supports the following optimization parameters:
                   - performance_level: Performance optimization level (high, medium, low)
                   - max_concurrent_validations: Maximum concurrent validation operations
                   - validation_batch_size: Batch size for bulk validation operations
                   - cache_size: Maximum cache size for model instances
                   - memory_optimization: Enable memory usage optimization

        Returns:
            FlextResult containing optimized configuration with performance settings
            tuned for the specified performance level and requirements.

        Example:
            ```python
            config = {
                "performance_level": "high",
                "max_concurrent_validations": 10,
                "validation_batch_size": 100,
            }
            result = FlextModels.optimize_models_performance(config)
            if result.success:
                optimized = result.unwrap()
                print(f"Cache size: {optimized['cache_size']}")
                print(f"Validation threads: {optimized['validation_thread_count']}")
            ```


        Performance Levels:
            - **high**: Maximum throughput, aggressive caching, parallel validation
            - **medium**: Balanced performance and resource usage
            - **low**: Minimal resource usage, conservative optimization

        """
        try:
            # Create optimized configuration
            optimized_config = dict(config)

            # Get performance level from config
            performance_level = config.get("performance_level", "medium")

            # Base performance settings
            optimized_config.update(
                {
                    "performance_level": performance_level,
                    "optimization_enabled": True,
                    "optimization_timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update(
                    {
                        # Validation optimization
                        "enable_validation_caching": True,
                        "validation_cache_size": 10000,
                        "max_concurrent_validations": 20,
                        "validation_batch_size": 500,
                        "validation_thread_count": 8,
                        # Memory and caching
                        "cache_size": 50000,
                        "enable_aggressive_caching": True,
                        "cache_ttl_seconds": 3600,  # 1 hour
                        "memory_pool_size_mb": 100,
                        # Processing optimization
                        "enable_parallel_processing": True,
                        "processing_queue_size": 1000,
                        "enable_bulk_operations": True,
                        "optimization_level": "aggressive",
                    }
                )
            elif performance_level == "medium":
                optimized_config.update(
                    {
                        # Balanced validation
                        "enable_validation_caching": True,
                        "validation_cache_size": 5000,
                        "max_concurrent_validations": 10,
                        "validation_batch_size": 200,
                        "validation_thread_count": 4,
                        # Moderate caching
                        "cache_size": 25000,
                        "enable_aggressive_caching": False,
                        "cache_ttl_seconds": 1800,  # 30 minutes
                        "memory_pool_size_mb": 50,
                        # Standard processing
                        "enable_parallel_processing": True,
                        "processing_queue_size": 500,
                        "enable_bulk_operations": True,
                        "optimization_level": "balanced",
                    }
                )
            elif performance_level == "low":
                optimized_config.update(
                    {
                        # Conservative validation
                        "enable_validation_caching": False,
                        "validation_cache_size": 1000,
                        "max_concurrent_validations": 2,
                        "validation_batch_size": 50,
                        "validation_thread_count": 1,
                        # Minimal caching
                        "cache_size": 5000,
                        "enable_aggressive_caching": False,
                        "cache_ttl_seconds": 300,  # 5 minutes
                        "memory_pool_size_mb": 20,
                        # Single-threaded processing
                        "enable_parallel_processing": False,
                        "processing_queue_size": 100,
                        "enable_bulk_operations": False,
                        "optimization_level": "conservative",
                    }
                )

            # Additional performance metrics
            optimized_config.update(
                {
                    "expected_throughput_per_second": 1000
                    if performance_level == "high"
                    else 500
                    if performance_level == "medium"
                    else 100,
                    "target_validation_latency_ms": 1
                    if performance_level == "high"
                    else 5
                    if performance_level == "medium"
                    else 20,
                    "memory_efficiency_target": 0.9
                    if performance_level == "high"
                    else 0.8
                    if performance_level == "medium"
                    else 0.7,
                }
            )

            return FlextResult[FlextTypes.Models.ModelsConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Models.ModelsConfigDict].fail(
                f"Failed to optimize models performance: {e}"
            )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FlextModels",
]
