"""FLEXT Core Interfaces - Configuration Layer Contract Definitions.

Clean Architecture interface definitions enabling dependency inversion, extensibility,
and consistent contracts across the 32-project FLEXT ecosystem. Foundation for
plugin architecture, service boundaries, and domain-driven design patterns.

Module Role in Architecture:
    Configuration Layer → Interface Contracts → Clean Architecture Boundaries

    This module provides interface abstractions used throughout FLEXT projects:
    - Protocol-based interfaces for structural typing and implementation flexibility
    - Abstract base classes for enforcing implementation contracts
    - Service lifecycle interfaces for start/stop/health operations
    - Plugin interfaces for runtime extensibility without core modification

Interface Architecture Patterns:
    Dependency Inversion: Abstractions independent of concrete implementations
    Protocol-Based Typing: Structural typing for maximum implementation flexibility
    Plugin Architecture: Runtime extensibility through well-defined contracts
    Service Boundaries: Clear interface definition for Clean Architecture layers

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Validation, service, handler, repository interfaces
    🚧 Active Development: Plugin architecture foundation (Priority 3 - October 2025)
    📋 TODO Integration: Event sourcing interfaces (Priority 1)

Interface Categories:
    Validation Interfaces: FlextValidator protocol and FlextValidationRule ABC
    Service Interfaces: FlextService lifecycle and FlextConfigurable protocol
    Handler Interfaces: FlextHandler and FlextMiddleware for CQRS patterns
    Repository Interfaces: FlextRepository and FlextUnitOfWork for data access
    Plugin Interfaces: FlextPlugin and FlextPluginContext for extensibility
    Event Interfaces: FlextEventPublisher and FlextEventSubscriber patterns

Ecosystem Usage Patterns:
    # FLEXT Service Implementation
    class ApiService(FlextService):
        def start(self) -> FlextResult[None]: ...
        def health_check(self) -> FlextResult[TAnyDict]: ...

    # Singer Tap/Target Validation
    class OracleValidator(FlextValidator):
        def validate(self, value: object) -> FlextResult[object]: ...

    # Plugin Development
    class CustomPlugin(FlextPlugin):
        def initialize(self, context: FlextPluginContext) -> FlextResult[None]: ...

    # Repository Pattern (DDD)
    class UserRepository(FlextRepository):
        def find_by_id(self, entity_id: str) -> FlextResult[object]: ...

Clean Architecture Benefits:
    - Dependency inversion preventing tight coupling to implementations
    - Domain layer independence from infrastructure concerns
    - Plugin architecture enabling runtime extensibility
    - Testability through interface mocking and substitution

Quality Standards:
    - All interfaces must use FlextResult for consistent error handling
    - Protocols must be runtime-checkable when used for dynamic validation
    - Abstract base classes must provide comprehensive implementation guidance
    - Interface evolution must maintain backward compatibility

See Also:
    docs/TODO.md: Priority 3 - Plugin architecture foundation
    abc: Abstract base class patterns for interface definition
    typing: Protocol definitions and runtime type checking

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from structlog.stdlib import BoundLogger

    from flext_core.flext_types import TAnyDict
    from flext_core.result import FlextResult

# =============================================================================
# VALIDATION INTERFACES
# =============================================================================


@runtime_checkable
class FlextValidator(Protocol):
    """Protocol for custom validators enabling flexible validation implementation.

    Runtime-checkable protocol defining the contract for custom validation logic
    with FlextResult integration for consistent error handling. Supports structural
    typing allowing any class with matching validate method to be used as validator.

    Protocol Features:
        - Structural typing for maximum implementation flexibility
        - Runtime type checking through @runtime_checkable decorator
        - FlextResult integration for consistent error handling
        - Generic object validation supporting any data type
        - Integration with validation pipelines and chains

    Implementation Guidelines:
        - Return validated/transformed value in success case
        - Use descriptive error messages in failure case
        - Support immutable validation without side effects
        - Handle edge cases gracefully with appropriate error messages
        - Maintain performance for high-frequency validation scenarios

    Usage Patterns:
        # Protocol-compliant validator implementation
        class EmailValidator:
            def validate(self, value: object) -> FlextResult[object]:
                if not _BaseValidators.is_string(value):
                    return FlextResult.fail("Email must be a string")

                if not _BaseValidators.is_email(value):
                    return FlextResult.fail("Invalid email format")

                return FlextResult.ok(value.lower().strip())

        # Runtime type checking
        def use_validator(validator: FlextValidator, data: object):
            if isinstance(validator, FlextValidator):
                return validator.validate(data)
            raise FlextTypeError(
                "Expected FlextValidator protocol",
                expected_type="FlextValidator",
                actual_type=type(validator)
            )

        # Structural typing usage
        email_validator = EmailValidator()
        result = email_validator.validate("user@example.com")

    """

    def validate(self, value: object) -> FlextResult[object]:
        """Validate value and return result with transformation support.

        Core validation method that processes input value and returns either
        validated/transformed value on success or descriptive error on failure.

        Args:
            value: Value to validate (any type supported)

        Returns:
            FlextResult containing validated/transformed value on success
            or error message with context on validation failure

        Usage:
            validator = MyValidator()
            result = validator.validate("input_data")
            if result.success:
                validated_value = result.data
            else:
                error_message = result.error

        """
        ...


class FlextValidationRule(ABC):
    """Abstract base class for validation rules with boolean evaluation.

    Abstract base class defining the contract for validation rules that perform
    boolean evaluation with separated error message generation. Provides a
    template for implementing reusable validation logic with consistent error reporting.

    Rule Design Patterns:
        - Boolean evaluation through check method for simple pass/fail logic
        - Separated error message generation for customizable error reporting
        - Reusable rule composition for complex validation scenarios
        - Template method pattern for consistent validation rule implementation
        - Integration with validation pipelines and rule engines

    Implementation Guidelines:
        - Keep check method pure and side-effect free
        - Provide descriptive error messages that help users understand failures
        - Handle edge cases gracefully without raising exceptions
        - Optimize for performance in high-frequency validation scenarios
        - Support composability for building complex validation logic

    Usage Patterns:
        # Custom validation rule implementation
        class PositiveNumberRule(FlextValidationRule):
            def check(self, value: object) -> bool:
                return isinstance(value, (int, float)) and value > 0

            def error_message(self) -> str:
                return "Value must be a positive number"

        # Email format validation rule
        class EmailFormatRule(FlextValidationRule):
            def check(self, value: object) -> bool:
                return _BaseValidators.is_email(value)

            def error_message(self) -> str:
                return "Invalid email format"

        # Rule composition for complex validation
        class UserAgeRule(FlextValidationRule):
            def __init__(self, min_age: int = 18, max_age: int = 120):
                self.min_age = min_age
                self.max_age = max_age

            def check(self, value: object) -> bool:
                if not isinstance(value, int):
                    return False
                return self.min_age <= value <= self.max_age

            def error_message(self) -> str:
                return f"Age must be between {self.min_age} and {self.max_age}"

        # Using rules in validation logic
        def validate_with_rule(
            rule: FlextValidationRule,
            value: object
        ) -> FlextResult[object]:
            if rule.check(value):
                return FlextResult.ok(value)
            return FlextResult.fail(rule.error_message())

    """

    @abstractmethod
    def check(self, value: object) -> bool:
        """Check if value passes validation rule.

        Core validation logic that evaluates whether the provided value
        satisfies the rule's criteria. Should be implemented as a pure
        function without side effects.

        Args:
            value: Value to validate against this rule

        Returns:
            True if value passes validation, False otherwise

        Implementation Notes:
            - Should handle any input type gracefully
            - Must not raise exceptions for invalid input types
            - Should be optimized for performance if used frequently
            - Keep logic simple and focused on single validation concern

        """
        ...

    @abstractmethod
    def error_message(self) -> str:
        """Get human-readable error message for validation failure.

        Provides descriptive error message that explains why validation
        failed and what the expected criteria are. Used for user feedback
        and debugging purposes.

        Returns:
            Clear, actionable error message explaining validation failure

        Implementation Notes:
            - Should be descriptive and help users understand the requirement
            - Include specific criteria when helpful (e.g., value ranges)
            - Use consistent language and formatting across rules
            - Avoid technical jargon in user-facing messages

        """
        ...


# =============================================================================
# SERVICE INTERFACES
# =============================================================================


class FlextService(ABC):
    """Base interface for all FLEXT services."""

    @abstractmethod
    def start(self) -> FlextResult[None]:
        """Start the service.

        Returns:
            Result of startup

        """
        ...

    @abstractmethod
    def stop(self) -> FlextResult[None]:
        """Stop the service.

        Returns:
            Result of shutdown

        """
        ...

    @abstractmethod
    def health_check(self) -> FlextResult[TAnyDict]:
        """Check service health.

        Returns:
            Result with health status

        """
        ...


@runtime_checkable
class FlextConfigurable(Protocol):
    """Protocol for configurable components."""

    def configure(self, settings: Mapping[str, object]) -> FlextResult[None]:
        """Configure component with settings.

        Args:
            settings: Configuration settings

        Returns:
            Result of configuration

        """
        ...


# =============================================================================
# HANDLER INTERFACES
# =============================================================================


class FlextHandler(ABC):
    """Base interface for command/event handlers."""

    @abstractmethod
    def can_handle(self, message: object) -> bool:
        """Check if handler can process message.

        Args:
            message: Message to check

        Returns:
            True if can handle

        """
        ...

    @abstractmethod
    def handle(self, message: object) -> FlextResult[object]:
        """Handle the message.

        Args:
            message: Message to handle

        Returns:
            Result of handling

        """
        ...


class FlextMiddleware(ABC):
    """Middleware interface for processing pipeline."""

    @abstractmethod
    def process(
        self,
        message: object,
        next_handler: FlextHandler,
    ) -> FlextResult[object]:
        """Process message in pipeline.

        Args:
            message: Message to process
            next_handler: Next handler in chain

        Returns:
            Result from pipeline

        """
        ...


# =============================================================================
# REPOSITORY INTERFACES
# =============================================================================


class FlextRepository(ABC):
    """Base repository interface for data access."""

    @abstractmethod
    def find_by_id(self, entity_id: str) -> FlextResult[object]:
        """Find entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Result with entity or not found error

        """
        ...

    @abstractmethod
    def save(self, entity: object) -> FlextResult[None]:
        """Save entity.

        Args:
            entity: Entity to save

        Returns:
            Result of save operation

        """
        ...

    @abstractmethod
    def delete(self, entity_id: str) -> FlextResult[None]:
        """Delete entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Result of delete operation

        """
        ...


class FlextUnitOfWork(ABC):
    """Unit of Work pattern interface."""

    @abstractmethod
    def commit(self) -> FlextResult[None]:
        """Commit all changes.

        Returns:
            Result of commit

        """
        ...

    @abstractmethod
    def rollback(self) -> FlextResult[None]:
        """Rollback all changes.

        Returns:
            Result of rollback

        """
        ...

    @abstractmethod
    def __enter__(self) -> FlextUnitOfWork:
        """Enter context."""
        ...

    @abstractmethod
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context with automatic rollback on error.

        Args:
            exc_type: Type of exception
            exc_val: Value of exception
            exc_tb: Traceback of exception

        """
        ...


# =============================================================================
# PLUGIN INTERFACES
# =============================================================================


class FlextPlugin(ABC):
    """Base interface for plugins.

    Args:
        **kwargs: Additional keyword arguments

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name.

        Returns:
            Unique plugin name

        """
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version.

        Returns:
            Semantic version string

        """
        ...

    @abstractmethod
    def initialize(self, context: FlextPluginContext) -> FlextResult[None]:
        """Initialize plugin with context.

        Args:
            context: Plugin context

        Returns:
            Result of initialization

        """
        ...

    @abstractmethod
    def shutdown(self) -> FlextResult[None]:
        """Shutdown plugin cleanly.

        Returns:
            Result of shutdown

        """
        ...


@runtime_checkable
class FlextPluginContext(Protocol):
    """Protocol for plugin context."""

    @property
    def logger(self) -> BoundLogger:
        """Get logger for plugin.

        Returns:
            Logger for plugin

        """
        ...

    @property
    def config(self) -> Mapping[str, object]:
        """Get plugin configuration.

        Returns:
            Plugin configuration

        """
        ...

    def get_service(self, service_name: str) -> FlextResult[object]:
        """Get service by name.

        Args:
            service_name: Name of service

        Returns:
            FlextResult with service or error

        """
        ...


# =============================================================================
# EVENT INTERFACES
# =============================================================================


class FlextEventPublisher(ABC):
    """Interface for publishing events.

    Args:
        **kwargs: Additional keyword arguments

    """

    @abstractmethod
    def publish(self, event: object) -> FlextResult[None]:
        """Publish event.

        Args:
            event: Event to publish

        Returns:
            Result of publish

        """
        ...


class FlextEventSubscriber(ABC):
    """Interface for subscribing to events.

    Args:
        **kwargs: Additional keyword arguments

    """

    @abstractmethod
    def subscribe(
        self,
        event_type: type[object],
        handler: FlextHandler,
    ) -> FlextResult[None]:
        """Subscribe to event type.

        Args:
            event_type: Type of events to receive
            handler: Handler for events

        Returns:
            Result of subscription

        """
        ...

    @abstractmethod
    def unsubscribe(
        self,
        event_type: type[object],
        handler: FlextHandler,
    ) -> FlextResult[None]:
        """Unsubscribe from event type.

        Args:
            event_type: Type of events
            handler: Handler to remove

        Returns:
            Result of unsubscription

        """
        ...


# Export API
__all__: list[str] = [
    "FlextConfigurable",
    # Events
    "FlextEventPublisher",
    "FlextEventSubscriber",
    # Handlers
    "FlextHandler",
    "FlextMiddleware",
    # Plugins
    "FlextPlugin",
    "FlextPluginContext",
    # Repository
    "FlextRepository",
    # Services
    "FlextService",
    "FlextUnitOfWork",
    "FlextValidationRule",
    # Validation
    "FlextValidator",
]
