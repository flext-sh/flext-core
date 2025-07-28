"""FLEXT Core - Enterprise Foundation Library.

Comprehensive foundation library for the FLEXT ecosystem providing
Clean Architecture, Domain-Driven Design, and CQRS patterns with type-safe operations.

Architecture:
    - Clean Architecture with clear separation of concerns
    - Domain-Driven Design patterns for business logic modeling
    - CQRS implementation for command-query responsibility segregation
    - Railway-oriented programming for error handling
    - Type-safe operations with comprehensive generic support

Core Components:
    - FlextResult: Type-safe error handling with railway programming
    - FlextEntity: Domain entities with rich behavior and validation
    - FlextValueObject: Immutable value objects with validation
    - FlextAggregateRoot: DDD aggregate roots with event sourcing
    - FlextDomainService: Domain services for complex business logic
    - FlextTypes: Comprehensive type system with protocols and aliases

Enterprise Features:
    - Type safety through extensive generic programming
    - Error handling without exception propagation
    - Validation system with Pydantic integration
    - Logging system with structured context management
    - Dependency injection container with type-safe operations
    - Configuration management with environment variable support

Public API Categories:
    - Core patterns: FlextResult, type system, constants
    - DDD patterns: Entities, value objects, aggregates, domain services
    - Type system: Generic types, protocols, and domain-specific aliases
    - Constants: Environment settings, field types, log levels
    - Version information: Library version and authorship

Design Principles:
    - Single responsibility principle with focused components
    - Open/closed principle through extensible patterns
    - Liskov substitution with compatible implementations
    - Interface segregation through protocol-based design
    - Dependency inversion through abstraction dependencies

Usage Patterns:
    # Basic error handling
    from flext_core import FlextResult
    result = FlextResult.ok("success")

    # Domain modeling
    from flext_core import FlextEntity, TEntityId
    class User(FlextEntity):
        def __init__(self, user_id: TEntityId, name: str):
            super().__init__(user_id)
            self.name = name

    # Type-safe operations
    from flext_core import T, U
    def process_data[T, U](data: T, transformer: Callable[[T], U]) -> U:
        return transformer(data)

Dependencies:
    - pydantic: Data validation and configuration management
    - typing: Type system and generic programming support
    - Standard library: Core Python functionality

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.commands import FlextCommands
from flext_core.config import FlextConfig
from flext_core.constants import (
    DEFAULT_TIMEOUT,
    EMAIL_PATTERN,
    ERROR_CODES,
    VERSION,
    FlextConstants,
    FlextEnvironment,
    FlextFieldType,
    FlextLogLevel,
)
from flext_core.container import (
    FlextContainer,
    ServiceKey,
    configure_flext_container,
    get_flext_container,
    get_typed,
    register_typed,
)
from flext_core.core import FlextCore, flext_core
from flext_core.decorators import FlextDecorators
from flext_core.domain_services import FlextDomainService
from flext_core.entities import FlextEntity, FlextEntityFactory
from flext_core.exceptions import (
    FlextAlreadyExistsError,
    FlextAuthenticationError,
    FlextConfigurationError,
    FlextConnectionError,
    FlextCriticalError,
    FlextError,
    FlextExceptions,
    FlextNotFoundError,
    FlextOperationError,
    FlextPermissionError,
    FlextProcessingError,
    FlextTimeoutError,
    FlextTypeError,
    FlextValidationError,
    clear_exception_metrics,
    get_exception_metrics,
)
from flext_core.fields import FlextFieldCore, FlextFields
from flext_core.handlers import FlextHandlers
from flext_core.loggings import (
    FlextLogContext,
    FlextLogger,
    FlextLoggerFactory,
    FlextLogging,
    create_log_context,
    get_logger,
)
from flext_core.mixins import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextEntityMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextValueObjectMixin,
)
from flext_core.payload import FlextEvent, FlextMessage, FlextPayload
from flext_core.result import FlextResult
from flext_core.types import (
    Comparable,
    E,
    FlextEntityId,
    FlextTypes,
    Identifiable,
    P,
    R,
    Serializable,
    T,
    TAnyDict,
    TAnyList,
    TCommand,
    TData,
    TEntity,
    TEntityId,
    TErrorCode,
    TErrorMessage,
    TEvent,
    Timestamped,
    TMessage,
    TPredicate,
    TQuery,
    TRequest,
    TRequestId,
    TResponse,
    TResult,
    TService,
    TTransformer,
    TValue,
    U,
    V,
    Validatable,
)
from flext_core.utilities import FlextUtilities
from flext_core.validation import FlextValidation, FlextValidators
from flext_core.value_objects import FlextValueObject, FlextValueObjectFactory
from flext_core.version import (
    AVAILABLE_FEATURES,
    MAX_PYTHON_VERSION,
    MIN_PYTHON_VERSION,
    FlextCompatibilityResult,
    FlextVersionInfo,
    check_python_compatibility,
    compare_versions,
    get_version_info,
    get_version_tuple,
    get_available_features,
    get_version_string,
    is_feature_available,
    validate_version_format,
)

# Version information
__version__ = "0.8.0"
__author__ = "FLEXT Contributors"

# Clean public API - semantically organized by functional categories
__all__ = [
    # === CORE PATTERNS ===
    "FlextResult",  # Railway-oriented programming for error handling
    "FlextCore",  # Central orchestration class for all FLEXT Core functionality
    "flext_core",  # Convenient global access to FlextCore singleton
    "FlextPayload",  # Generic message transport and serialization
    "FlextMessage",  # String message payload
    "FlextEvent",  # Domain event payload
    # === DOMAIN-DRIVEN DESIGN PATTERNS ===
    "FlextEntity",  # Domain entities with identity and behavior
    "FlextValueObject",  # Immutable value objects
    "FlextAggregateRoot",  # DDD aggregate roots with event sourcing
    "FlextDomainService",  # Domain services for complex business logic
    "FlextEntityFactory",  # Factory pattern for entity creation
    "FlextValueObjectFactory",  # Factory pattern for value object creation
    # === CQRS AND MESSAGING ===
    "FlextCommands",  # CQRS command patterns
    "FlextHandlers",  # Message and event handlers
    # === EXCEPTION HANDLING ===
    "FlextError",  # Base exception with metrics
    "FlextValidationError",  # Validation failures
    "FlextTypeError",  # Type conversion errors
    "FlextOperationError",  # Operation failures
    "FlextConfigurationError",  # Configuration errors
    "FlextConnectionError",  # Connection failures
    "FlextAuthenticationError",  # Authentication failures
    "FlextPermissionError",  # Permission denials
    "FlextNotFoundError",  # Resource not found
    "FlextAlreadyExistsError",  # Resource conflicts
    "FlextTimeoutError",  # Timeout errors
    "FlextProcessingError",  # Processing failures
    "FlextCriticalError",  # Critical system errors
    "FlextExceptions",  # Exception factory
    "get_exception_metrics",  # Exception metrics
    "clear_exception_metrics",  # Clear metrics
    # === FIELD DEFINITIONS ===
    "FlextFields",  # Field factory and registry
    "FlextFieldCore",  # Core field implementation
    # === LOGGING SYSTEM ===
    "FlextLogger",  # Structured logger
    "FlextLoggerFactory",  # Logger factory
    "FlextLogContext",  # Log context manager
    "FlextLogging",  # Logging facade
    "get_logger",  # Get logger utility
    "create_log_context",  # Create log context
    # === BEHAVIORAL MIXINS ===
    "FlextEntityMixin",  # Complete entity pattern (ID + timestamps + validation)
    "FlextValueObjectMixin",  # Complete value object pattern
    "FlextIdentifiableMixin",  # Unique identification behavior
    "FlextTimestampMixin",  # Creation and update tracking
    "FlextValidatableMixin",  # Validation state management
    "FlextSerializableMixin",  # Dictionary serialization
    "FlextComparableMixin",  # Comparison operations
    "FlextLoggableMixin",  # Structured logging
    "FlextTimingMixin",  # Execution timing
    "FlextCacheableMixin",  # Caching capabilities
    # === DEPENDENCY INJECTION ===
    "FlextContainer",  # Main DI container
    "ServiceKey",  # Type-safe service keys
    "get_flext_container",  # Global container access
    "configure_flext_container",  # Container configuration
    "register_typed",  # Type-safe service registration
    "get_typed",  # Type-safe service retrieval
    # === CONFIGURATION AND CONSTANTS ===
    "FlextConfig",  # Configuration management
    "FlextConstants",  # Core constants
    "FlextEnvironment",  # Environment enumeration
    "FlextFieldType",  # Field type enumeration
    "FlextLogLevel",  # Log level enumeration
    "DEFAULT_TIMEOUT",  # Default timeout value
    "EMAIL_PATTERN",  # Email validation pattern
    "ERROR_CODES",  # Error code constants
    "VERSION",  # Library version
    # === UTILITIES AND DECORATORS ===
    "FlextDecorators",  # Decorator patterns
    "FlextUtilities",  # Utility functions
    "FlextValidation",  # Validation utilities
    "FlextValidators",  # Validation validators
    "FlextTypes",  # Type system utilities
    # === TYPE SYSTEM - Generic Types (Single Letters) ===
    "T",  # Primary generic type
    "U",  # Secondary generic type
    "V",  # Tertiary generic type
    "E",  # Exception/Error type
    "P",  # Parameter type
    "R",  # Return type
    # === TYPE SYSTEM - Domain Types (T<Name> Convention) ===
    "TEntity",  # Entity type
    "TEntityId",  # Entity identifier type
    "FlextEntityId",  # Concrete entity identifier type
    "TValue",  # Value object type
    "TData",  # Data transfer object type
    "TResult",  # Result wrapper type
    "TMessage",  # Message type
    "TEvent",  # Domain event type
    "TCommand",  # Command type
    "TQuery",  # Query type
    "TRequest",  # Request type
    "TRequestId",  # Request identifier type
    "TResponse",  # Response type
    "TService",  # Service type
    "TTransformer",  # Transformer function type
    "TPredicate",  # Predicate function type
    "TErrorMessage",  # Error message type
    "TErrorCode",  # Error code type
    "TAnyDict",  # Generic dictionary type
    "TAnyList",  # Generic list type
    # === TYPE SYSTEM - Protocol Types ===
    "Identifiable",  # Protocol for identifiable objects
    "Timestamped",  # Protocol for timestamped objects
    "Validatable",  # Protocol for validatable objects
    "Serializable",  # Protocol for serializable objects
    "Comparable",  # Protocol for comparable objects
    # === VERSION MANAGEMENT ===
    "get_version_tuple",  # Version as tuple
    "get_version_info",  # Complete version info
    "get_version_string",  # Formatted version
    "check_python_compatibility",  # Python compatibility
    "is_feature_available",  # Feature availability
    "get_available_features",  # Available features list
    "compare_versions",  # Version comparison
    "validate_version_format",  # Version validation
    "FlextVersionInfo",  # Version info type
    "FlextCompatibilityResult",  # Compatibility result type
    "AVAILABLE_FEATURES",  # Available features dict
    "MIN_PYTHON_VERSION",  # Minimum Python version
    "MAX_PYTHON_VERSION",  # Maximum Python version
    "__version__",  # Library version string
]
