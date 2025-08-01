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
from flext_core.config import (
    FlextBaseSettings,
    FlextConfig,
    FlextConfigDefaults,
    FlextConfigOps,
    FlextConfigValidation,
    merge_configs,
)
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
    create_module_container_utilities,
    get_flext_container,
    get_typed,
    register_typed,
)
from flext_core.core import FlextCore, flext_core
from flext_core.decorators import (
    FlextDecorators,
    FlextErrorHandlingDecorators,
    FlextImmutabilityDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
    FlextValidationDecorators,
)
from flext_core.domain_services import FlextDomainService
from flext_core.entities import FlextEntity, FlextEntityFactory
from flext_core.exceptions import (
    FlextAlreadyExistsError,
    FlextAuthenticationError,
    FlextConfigurationError,
    FlextConnectionError,
    FlextCriticalError,
    FlextError,
    FlextNotFoundError,
    FlextOperationError,
    FlextPermissionError,
    FlextProcessingError,
    FlextTimeoutError,
    FlextTypeError,
    FlextValidationError,
    create_module_exception_classes,
)
from flext_core.fields import FlextFieldCore, FlextFields

# Smart core removed - experimental features
# Smart validation removed - experimental features
from flext_core.flext_types import (
    Comparable,
    E,
    FlextEntityId,
    FlextTypes,
    P,
    R,
    Serializable,
    T,
    TAnyDict,
    TAnyList,
    TAnyObject,
    TCommand,
    TConfigDict,
    TData,
    TEntity,
    TEntityId,
    TErrorCode,
    TErrorMessage,
    TEvent,
    Timestamped,
    TLogMessage,
    TMessage,
    TPredicate,
    TQuery,
    TRequest,
    TRequestId,
    TResponse,
    TResult,
    TService,
    TTransformer,
    TUserData,
    TValue,
    U,
    V,
    Validatable,
)
from flext_core.guards import (
    ValidatedModel,
    immutable,
    is_dict_of,
    make_builder,
    make_factory,
    pure,
    require_in_range,
    require_non_empty,
    require_not_none,
    require_positive,
)
from flext_core.handlers import FlextHandlers
from flext_core.interfaces import (
    FlextConfigurable,
    FlextEventPublisher,
    FlextEventSubscriber,
    FlextHandler,
    FlextMiddleware,
    FlextPlugin,
    FlextPluginContext,
    FlextRepository,
    FlextService,
    FlextUnitOfWork,
    FlextValidationRule,
    FlextValidator,
)
from flext_core.loggings import (
    FlextLogContext,
    FlextLogger,
    FlextLoggerFactory,
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
from flext_core.result import FlextResult, chain, safe_call

# Singer Protocol Base Exceptions - Eliminates duplication across Singer projects
from flext_core.singer_base import (
    FlextSingerAuthenticationError,
    FlextSingerConfigurationError,
    FlextSingerConnectionError,
    FlextSingerError,
    FlextSingerProcessingError,
    FlextSingerValidationError,
    FlextTapError,
    FlextTargetError,
    FlextTransformError,
)
from flext_core.utilities import FlextGenerators, FlextUtilities
from flext_core.validation import FlextPredicates, FlextValidation, FlextValidators
from flext_core.value_objects import FlextValueObject, FlextValueObjectFactory
from flext_core.version import (
    AVAILABLE_FEATURES,
    MAX_PYTHON_VERSION,
    MIN_PYTHON_VERSION,
    FlextCompatibilityResult,
    FlextVersionInfo,
    check_python_compatibility,
    compare_versions,
    get_available_features,
    get_version_info,
    get_version_string,
    get_version_tuple,
    is_feature_available,
    validate_version_format,
)

# Version information
__version__ = "0.9.0"
__author__ = "FLEXT Contributors"

# Clean public API - alphabetically organized for ruff compliance
__all__ = [
    "AVAILABLE_FEATURES",
    "DEFAULT_TIMEOUT",
    "EMAIL_PATTERN",
    "ERROR_CODES",
    "MAX_PYTHON_VERSION",
    "MIN_PYTHON_VERSION",
    "VERSION",
    "Comparable",
    "E",
    "FlextAggregateRoot",
    "FlextAlreadyExistsError",
    "FlextAuthenticationError",
    "FlextBaseSettings",
    "FlextCacheableMixin",
    "FlextCommands",
    "FlextComparableMixin",
    "FlextCompatibilityResult",
    "FlextConfig",
    "FlextConfigDefaults",
    "FlextConfigOps",
    "FlextConfigValidation",
    "FlextConfigurable",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextConstants",
    "FlextContainer",
    "FlextCore",
    "FlextCriticalError",
    "FlextDecorators",
    "FlextDomainService",
    "FlextEntity",
    "FlextEntityFactory",
    "FlextEntityId",
    "FlextEntityMixin",
    "FlextEnvironment",
    "FlextError",
    "FlextErrorHandlingDecorators",
    "FlextEvent",
    "FlextEventPublisher",
    "FlextEventSubscriber",
    "FlextExceptions",
    "FlextFieldCore",
    "FlextFieldType",
    "FlextFields",
    "FlextGenerators",
    "FlextHandler",
    "FlextHandlers",
    "FlextIdentifiableMixin",
    "FlextImmutabilityDecorators",
    "FlextLogContext",
    "FlextLogLevel",
    "FlextLoggableMixin",
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLoggingDecorators",
    "FlextMessage",
    "FlextMiddleware",
    "FlextNotFoundError",
    "FlextOperationError",
    "FlextPayload",
    "FlextPerformanceDecorators",
    "FlextPermissionError",
    "FlextPlugin",
    "FlextPluginContext",
    "FlextPredicates",
    "FlextProcessingError",
    "FlextRepository",
    "FlextResult",
    "FlextSerializableMixin",
    "FlextService",
    "FlextSingerAuthenticationError",
    "FlextSingerConfigurationError",
    "FlextSingerConnectionError",
    # Singer Protocol Base Exceptions (DRY implementation for all Singer projects)
    "FlextSingerError",
    "FlextSingerProcessingError",
    "FlextSingerValidationError",
    "FlextTapError",
    "FlextTargetError",
    "FlextTimeoutError",
    "FlextTimestampMixin",
    "FlextTimingMixin",
    "FlextTransformError",
    "FlextTypeError",
    "FlextTypes",
    "FlextUnitOfWork",
    "FlextUtilities",
    "FlextValidatableMixin",
    "FlextValidation",
    "FlextValidationDecorators",
    "FlextValidationError",
    "FlextValidationRule",
    "FlextValidator",
    "FlextValidators",
    "FlextValueObject",
    "FlextValueObjectFactory",
    "FlextValueObjectMixin",
    "FlextVersionInfo",
    "P",
    "R",
    "Serializable",
    "ServiceKey",
    "T",
    "TAnyDict",
    "TAnyList",
    "TAnyObject",
    "TCommand",
    "TConfigDict",
    "TData",
    "TEntity",
    "TEntityId",
    "TErrorCode",
    "TErrorMessage",
    "TEvent",
    "TLogMessage",
    "TMessage",
    "TPredicate",
    "TQuery",
    "TRequest",
    "TRequestId",
    "TResponse",
    "TResult",
    "TService",
    "TTransformer",
    "TUserData",
    "TValue",
    "Timestamped",
    "U",
    "V",
    "Validatable",
    "ValidatedModel",
    "__version__",
    "chain",
    "check_python_compatibility",
    "compare_versions",
    "configure_flext_container",
    "create_log_context",
    "create_module_container_utilities",
    "create_module_exception_classes",
    "flext_core",
    "get_available_features",
    "get_flext_container",
    "get_logger",
    "get_typed",
    "get_version_info",
    "get_version_string",
    "get_version_tuple",
    "immutable",
    "is_dict_of",
    "is_feature_available",
    "make_builder",
    "make_factory",
    "merge_configs",
    "pure",
    "register_typed",
    "require_in_range",
    "require_non_empty",
    "require_not_none",
    "require_positive",
    "safe_call",
    "validate_version_format",
]
