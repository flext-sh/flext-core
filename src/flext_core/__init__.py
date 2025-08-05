"""FLEXT Core - Foundation Layer Public API Gateway.

The architectural foundation that establishes consistent patterns across all 32 projects
in the FLEXT data integration ecosystem. This module serves as the public API gateway,
providing enterprise-grade patterns for Clean Architecture, Domain-Driven Design,
and railway-oriented programming.

Module Role in Architecture:
    Foundation Layer → Public API Gateway → Unified Import Interface

    This foundation module provides the unified public API that enables:
    - Consistent import patterns across all 32 ecosystem projects
    - Type system foundation with comprehensive generic type definitions
    - Version compatibility and feature detection for progressive enhancement
    - Namespace management with FlextXxx prefixing for conflict avoidance
    - Architectural layer organization with clear dependency management

Architecture Layers Export Organization:
    Foundation Layer: Types, constants, version management (this module)
    Core Pattern Layer: FlextResult, FlextContainer, exceptions, utilities
    Configuration Layer: Settings, logging, payload, interfaces
    Domain Layer: Entities, value objects, aggregates, domain services
    CQRS Layer: Commands, handlers, validation
    Extension Layer: Mixins, decorators, fields, guards

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: FlextResult, FlextContainer, Domain Patterns, Configuration
    🚧 In Development: Event Sourcing, Advanced CQRS, Plugin Architecture
    📋 Planned: Python-Go Bridge, Enterprise Observability

Core Patterns Exported:
    FlextResult[T]: Railway-oriented programming for type-safe error handling
    FlextContainer: Enterprise dependency injection with type safety
    FlextEntity: Rich domain entities with business logic and events
    FlextValueObject: Immutable value objects with validation
    FlextAggregateRoot: DDD aggregates with event sourcing (in development)
    FlextBaseSettings: Environment-aware configuration management

Ecosystem Usage Patterns:
    # FLEXT Service Railway-Oriented Programming
    from flext_core import FlextResult

    def process_user_data(data: dict) -> FlextResult[User]:
        return (
            validate_input(data)
            .map(transform_data)
            .flat_map(save_to_database)
            .map(format_response)
        )

    # Domain Modeling across 32 Projects
    from flext_core import FlextEntity

    class OracleInventoryItem(FlextEntity):
        sku: str
        quantity: int
        location: str

        def transfer_to_location(self, target: str) -> FlextResult[None]:
            if not self.validate_location(target):
                return FlextResult.fail(f"Invalid location: {target}")

            self.location = target
            self.add_domain_event({
                "type": "InventoryTransferred",
                "sku": self.sku,
                "target_location": target
            })
            return FlextResult.ok(None)

    # Singer Tap/Target Configuration (15 projects)
    from flext_core import FlextContainer, FlextDatabaseConfig

    container = get_flext_container()
    db_config = FlextDatabaseConfig(
        host="oracle-wms.enterprise.com",
        port=1521,
        username="tap_user",
        service_name="WMS_PROD"
    )
    container.register("oracle_config", db_config)

    # Cross-Service Messaging (flext-api, flexcore integration)
    from flext_core import FlextMessage, FlextEvent

    message_result = FlextMessage.create_message(
        "Oracle extraction completed",
        level="info",
        source="flext-tap-oracle-wms"
    )

    event_result = FlextEvent.create_event(
        "DataExtractionCompleted",
        {"records": 15000, "duration": 45.2},
        aggregate_id="extraction_001"
    )

Ecosystem Integration Benefits:
    - Used by all 32 FLEXT projects as architectural foundation
    - Provides consistent error handling across 15,000+ function signatures
    - Enables zero-downtime updates through semantic versioning
    - Supports enterprise production deployments with type safety
    - Cross-language compatibility for Python-Go bridge (FlexCore integration)

Quality Requirements:
    - Python 3.13+ only (no backward compatibility)
    - 95% test coverage minimum across all exported modules
    - Strict MyPy compliance with zero tolerance for type errors
    - Railway-oriented error handling throughout ecosystem
    - Comprehensive type annotations for compile-time safety

See Also:
    docs/python-module-organization.md: Complete module architecture guide
    docs/TODO.md: Development roadmap and gaps analysis
    examples/: 17 comprehensive working examples demonstrating patterns

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.commands import FlextCommands
from flext_core.config import (
    FlextBaseSettings,
    FlextConfig as FlextConfigLegacy,  # Avoid conflict with semantic FlextConfig
    FlextConfigDefaults,
    FlextConfigOps,
    FlextConfigValidation,
    merge_configs,
)
from flext_core.config_hierarchical import (
    # Hierarchical configuration system
    FlextHierarchicalConfigManager,
    create_application_project_config,
    create_infrastructure_project_config,
    create_integration_project_config,
    import_complete_config_system,
)
from flext_core.config_models import (
    # TypedDict definitions
    DatabaseConfigDict,
    # Core infrastructure models
    FlextApplicationConfig,
    FlextBaseConfigModel,
    FlextDatabaseConfig,
    # Settings classes
    FlextDatabaseSettings,
    FlextDataIntegrationConfig,
    FlextJWTConfig,
    FlextLDAPConfig,
    FlextObservabilityConfig,
    FlextOracleConfig,
    FlextRedisConfig,
    FlextRedisSettings,
    FlextSingerConfig,
    JWTConfigDict,
    LDAPConfigDict,
    ObservabilityConfigDict,
    OracleConfigDict,
    RedisConfigDict,
    SingerConfigDict,
    # Factory functions
    create_database_config,
    create_ldap_config,
    create_oracle_config,
    create_redis_config,
    # Utilities
    load_config_from_env,
    validate_config,
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
from flext_core.context import FlextContext
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
from flext_core.entities import FlextEntity as FlextEntityDeprecated, FlextEntityFactory
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
    create_context_exception_factory,
    create_module_exception_classes,
)
from flext_core.fields import FlextFieldCore, FlextFields

# Legacy flext_types compatibility imports (will be deprecated)
from flext_core.flext_types import (
    Comparable,
    FlextEntityId,
    Serializable,
    TAnyDict,
    TAnyList,
    TAnyObject,
    TCommand,
    TConfigDict,
    TConnectionString,
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
    TServiceName,
    TTransformer,
    TUserData,
    TValue,
    Validatable,
)

# ===== HARMONIZED FOUNDATION - New Single Source of Truth =====
# Phase 1: Import from new foundation.py (10 harmonized items)
from flext_core.foundation import (
    FlextConfig as FlextConfigHarmonized,
    # Semantic enums (4 items)
    FlextConnectionType,
    FlextDataFormat,
    FlextEntity as FlextEntityHarmonized,
    # Factory patterns (1 item)
    FlextFactory as FlextFactoryHarmonized,
    # Base classes (4 items)
    FlextModel as FlextModelHarmonized,
    FlextOperationStatus as FlextOperationStatusHarmonized,
    # Semantic namespace (1 item)
    FlextTypes as FlextTypesHarmonized,
    FlextValue as FlextValueHarmonized,
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
    FlextLogEntry,
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
from flext_core.models import (
    FlextAuth,
    # LEGACY MODELS - Backward compatibility (will be deprecated)
    # Base models
    FlextBaseModel,
    FlextConfig,
    # TypedDict definitions
    FlextConnectionDict,
    # Core semantic enums (moved to semantic_types.py)
    FlextData,
    # Configuration models
    FlextDatabaseModel,
    # Domain models
    FlextDomainEntity,
    FlextDomainValueObject,
    FlextEntity,
    FlextEntityDict,
    FlextEntityStatus,
    FlextFactory,
    FlextImmutableModel,
    # SEMANTIC PATTERN FOUNDATION - New Layer 0 types
    FlextModel,
    FlextMutableModel,
    FlextObs,
    FlextOperationDict,
    # Operation models
    FlextOperationModel,
    FlextOperationStatus,
    FlextOracleModel,
    # Service models
    FlextServiceModel,
    # Data integration models
    FlextSingerStreamModel,
    FlextValue,
    FlextValueObjectDict,
    # Factory functions
    create_database_model,
    create_operation_model,
    create_oracle_model,
    create_service_model,
    create_singer_stream_model,
    # Utilities
    model_to_dict_safe,
    validate_all_models,
)

# SEMANTIC PATTERN OBSERVABILITY - New Foundation Components
from flext_core.observability import (
    FlextLoggerProtocol,
    FlextObservabilityProtocol,
    configure_minimal_observability,
    get_observability,
)
from flext_core.payload import FlextEvent, FlextMessage, FlextPayload

# SEMANTIC PATTERN PROTOCOLS - Layer 1 contracts
from flext_core.protocols import (
    AuthProtocol,
    ConnectionProtocol,
    ObservabilityProtocol,
)
from flext_core.result import FlextResult, chain, safe_call

# Schema Processing - Reusable components for LDIF/Schema/ACL processing
from flext_core.schema_processing import (
    BaseConfigManager,
    BaseEntry,
    BaseFileWriter,
    BaseProcessor,
    BaseSorter,
    ConfigAttributeValidator,
    EntryType,
    EntryValidator,
    ProcessingPipeline,
    RegexProcessor,
)

# ===== UNIFIED SEMANTIC API - Single Source of Truth =====
# New harmonized semantic architecture (replaces all fragmented patterns)
from flext_core.semantic import (
    FlextBusinessError as FlextSemanticBusinessError,
    # Convenience exports for most common usage
    FlextError as FlextSemanticError,
    FlextSemantic,
    FlextValidationError as FlextSemanticValidationError,
    create_business_error as create_semantic_business_error,
    create_validation_error as create_semantic_validation_error,
)

# SEMANTIC TYPE SYSTEM - New hierarchical type organization (PREFERRED)
from flext_core.semantic_types import (
    AuthProtocol as SemanticAuthProtocol,
    ConnectionProtocol as SemanticConnectionProtocol,
    FlextTypeExtension,
    FlextTypeFactory,
    FlextTypes,
    ObservabilityProtocol as SemanticObservabilityProtocol,
    SingerProtocol,
)

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
from flext_core.testing_utilities import (
    create_api_test_response,
    create_ldap_test_config,
    create_oud_connection_config,
)

# Smart core removed - experimental features
# Smart validation removed - experimental features
# New standardized type system (Python 3.13) - Legacy compatibility
from flext_core.types import (
    E,
    F,
    FlextTypesCompat,
    P,
    R,
    T,
    TComparable,
    TSerializable,
    TValidatable,
    U,
    V,
    migrate_from_flext_types,
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
    "AuthProtocol",
    "BaseConfigManager",
    "BaseEntry",
    "BaseFileWriter",
    "BaseProcessor",
    "BaseSorter",
    "Comparable",
    "ConfigAttributeValidator",
    "ConnectionProtocol",
    "DatabaseConfigDict",
    "E",
    "EntryType",
    "EntryValidator",
    "F",
    "FlextAggregateRoot",
    "FlextAlreadyExistsError",
    "FlextApplicationConfig",
    "FlextAuth",
    "FlextAuthenticationError",
    "FlextBaseConfigModel",
    "FlextBaseModel",
    "FlextBaseSettings",
    "FlextCacheableMixin",
    "FlextCommands",
    "FlextComparableMixin",
    "FlextCompatibilityResult",
    "FlextConfig",
    "FlextConfigDefaults",
    "FlextConfigHarmonized",
    "FlextConfigLegacy",
    "FlextConfigOps",
    "FlextConfigValidation",
    "FlextConfigurable",
    "FlextConfigurationError",
    "FlextConnectionDict",
    "FlextConnectionError",
    "FlextConnectionType",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCore",
    "FlextCriticalError",
    "FlextData",
    "FlextDataFormat",
    "FlextDataIntegrationConfig",
    "FlextDatabaseConfig",
    "FlextDatabaseModel",
    "FlextDatabaseSettings",
    "FlextDecorators",
    "FlextDomainEntity",
    "FlextDomainService",
    "FlextDomainValueObject",
    "FlextEntity",
    "FlextEntityDeprecated",
    "FlextEntityDict",
    "FlextEntityFactory",
    "FlextEntityHarmonized",
    "FlextEntityId",
    "FlextEntityMixin",
    "FlextEntityStatus",
    "FlextEnvironment",
    "FlextError",
    "FlextErrorHandlingDecorators",
    "FlextEvent",
    "FlextEventPublisher",
    "FlextEventSubscriber",
    "FlextFactory",
    "FlextFactoryHarmonized",
    "FlextFieldCore",
    "FlextFieldType",
    "FlextFields",
    "FlextGenerators",
    "FlextHandler",
    "FlextHandlers",
    "FlextHierarchicalConfigManager",
    "FlextIdentifiableMixin",
    "FlextImmutabilityDecorators",
    "FlextImmutableModel",
    "FlextJWTConfig",
    "FlextLDAPConfig",
    "FlextLogContext",
    "FlextLogEntry",
    "FlextLogLevel",
    "FlextLoggableMixin",
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLoggerProtocol",
    "FlextLoggingDecorators",
    "FlextMessage",
    "FlextMiddleware",
    "FlextModel",
    "FlextModelHarmonized",
    "FlextMutableModel",
    "FlextNotFoundError",
    "FlextObs",
    "FlextObservabilityConfig",
    "FlextObservabilityProtocol",
    "FlextOperationDict",
    "FlextOperationError",
    "FlextOperationModel",
    "FlextOperationStatus",
    "FlextOperationStatusHarmonized",
    "FlextOracleConfig",
    "FlextOracleModel",
    "FlextPayload",
    "FlextPerformanceDecorators",
    "FlextPermissionError",
    "FlextPlugin",
    "FlextPluginContext",
    "FlextPredicates",
    "FlextProcessingError",
    "FlextRedisConfig",
    "FlextRedisSettings",
    "FlextRepository",
    "FlextResult",
    "FlextSemantic",
    "FlextSemanticBusinessError",
    "FlextSemanticError",
    "FlextSemanticValidationError",
    "FlextSerializableMixin",
    "FlextService",
    "FlextServiceModel",
    "FlextSingerAuthenticationError",
    "FlextSingerConfig",
    "FlextSingerConfigurationError",
    "FlextSingerConnectionError",
    "FlextSingerError",
    "FlextSingerProcessingError",
    "FlextSingerStreamModel",
    "FlextSingerValidationError",
    "FlextTapError",
    "FlextTargetError",
    "FlextTimeoutError",
    "FlextTimestampMixin",
    "FlextTimingMixin",
    "FlextTransformError",
    "FlextTypeError",
    "FlextTypeExtension",
    "FlextTypeFactory",
    "FlextTypes",
    "FlextTypesCompat",
    "FlextTypesHarmonized",
    "FlextUnitOfWork",
    "FlextUtilities",
    "FlextValidatableMixin",
    "FlextValidation",
    "FlextValidationDecorators",
    "FlextValidationError",
    "FlextValidationRule",
    "FlextValidator",
    "FlextValidators",
    "FlextValue",
    "FlextValueHarmonized",
    "FlextValueObject",
    "FlextValueObjectDict",
    "FlextValueObjectFactory",
    "FlextValueObjectMixin",
    "FlextVersionInfo",
    "JWTConfigDict",
    "LDAPConfigDict",
    "ObservabilityConfigDict",
    "ObservabilityProtocol",
    "OracleConfigDict",
    "P",
    "ProcessingPipeline",
    "R",
    "RedisConfigDict",
    "RegexProcessor",
    "SemanticAuthProtocol",
    "SemanticConnectionProtocol",
    "SemanticObservabilityProtocol",
    "SemanticOperationStatus",
    "Serializable",
    "ServiceKey",
    "SingerConfigDict",
    "SingerProtocol",
    "T",
    "TAnyDict",
    "TAnyList",
    "TAnyObject",
    "TCommand",
    "TComparable",
    "TConfigDict",
    "TConnectionString",
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
    "TSerializable",
    "TService",
    "TServiceName",
    "TTransformer",
    "TUserData",
    "TValidatable",
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
    "configure_minimal_observability",
    "create_api_test_response",
    "create_application_project_config",
    "create_context_exception_factory",
    "create_database_config",
    "create_database_model",
    "create_infrastructure_project_config",
    "create_integration_project_config",
    "create_ldap_config",
    "create_ldap_test_config",
    "create_log_context",
    "create_module_container_utilities",
    "create_module_exception_classes",
    "create_operation_model",
    "create_oracle_config",
    "create_oracle_model",
    "create_oud_connection_config",
    "create_redis_config",
    "create_semantic_business_error",
    "create_semantic_validation_error",
    "create_service_model",
    "create_singer_stream_model",
    "flext_core",
    "get_available_features",
    "get_flext_container",
    "get_logger",
    "get_observability",
    "get_typed",
    "get_version_info",
    "get_version_string",
    "get_version_tuple",
    "immutable",
    "import_complete_config_system",
    "is_dict_of",
    "is_feature_available",
    "load_config_from_env",
    "make_builder",
    "make_factory",
    "merge_configs",
    "migrate_from_flext_types",
    "model_to_dict_safe",
    "pure",
    "register_typed",
    "require_in_range",
    "require_non_empty",
    "require_not_none",
    "require_positive",
    "safe_call",
    "validate_all_models",
    "validate_config",
    "validate_version_format",
]
