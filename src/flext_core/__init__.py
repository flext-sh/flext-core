"""FLEXT Core foundation library."""

from __future__ import annotations

from types import ModuleType as _ModuleType

# Initialize public exports early so modules can safely extend/modify
# this list before the full export table is built. Defining it here
# prevents "used before definition" diagnostics from linters/type
# checkers when other imports mutate `__all__` during module init.
__all__: list[str] = []

# =============================================================================
# VERSION INFORMATION
# =============================================================================
from flext_core.__version__ import (
    FlextCompatibilityResult,
    FlextVersionInfo,
    __version__,
    check_python_compatibility,
    compare_versions,
    get_available_features,
    get_version_info,
    get_version_string,
    get_version_tuple,
    is_feature_available,
    validate_version_format,
)

__version_info__ = tuple(int(x) for x in __version__.split(".") if x.isdigit())

# =============================================================================
# CORE FOUNDATION PATTERNS
# =============================================================================

# Core APIs and utilities
from flext_core import interfaces

# Aggregate root
from flext_core.aggregate_root import FlextAggregateRoot

# Commands (minimal - only what exists)
from flext_core.commands import FlextCommands

# Configuration models - Only export what actually exists
from flext_core.config import (
    DEFAULT_ENVIRONMENT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PAGE_SIZE,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    CONFIG_VALIDATION_MESSAGES,
    FlextAbstractConfig,
    FlextConfig,
    FlextConfigDefaults,
    FlextConfigFactory,
    FlextConfigOps,
    FlextConfigValidation,
    FlextSystemDefaults,
    FlextDatabaseConfig,
    FlextBaseConfigModel,
    FlextJWTConfig,
    FlextLDAPConfig,
    FlextObservabilityConfig,
    FlextOracleConfig,
    FlextRedisConfig,
    FlextSettings,
    FlextSingerConfig,
    load_config_from_env,
    merge_configs,
    safe_get_env_var,
    safe_load_json_file,
    validate_config,
)

# Constants and enums
from flext_core.constants import (
    ERROR_CODES,
    MESSAGES,
    SERVICE_NAME_EMPTY,
    FlextConnectionType,
    FlextConstants,
    FlextDataFormat,
    FlextEntityStatus,
    FlextEnvironment,
    # Additional constants and enums
    FlextFieldType,
    FlextLogLevel,
    FlextOperationStatus,
)

# Performance constants
BYTES_PER_KB = FlextConstants.Performance.BYTES_PER_KB
SECONDS_PER_MINUTE = FlextConstants.Performance.SECONDS_PER_MINUTE
SECONDS_PER_HOUR = FlextConstants.Performance.SECONDS_PER_HOUR


# Container exports with direct imports for better type safety
from flext_core.container import (
    FlextContainer,
    FlextContainerUtils,
    FlextGlobalContainerManager,
    FlextServiceKey,
    FlextServiceRegistrar,
    FlextServiceRetriever,
    configure_flext_container,
    create_module_container_utilities,
    get_flext_container,
    get_typed,
    register_typed,
)

# Context
from flext_core.context import FlextContext

# Core
from flext_core.core import FlextCore, flext_core

# Decorators
from flext_core.decorators import (
    FlextAbstractDecorator,
    FlextAbstractErrorHandlingDecorator,
    FlextAbstractLoggingDecorator,
    FlextAbstractPerformanceDecorator,
    FlextAbstractValidationDecorator,
    FlextDecoratorFactory,
    FlextDecorators,
    FlextDecoratorUtils,
    FlextErrorHandlingDecorators,
    FlextFunctionalDecorators,
    FlextImmutabilityDecorators,
    FlextLoggingDecorators,
    FlextPerformanceDecorators,
    FlextValidationDecorators,
    # Internal decorator functions
    _flext_cache_decorator,
    _flext_safe_call_decorator,
    _flext_timing_decorator,
    _flext_validate_input_decorator,
    # Compatibility aliases
    _validate_input_decorator,
    _safe_call_decorator,
    _decorators_base,
    _BaseImmutabilityDecorators,
    _BaseDecoratorFactory,
)

# Delegation System
from flext_core.delegation_system import (
    FlextDelegatedProperty,
    FlextMixinDelegator,
    create_mixin_delegator,
    validate_delegation_system,
)

# Domain services
from flext_core.domain_services import (
    FlextDomainService,
    OperationType,
)

# Exception handling
from flext_core.exceptions import (
    FlextAbstractBusinessError,
    FlextAbstractConfigurationError,
    FlextAbstractError,
    FlextAbstractErrorFactory,
    FlextAbstractInfrastructureError,
    FlextAbstractValidationError,
    FlextAlreadyExistsError,
    FlextAttributeError,
    FlextAuthenticationError,
    FlextConfigurationError,
    FlextConnectionError,
    FlextCriticalError,
    FlextError,
    FlextExceptionMetrics,
    FlextExceptions,
    FlextNotFoundError,
    FlextOperationError,
    FlextPermissionError,
    FlextProcessingError,
    FlextTimeoutError,
    FlextTypeError,
    FlextValidationError,
    clear_exception_metrics,
    # Exception functions
    create_context_exception_factory,
    create_module_exception_classes,
    get_exception_metrics,
)

# Fields - Fixed imports using proper source
from flext_core.fields import (
    FlextFieldCore,
    FlextFieldMetadata,
    FlextFieldRegistry,
    FlextFields,
    flext_create_boolean_field,
    flext_create_integer_field,
    flext_create_string_field,
)


# Back-compat alias: expose FlextFieldCoreMetadata at module level
FlextFieldCoreMetadata = FlextFieldMetadata

# Import field types from fields module (now defined locally to avoid circular imports)
from flext_core.typings import (
    FlextFieldId,
    FlextFieldName,
    FlextFieldTypeStr,
    FlextCallable,
    FlextDecoratedFunction,
)

# Protocols
from flext_core.protocols import (
    FlextValidator,
    FlextValidationRule,
)

# Type-only imports were previously used here. Avoiding empty TYPE_CHECKING
# import blocks because they cause syntax errors and can interfere with
# Pydantic runtime behavior. Heavy typing-only symbols are imported from
# `flext_core.typings` at runtime below where necessary.


# Handlers
from flext_core.handlers import (
    FlextAbstractHandler,
    FlextAbstractHandlerChain,
    FlextAbstractHandlerRegistry,
    FlextAbstractMetricsHandler,
    FlextAuthorizingHandler,
    FlextBaseHandler,
    FlextEventHandler,
    FlextHandlerChain,
    FlextHandlerRegistry,
    FlextHandlers,
    FlextMetricsHandler,
    FlextValidatingHandler,
    HandlersFacade,
    FlextCommandHandler as _FlextCommandHandler,
    FlextQueryHandler as _FlextQueryHandler,
)
from flext_core.loggings import (
    FlextLogContext,
    FlextLogContextManager,
    FlextLogger,
    FlextLoggerFactory,
    create_log_context,
    flext_get_logger,
    get_logger,
)

# Mixins
from flext_core.mixins import (
    FlextAbstractEntityMixin,
    FlextAbstractIdentifiableMixin,
    FlextAbstractLoggableMixin,
    FlextAbstractMixin,
    FlextAbstractSerializableMixin,
    FlextAbstractServiceMixin,
    FlextAbstractTimestampMixin,
    FlextAbstractValidatableMixin,
    FlextCacheableMixin,
    FlextCommandMixin,
    FlextComparableMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    # Legacy compatibility aliases
    LegacyCompatibleTimestampMixin,
    LegacyCompatibleIdentifiableMixin,
    LegacyCompatibleValidatableMixin,
    LegacyCompatibleSerializableMixin,
    LegacyCompatibleLoggableMixin,
    LegacyCompatibleTimingMixin,
    LegacyCompatibleComparableMixin,
    LegacyCompatibleCacheableMixin,
    LegacyCompatibleEntityMixin,
    LegacyCompatibleCommandMixin,
    LegacyCompatibleDataMixin,
    LegacyCompatibleFullMixin,
    LegacyCompatibleServiceMixin,
    LegacyCompatibleValueObjectMixin,
)

# Domain models
from flext_core.models import (
    # Additional models
    DomainEventDict,
    FlextAuth,
    FlextConnectionDict,
    FlextData,
    FlextDatabaseModel,
    FlextEntity,
    FlextEntityDict,
    FlextFactory,
    FlextLegacyConfig,
    FlextModel,
    FlextObs,
    FlextOperationDict,
    FlextOperationModel,
    FlextOracleModel,
    FlextServiceModel,
    FlextSingerStreamModel,
    FlextValue,
    FlextValueObjectDict,
    #  Python 3.13 type aliases for JSON Schema
    JsonSchemaValue,
    JsonSchemaFieldInfo,
    JsonSchemaDefinition,
    FlextModelDict,
    FlextValidationContext,
    FlextFieldValidationInfo,
    # Model functions
    create_database_model,
    create_operation_model,
    create_oracle_model,
    create_service_model,
    model_to_dict_safe,
    validate_all_models,
)

# Observability
from flext_core.observability import (
    FlextConsoleLogger,
    FlextInMemoryMetrics,
    FlextMinimalObservability,
    FlextNoOpSpan,
    FlextNoOpTracer,
    FlextSimpleAlerts,
    FlextSimpleObservability,
    get_console_logger,
    get_global_observability,
    get_memory_metrics,
    get_noop_tracer,
    get_observability,
    get_simple_observability,
    reset_global_observability,
    # Legacy aliases
    InMemoryMetrics,
    MinimalObservability,
    NoOpTracer,
    SimpleAlerts,
)

# Legacy compatibility imports
from flext_core.legacy import (
    BaseConfigManager,
    ConsoleLogger,
    FlextValueObjectFactory,
    # Legacy schema processing aliases
    BaseEntry,
    BaseFileWriter,
    BaseProcessor,
    BaseSorter,
    ConfigAttributeValidator,
    EntryType,
    # NOTE: Version utilities are imported from __version__.py above, not legacy.py
)

# Payload
from flext_core.payload import (
    FlextEvent,
    FlextMessage,
    FlextPayload,
    create_cross_service_event,
    create_cross_service_message,
    get_serialization_metrics,
    validate_cross_service_protocol,
)

# Protocols
from flext_core.protocols import (
    FlextAlertsProtocol,
    FlextConfigurable as FlextConfigurableProtocol,
    FlextLoggerProtocol,
    FlextMetricsProtocol,
    FlextObservabilityProtocol,
    FlextPlugin,
    FlextPluginContext,
    FlextPluginLoader,
    FlextPluginRegistry,
    FlextRepository,
    FlextTracerProtocol,
    FlextValidator as FlextValidatorProtocol,
)

# Primary foundation classes
from flext_core.result import FlextResult

# RootModel patterns
from flext_core.root_models import (
    FlextEntityId,
    FlextVersion,
    FlextTimestamp,
    FlextMetadata,
    FlextEventList,
    FlextHost,
    FlextPort,
    FlextConnectionString,
    FlextEmailAddress,
    FlextServiceName,
    FlextPercentage,
    FlextErrorCode,
    FlextErrorMessage,
    create_entity_id,
    create_version,
    create_email,
    create_service_name,
    create_host_port,
    from_legacy_dict,
    to_legacy_dict,
)

# Schema Processing
from flext_core.schema_processing import (
    FlextBaseConfigManager,
    FlextBaseEntry,
    FlextBaseFileWriter,
    FlextBaseProcessor,
    FlextBaseSorter,
    FlextConfigAttributeValidator,
    FlextEntryType,
    FlextEntryValidator,
    FlextProcessingPipeline,
    FlextRegexProcessor,
    ProcessingPipeline,
    # Legacy alias
    RegexProcessor,
)

# Semantic
from flext_core.semantic import (
    FlextSemantic,
    FlextSemanticError,
    FlextSemanticModel,
    FlextSemanticObservability,
)


# Type definitions - complete set
from flext_core.typings import (
    E,
    F,
    FlextTypes,
    P,
    R,
    # Common type variables
    T,
    TAnyDict,
    TAnyList,
    # Additional type variables
    TAnyObject,
    TCallable,
    TCommand,
    TCommandBusId,
    TCommandId,
    TCommandMetadata,
    TCommandPayload,
    TCommandPriority,
    TCommandResult,
    TCommandType,
    TComparable,
    TConfig,
    TConfigDict,
    TConnection,
    # Infrastructure types
    TConnectionString,
    # Service types
    TCorrelationId,
    TCredentials,
    TData,
    TDict,
    TEntity,
    TEntityId,
    TErrorCode,
    TErrorHandler,
    TErrorMessage,
    # CQRS types
    TEvent,
    # Callable types
    TFactory,
    THandlerName,
    TList,
    # Message and error types
    TLogMessage,
    TMessage,
    TMiddlewareName,
    TOptional,
    TPaginationToken,
    TPredicate,
    TQuery,
    TQueryCriteria,
    TQueryId,
    TQueryProjection,
    TQueryResult,
    TQueryType,
    TRequest,
    TResponse,
    TResult,
    TSerializable,
    TService,
    TServiceKey,
    # Service types
    TServiceName,
    TStringDict,
    # Auth types
    TToken,
    TTransformer,
    TUserData,
    TUserId,
    TValidatable,
    TValidationRule,
    TValidator,
    TValue,
    # Business types (testing convenience)
    TBusinessCode,
    TBusinessId,
    TBusinessName,
    TBusinessStatus,
    TBusinessType,
    # Cache types
    TCacheKey,
    TCacheTTL,
    TCacheValue,
    # File and request types
    # Field ids and names used at runtime
    U,
    V,
    # Domain-level TypeVars
)

# Additional utilities
from flext_core.utilities import (
    Console,
    FlextBaseFactory,
    FlextConsole,
    FlextConversions,
    FlextFormatters,
    FlextGenerators,
    FlextGenericFactory,
    FlextIdGenerator,
    FlextPerformance,
    FlextTextProcessor,
    FlextTimeUtils,
    FlextTypeGuards,
    FlextUtilities,
    FlextUtilityFactory,
    flext_clear_performance_metrics,
    flext_get_performance_metrics,
    flext_record_performance,
    flext_safe_int_conversion,
    flext_track_performance,
    generate_correlation_id,
    generate_id,
    generate_iso_timestamp,
    generate_uuid,
    is_not_none,
    safe_call,
    safe_int_conversion_with_default,
    truncate,
)

# Validation
from flext_core.validation import (
    FlextDomainValidator,
    FlextPredicates,
    FlextValidation,
    FlextValidators,
    flext_validate_email,
    flext_validate_non_empty_string,
    flext_validate_numeric,
    flext_validate_required,
    flext_validate_service_name,
    flext_validate_string,
)

# Guard utilities (exported in __all__)
from flext_core.guards import (
    FlextGuards,
    FlextValidatedModel,
    FlextValidationUtils,
    ValidatedModel,
    immutable,
    is_dict_of,
    is_instance_of,
    is_list_of,
    make_builder,
    make_factory,
    pure,
    require_in_range,
    require_non_empty,
    require_not_none,
    require_positive,
    safe,
    validated,
)

# Value objects
from flext_core.value_objects import FlextValueObject

# Type adapters
from flext_core.type_adapters import ValidationAdapters


# Direct import for better type safety
from flext_core import constants

with suppress(
    Exception,
):  # pragma: no cover
    pass


# NOTE: Command/Query handler abstract classes are provided by
# `flext_core.handlers` module. Avoid redefining them here to prevent
# type identity mismatches between runtime classes and `.pyi` stubs
# which cause mypy/pyright errors about incompatible method overrides.
# Re-export the concrete definitions from the handlers module instead.


# =============================================================================
# CLEAN ARCHITECTURE IMPLEMENTATION
# =============================================================================
""" FLEXT Core foundation library."""

_module_type_ref: type[_ModuleType] | None = _ModuleType

# =============================================================================
# EXPORTS - Clean, collision-free public API
# =============================================================================

__all__ += [
    "__version__",
    "__version_info__",
    "FlextVersionInfo",
    "FlextCompatibilityResult",
    "check_python_compatibility",
    "compare_versions",
    "get_available_features",
    "get_version_info",
    "get_version_string",
    "get_version_tuple",
    "is_feature_available",
    "validate_version_format",
    "FlextResult",
    "FlextContainer",
    "get_flext_container",
    "FlextServiceKey",
    "FlextServiceRegistrar",
    "FlextServiceRetriever",
    "FlextGlobalContainerManager",
    "FlextContainerUtils",
    "configure_flext_container",
    "get_typed",
    "register_typed",
    "create_module_container_utilities",
    "get_logger",
    "create_log_context",
    "flext_get_logger",
    "FlextLogContext",
    "FlextLogContextManager",
    "FlextLogger",
    "FlextLoggerFactory",
    "safe_call",
    "FlextConfig",
    "FlextLDAPConfig",
    "FlextDatabaseConfig",
    "FlextRedisConfig",
    "FlextJWTConfig",
    "FlextOracleConfig",
    "FlextSingerConfig",
    "FlextBaseConfigModel",
    "FlextSettings",
    "FlextModel",
    # Additional config classes
    "FlextAbstractConfig",
    "FlextConfigDefaults",
    "FlextSystemDefaults",
    "FlextConfigFactory",
    "FlextConfigOps",
    "FlextConfigValidation",
    "FlextObservabilityConfig",
    "merge_configs",
    "load_config_from_env",
    "safe_get_env_var",
    "safe_load_json_file",
    "validate_config",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_ENVIRONMENT",
    "CONFIG_VALIDATION_MESSAGES",
    "FlextFieldType",
    "FlextConnectionType",
    "FlextDataFormat",
    "ERROR_CODES",
    "MESSAGES",
    "SERVICE_NAME_EMPTY",
    "FlextEntity",
    "FlextValue",
    "FlextFactory",
    "DomainEventDict",
    "FlextData",
    "FlextAuth",
    "FlextObs",
    "FlextEntityDict",
    "FlextValueObjectDict",
    "FlextOperationDict",
    "FlextConnectionDict",
    "FlextDatabaseModel",
    "FlextOracleModel",
    "FlextLegacyConfig",
    "FlextOperationModel",
    "FlextServiceModel",
    "FlextSingerStreamModel",
    #  Python 3.13 type aliases
    "JsonSchemaValue",
    "JsonSchemaFieldInfo",
    "JsonSchemaDefinition",
    "FlextModelDict",
    "FlextValidationContext",
    "FlextFieldValidationInfo",
    "create_database_model",
    "create_oracle_model",
    "create_operation_model",
    "create_service_model",
    "validate_all_models",
    "model_to_dict_safe",
    "FlextTypes",
    "FlextEntityId",
    "TEntityId",
    "T",
    "U",
    "V",
    "R",
    "E",
    "F",
    "P",
    "TEntity",
    "TValue",
    "TData",
    "TConfig",
    "TConfigDict",
    "TAnyDict",
    "TDict",
    "TList",
    "TStringDict",
    "TAnyObject",
    "TCommand",
    "TQuery",
    "TRequest",
    "TResponse",
    "TResult",
    "TService",
    "TOptional",
    "TComparable",
    "TSerializable",
    "TValidatable",
    "TLogMessage",
    "TErrorMessage",
    "TUserData",
    "TErrorCode",
    "TErrorHandler",
    "TEvent",
    "TMessage",
    "TCommandId",
    "TCommandType",
    "THandlerName",
    "TCommandPayload",
    "TCommandResult",
    "TCommandMetadata",
    "TMiddlewareName",
    "TValidationRule",
    "TCommandBusId",
    "TCommandPriority",
    "TQueryId",
    "TQueryType",
    "TQueryResult",
    "TQueryCriteria",
    "TQueryProjection",
    "TPaginationToken",
    "TServiceName",
    "TServiceKey",
    "TFactory",
    "TTransformer",
    "TPredicate",
    "TValidator",
    "TCallable",
    "TConnectionString",
    "TAnyList",
    "TToken",
    "TCredentials",
    "TConnection",
    "TUserId",
    "TCorrelationId",
    # Business types (testing convenience)
    "TBusinessCode",
    "TBusinessId",
    "TBusinessName",
    "TBusinessStatus",
    "TBusinessType",
    # Cache types (testing convenience)
    "TCacheKey",
    "TCacheValue",
    "TCacheTTL",
    "FlextExceptions",
    "FlextExceptionMetrics",
    "FlextAbstractError",
    "FlextAbstractValidationError",
    "FlextAbstractBusinessError",
    "FlextAbstractInfrastructureError",
    "FlextAbstractConfigurationError",
    "FlextAbstractErrorFactory",
    "FlextError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextProcessingError",
    "FlextValidationError",
    "FlextTimeoutError",
    "FlextAuthenticationError",
    "FlextPermissionError",
    "FlextNotFoundError",
    "FlextAlreadyExistsError",
    "FlextTypeError",
    "FlextAttributeError",
    "FlextOperationError",
    "FlextCriticalError",
    "create_context_exception_factory",
    "create_module_exception_classes",
    "get_exception_metrics",
    "clear_exception_metrics",
    "FlextLogLevel",
    "FlextConstants",
    "FlextEnvironment",
    "FlextOperationStatus",
    "FlextEntityStatus",
    "FlextValueObject",
    "FlextDomainService",
    "OperationType",
    "FlextAggregateRoot",
    "FlextValidation",
    "FlextCommands",
    "FlextValidators",
    "FlextPredicates",
    "FlextDomainValidator",
    "flext_validate_required",
    "flext_validate_string",
    "flext_validate_numeric",
    "flext_validate_email",
    "flext_validate_non_empty_string",
    "flext_validate_service_name",
    "FlextDecorators",
    "FlextHandlers",
    "FlextAbstractHandler",
    "FlextAbstractHandlerChain",
    "FlextAbstractHandlerRegistry",
    "FlextAbstractMetricsHandler",
    "FlextBaseHandler",
    "FlextAuthorizingHandler",
    "FlextEventHandler",
    "FlextMetricsHandler",
    "FlextHandlerChain",
    "FlextHandlerRegistry",
    "FlextValidatingHandler",
    "HandlersFacade",
    "FlextCommandHandler",
    "FlextQueryHandler",
    "FlextUtilities",
    "FlextIdGenerator",
    "FlextGenerators",
    "FlextPayload",
    "FlextMessage",
    "FlextEvent",
    "create_cross_service_event",
    "create_cross_service_message",
    "get_serialization_metrics",
    "validate_cross_service_protocol",
    "FlextFieldCore",
    "FlextFieldCoreMetadata",
    "FlextFieldId",
    "FlextFieldMetadata",
    "FlextFieldName",
    "FlextFieldRegistry",
    "FlextFieldTypeStr",
    "FlextCallable",
    "FlextDecoratedFunction",
    "FlextFields",
    "FlextValidator",
    "FlextValidationRule",
    "flext_create_boolean_field",
    "flext_create_integer_field",
    "flext_create_string_field",
    "FlextCore",
    "flext_core",
    # RootModel patterns
    "FlextEntityId",
    "FlextVersion",
    "FlextTimestamp",
    "FlextMetadata",
    "FlextEventList",
    "FlextHost",
    "FlextPort",
    "FlextConnectionString",
    "FlextEmailAddress",
    "FlextServiceName",
    "FlextPercentage",
    "FlextErrorCode",
    "FlextErrorMessage",
    "create_entity_id",
    "create_version",
    "create_email",
    "create_service_name",
    "create_host_port",
    "from_legacy_dict",
    "to_legacy_dict",
    "FlextAbstractMixin",
    "FlextAbstractTimestampMixin",
    "FlextAbstractIdentifiableMixin",
    "FlextAbstractLoggableMixin",
    "FlextAbstractValidatableMixin",
    "FlextAbstractSerializableMixin",
    "FlextAbstractEntityMixin",
    "FlextAbstractServiceMixin",
    "FlextTimestampMixin",
    "FlextIdentifiableMixin",
    "FlextLoggableMixin",
    "FlextTimingMixin",
    "FlextValidatableMixin",
    "FlextSerializableMixin",
    "FlextComparableMixin",
    "FlextCacheableMixin",
    "FlextCommandMixin",
    #  mixins - no legacy compatibility needed
    "FlextConsoleLogger",
    "FlextNoOpSpan",
    "FlextNoOpTracer",
    "FlextInMemoryMetrics",
    "FlextSimpleObservability",
    "FlextSimpleAlerts",
    "FlextMinimalObservability",
    "get_console_logger",
    "get_noop_tracer",
    "get_memory_metrics",
    "get_simple_observability",
    "get_global_observability",
    "reset_global_observability",
    "get_observability",
    "FlextContext",
    "FlextGuards",
    "FlextValidatedModel",
    "FlextValidationUtils",
    "ValidatedModel",
    "immutable",
    "is_dict_of",
    "is_instance_of",
    "is_list_of",
    "is_not_none",
    "make_builder",
    "make_factory",
    "pure",
    "require_in_range",
    "require_non_empty",
    "require_not_none",
    "require_positive",
    "safe",
    "validated",
    "FlextSemanticModel",
    "FlextSemanticObservability",
    "FlextSemanticError",
    "FlextSemantic",
    "FlextEntryType",
    "FlextBaseEntry",
    "FlextEntryValidator",
    "FlextBaseProcessor",
    "FlextRegexProcessor",
    "FlextConfigAttributeValidator",
    "FlextBaseConfigManager",
    "FlextBaseSorter",
    "FlextBaseFileWriter",
    "FlextProcessingPipeline",
    "FlextPlugin",
    "FlextPluginContext",
    "FlextRepository",
    "FlextAlertsProtocol",
    "FlextConfigurableProtocol",
    "FlextLoggerProtocol",
    "FlextMetricsProtocol",
    "FlextObservabilityProtocol",
    "FlextTracerProtocol",
    "FlextValidatorProtocol",
    "FlextPluginLoader",
    "FlextPluginRegistry",
    # Protocol interfaces module
    "interfaces",
    "annotations",
    # Decorators adicionales
    "FlextAbstractDecorator",
    "FlextAbstractErrorHandlingDecorator",
    "FlextAbstractLoggingDecorator",
    "FlextAbstractPerformanceDecorator",
    "FlextAbstractValidationDecorator",
    "FlextDecoratorFactory",
    "FlextDecoratorUtils",
    "FlextValidationDecorators",
    "FlextErrorHandlingDecorators",
    "FlextPerformanceDecorators",
    "FlextLoggingDecorators",
    "FlextImmutabilityDecorators",
    "FlextFunctionalDecorators",
    "FlextDecoratorFactory",
    "_flext_cache_decorator",
    "_flext_safe_call_decorator",
    "_flext_timing_decorator",
    "_flext_validate_input_decorator",
    # Compatibility aliases
    "_validate_input_decorator",
    "_safe_call_decorator",
    "_decorators_base",
    "_BaseImmutabilityDecorators",
    "_BaseDecoratorFactory",
    # Utilities adicionales
    "FlextConsole",
    "Console",
    "FlextDecoratedFunction",
    "FlextPerformance",
    "FlextConversions",
    "FlextTextProcessor",
    "FlextTimeUtils",
    "FlextTypeGuards",
    "FlextFormatters",
    "FlextBaseFactory",
    "FlextGenericFactory",
    "FlextUtilityFactory",
    "flext_safe_int_conversion",
    "generate_correlation_id",
    "safe_int_conversion_with_default",
    "flext_clear_performance_metrics",
    "generate_id",
    "generate_uuid",
    "is_not_none",
    "truncate",
    "flext_get_performance_metrics",
    "flext_record_performance",
    "flext_track_performance",
    "generate_iso_timestamp",
    # Performance constants
    "BYTES_PER_KB",
    "SECONDS_PER_MINUTE",
    "SECONDS_PER_HOUR",
    # Legacy compatibility exports
    "ConsoleLogger",
    "BaseConfigManager",
    "LegacyCompatibleCacheableMixin",
    "FlextValueObjectFactory",
    # Additional legacy mixin aliases
    "LegacyCompatibleCommandMixin",
    "LegacyCompatibleComparableMixin",
    "LegacyCompatibleDataMixin",
    "LegacyCompatibleEntityMixin",
    "LegacyCompatibleFullMixin",
    "LegacyCompatibleIdentifiableMixin",
    "LegacyCompatibleLoggableMixin",
    "LegacyCompatibleSerializableMixin",
    "LegacyCompatibleServiceMixin",
    "LegacyCompatibleTimestampMixin",
    "LegacyCompatibleTimingMixin",
    "LegacyCompatibleValidatableMixin",
    "LegacyCompatibleValueObjectMixin",
    # Legacy observability aliases
    "InMemoryMetrics",
    "MinimalObservability",
    "NoOpTracer",
    "SimpleAlerts",
    # Legacy schema processing aliases
    "BaseEntry",
    "BaseFileWriter",
    "BaseProcessor",
    "BaseSorter",
    "ConfigAttributeValidator",
    "EntryType",
    "ProcessingPipeline",
    "RegexProcessor",
    # NOTE: Version utilities are exported from the main version imports above,
    # not legacy
    # Delegation System
    "FlextDelegatedProperty",
    "FlextMixinDelegator",
    "create_mixin_delegator",
    "validate_delegation_system",
    # Additional compatibility exports
    "FlextAbstractConfig",
    #  aliases that match the updated pattern
    "FlextValue",
    # Type adapters
    "ValidationAdapters",
]

# Public re-exports for compatibility
FlextCommandHandler = _FlextCommandHandler
FlextQueryHandler = _FlextQueryHandler
