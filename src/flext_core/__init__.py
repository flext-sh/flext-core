"""FLEXT Core foundation library.

Provides foundational patterns, types, and utilities for data integration.
All exports use FlextXXX naming convention for namespace safety.

Architecture:
    This is a pure foundation library providing patterns used across 32+ FLEXT projects.
    All exports follow a strict Flext* naming convention to avoid namespace collisions.

Reorganized Structure:
    - core_types.py: ALL type definitions and legacy T* aliases
    - core_exceptions.py: ALL exception patterns and error handling
    - core_config.py: ALL configuration models and management
    - core_mixins.py: ALL mixin patterns for behavioral composition
    - Other modules: handlers, commands, validation, utilities, etc.

Migration Notes:
    - All legacy imports remain functional via aliases
    - Prefer new core_* modules for new development
    - Old module references are maintained for backward compatibility
"""

# =============================================================================
# VERSION INFORMATION
# =============================================================================

from flext_core.__version__ import (
    __version__,
    FlextVersionInfo,
    FlextCompatibilityResult,
)

# =============================================================================
# CORE FOUNDATION PATTERNS - Consolidated modules (NEW)
# =============================================================================

# Core Types - Single source of truth for all types
from flext_core.core_types import (
    FlextTypes,
    # Type variables
    T,
    U,
    V,
    R,
    E,
    F,
    P,
    TComparable,
    TSerializable,
    TValidatable,
    TEntity,
    TAnyObject,
    TCommand,
    TQuery,
    TRequest,
    TResponse,
    TResult,
    TService,
    TOptional,
    EntryT,
    # Legacy T* aliases (comprehensive backward compatibility)
    TEntityId,
    TValue,
    TData,
    TConfig,
    TEvent,
    TMessage,
    TCommandId,
    TCommandType,
    THandlerName,
    TCommandPayload,
    TCommandResult,
    TCommandMetadata,
    TMiddlewareName,
    TValidationRule,
    TCommandBusId,
    TCommandPriority,
    TQueryId,
    TQueryType,
    TQueryResult,
    TQueryCriteria,
    TQueryProjection,
    TPaginationToken,
    TServiceName,
    TServiceKey,
    TFactory,
    TTransformer,
    TPredicate,
    TValidator,
    TCallable,
    TErrorHandler,
    TConnectionString,
    TLogMessage,
    TErrorCode,
    TErrorMessage,
    TAnyDict,
    TAnyList,
    TDict,
    TList,
    TStringDict,
    TUserData,
    TToken,
    TCredentials,
    TConnection,
    TUserId,
    TContextDict,
    TCorrelationId,
    TRequestId,
    TConfigDict,
    TFieldInfo,
    TFieldMetadata,
    # FlextTypes compatibility
    FlextEntityId,
    FlextSerializable,
    FlextValidatable,
    FlextValidator,
    # Protocol aliases
    Comparable,
    Serializable,
    Timestamped,
    Validatable,
    Cacheable,
    Configurable,
    # Utility functions
    get_centralized_types_usage_info,
)

# Core Exceptions - Single source of truth for all exceptions
from flext_core.core_exceptions import (
    # Abstract base classes
    FlextAbstractError,
    FlextAbstractValidationError,
    FlextAbstractBusinessError,
    FlextAbstractInfrastructureError,
    FlextAbstractConfigurationError,
    FlextAbstractErrorFactory,
    # Concrete exception classes
    FlextError,
    FlextValidationError,
    FlextTypeError,
    FlextAttributeError,
    FlextOperationError,
    FlextConfigurationError,
    FlextConnectionError,
    FlextAuthenticationError,
    FlextPermissionError,
    FlextNotFoundError,
    FlextAlreadyExistsError,
    FlextTimeoutError,
    FlextProcessingError,
    FlextCriticalError,
    # Factory and utilities
    FlextExceptions,
    create_module_exception_classes,
    create_context_exception_factory,
    get_exception_metrics,
    clear_exception_metrics,
    ERROR_CODES,
)

# Core Configuration - Single source of truth for all configuration
from flext_core.core_config import (
    # Abstract base classes and protocols
    FlextAbstractConfig,
    FlextAbstractSettings,
    FlextConfigValidatorProtocol,
    FlextConfigLoaderProtocol,
    FlextConfigMergerProtocol,
    FlextConfigSerializerProtocol,
    # Utility classes
    FlextConfigOperations,
    FlextConfigValidator,
    FlextConfigBuilder,
    FlextSettings,
    # Typed dictionaries
    FlextDatabaseConfigDict,
    FlextRedisConfigDict,
    FlextJWTConfigDict,
    FlextLDAPConfigDict,
    FlextOracleConfigDict,
    FlextSingerConfigDict,
    FlextObservabilityConfigDict,
    # Concrete configuration models
    FlextBaseConfigModel,
    FlextConfig,
    FlextDatabaseConfig,
    FlextRedisConfig,
    FlextJWTConfig,
    FlextOracleConfig,
    FlextLDAPConfig,
    FlextSingerConfig,
    FlextObservabilityConfig,
    FlextPerformanceConfig,
    FlextApplicationConfig,
    FlextDataIntegrationConfig,
    # Factory and management
    FlextConfigFactory,
    FlextConfigDefaults,
    FlextConfigOps,
    FlextConfigValidation,
    FlextConfigManager,
    # Legacy/compatibility functions
    safe_get_env_var,
    safe_load_json_file,
    merge_configs,
    # Constants
    DEFAULT_TIMEOUT,
    DEFAULT_RETRIES,
    DEFAULT_PAGE_SIZE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_ENVIRONMENT,
    CONFIG_VALIDATION_MESSAGES,
)

# Core Mixins - Single source of truth for all mixins
from flext_core.core_mixins import (
    # Abstract base mixins
    FlextAbstractMixin,
    FlextAbstractTimestampMixin,
    FlextAbstractIdentifiableMixin,
    FlextAbstractLoggableMixin,
    FlextAbstractValidatableMixin,
    FlextAbstractSerializableMixin,
    FlextAbstractEntityMixin,
    FlextAbstractServiceMixin,
    FlextAbstractTimingMixin,
    # Concrete mixins
    FlextTimestampMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextSerializableMixin,
    FlextComparableMixin,
    FlextCacheableMixin,
    # Composite mixins
    FlextEntityMixin,
    FlextValueObjectMixin,
    FlextCommandMixin,
    FlextServiceMixin,
    FlextDataMixin,
    FlextFullMixin,
    # Legacy compatibility
    LegacyCompatibleTimestampMixin,
    LegacyCompatibleIdentifiableMixin,
    LegacyCompatibleValidatableMixin,
    LegacyCompatibleSerializableMixin,
    LegacyCompatibleLoggableMixin,
    LegacyCompatibleTimingMixin,
    LegacyCompatibleComparableMixin,
    LegacyCompatibleCacheableMixin,
    LegacyCompatibleEntityMixin,
    LegacyCompatibleValueObjectMixin,
    LegacyCompatibleServiceMixin,
    LegacyCompatibleCommandMixin,
    LegacyCompatibleDataMixin,
    LegacyCompatibleFullMixin,
    # Utilities
    FlextValidators,
)

# =============================================================================
# LEGACY MODULE COMPATIBILITY - Maintain backward compatibility
# =============================================================================

# Essential patterns from legacy modules (backwards compatibility maintained)
from flext_core.result import FlextResult
from flext_core.container import (
    FlextContainer,
    FlextContainerUtils,
    configure_flext_container,
    create_module_container_utilities,
)

# Legacy modules - Import core functionality that wasn't consolidated
from flext_core.constants import (
    FlextConstants,
    FlextEnvironment,
    FlextLogLevel,
    FlextConnectionType,
    FlextDataFormat,
    FlextFieldType,
    FlextEntityStatus,
    FlextOperationStatus,
)

from flext_core.utilities import (
    FlextUtilities,
    FlextPerformance,
    FlextConversions,
    FlextTextProcessor,
    FlextTimeUtils,
    FlextIdGenerator,
    FlextGenerators,
    FlextTypeGuards,
    FlextBaseFactory,
    FlextGenericFactory,
    safe_call,
)

from flext_core.guards import (
    FlextGuards,
    ValidatedModel,
)

from flext_core.validation import (
    FlextValidation,
    FlextBaseValidator,
    FlextDomainValidator,
)

from flext_core.models import (
    FlextModel,
    FlextValue,
    FlextEntity,
    FlextFactory,
    FlextEntityFactory,
    FlextDomainEntity,
    FlextDomainValueObject,
    FlextAuth,
    FlextData,
    FlextObs,
    FlextDatabaseModel,
    FlextOperationModel,
    FlextOracleModel,
    FlextServiceModel,
    FlextSingerStreamModel,
)

from flext_core.value_objects import (
    FlextValueObject,
    FlextValueObjectFactory,
)

from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.domain_services import FlextDomainService

from flext_core.handlers import (
    FlextBaseHandler,
    FlextValidatingHandler,
    FlextAuthorizingHandler,
    FlextEventHandler,
    FlextMetricsHandler,
    FlextHandlerRegistry,
    FlextHandlerChain,
    FlextHandlers,
)

from flext_core.commands import FlextCommands

from flext_core.decorators import (
    FlextDecorators,
    FlextDecoratorUtils,
    FlextValidationDecorators,
    FlextErrorHandlingDecorators,
    FlextPerformanceDecorators,
    FlextFunctionalDecorators,
    FlextImmutabilityDecorators,
    FlextLoggingDecorators,
)

from flext_core.fields import (
    FlextFieldCore,
    FlextFieldMetadata,
    FlextFieldRegistry,
    FlextFields,
)

from flext_core.payload import (
    FlextPayload,
    FlextEvent,
    FlextMessage,
)

from flext_core.loggings import (
    FlextLogger,
    FlextLoggerFactory,
    FlextLogContextManager,
)

from flext_core.protocols import (
    FlextConnectionProtocol,
    FlextAuthProtocol,
    FlextObservabilityProtocol,
    FlextService,
    FlextMiddleware,
    FlextValidator as FlextValidatorProtocol,
    FlextValidationRule,
    FlextConfigurable as FlextConfigurableProtocol,
    FlextRepository,
    FlextUnitOfWork,
    FlextServiceFactory,
    FlextHandler,
    FlextMessageHandler,
    FlextEventProcessor,
    FlextMetricsCollector,
    FlextLoggerProtocol,
    FlextTracerProtocol,
    FlextMetricsProtocol,
    FlextPlugin,
    FlextPluginContext,
    FlextPluginLoader,
    FlextPluginRegistry,
)

from flext_core.observability import (
    FlextConsoleLogger,
    FlextNoOpTracer,
    FlextInMemoryMetrics,
    FlextSimpleObservability,
    FlextMinimalObservability,
    get_observability,
    get_console_logger,
    get_noop_tracer,
    get_memory_metrics,
)

# =============================================================================
# ESSENTIAL HELPER FUNCTIONS - Commonly used shortcuts
# =============================================================================

# Essential helper for container access (commonly used across ecosystem)
get_flext_container = FlextContainerUtils.get_flext_container

# Essential helper for logging (commonly used across ecosystem)
get_logger = FlextLoggerFactory.get_logger

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Only essential ones to avoid collisions
# =============================================================================

# Command pattern compatibility (expected by many tests)
FlextCommandHandler = FlextCommands.Handler
FlextCommandBus = FlextCommands.Bus
FlextQueryHandler = FlextCommands.QueryHandler

# Legacy module proxy for complex backward compatibility
import importlib as _importlib
from types import ModuleType


class _LegacyProxy:
    """Lazy proxy for legacy compatibility to avoid circular imports."""

    _legacy_module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._legacy_module is None:
            self._legacy_module = _importlib.import_module("flext_core.legacy")
        return self._legacy_module

    def __getattr__(self, name: str) -> object:
        module = self._load()
        return getattr(module, name)


legacy = _LegacyProxy()

# Legacy imports for factory functions used by tests/examples
from flext_core.legacy import (
    # Legacy compatibility classes
    LegacyBaseEntry,
    LegacyBaseProcessor,
    LegacyConsole,
    DecoratedFunction,
    # Legacy factory helpers
    create_database_config,
    create_redis_config,
    create_oracle_config,
    create_ldap_config,
    create_database_model,
    create_oracle_model,
    create_operation_model,
    create_service_model,
    create_singer_stream_model,
    create_base_handler,
    create_validating_handler,
    create_authorizing_handler,
    create_event_handler,
    create_metrics_handler,
    create_cross_service_message,
    create_cross_service_event,
    create_cache_decorator,
    create_safe_decorator,
    create_timing_decorator,
    create_validation_decorator,
    create_log_context,
)

# Backward-compatibility: provide _decorators_base alias expected by tests
import flext_core.base_decorators as _decorators_base

# =============================================================================
# EXPORTS - Clean, collision-free public API
# =============================================================================

__all__: list[str] = [
    # Version Information
    "__version__",
    "FlextVersionInfo",
    "FlextCompatibilityResult",
    # =========================================================================
    # CORE CONSOLIDATED MODULES (NEW - PREFERRED)
    # =========================================================================
    # Core Types (100+ exports from core_types.py)
    "FlextTypes",
    "T",
    "U",
    "V",
    "R",
    "E",
    "F",
    "P",
    "TComparable",
    "TSerializable",
    "TValidatable",
    "TEntity",
    "TAnyObject",
    "TCommand",
    "TQuery",
    "TRequest",
    "TResponse",
    "TResult",
    "TService",
    "TOptional",
    "EntryT",
    "TEntityId",
    "FlextEntityId",
    "TAnyDict",
    "TUserData",
    "TLogMessage",
    "TErrorMessage",
    "TConfigDict",
    "get_centralized_types_usage_info",
    # Protocol aliases
    "Comparable",
    "Serializable",
    "Timestamped",
    "Validatable",
    "Cacheable",
    "Configurable",
    "FlextSerializable",
    "FlextValidatable",
    "FlextValidator",
    # Core Exceptions (30+ exports from core_exceptions.py)
    "FlextError",
    "FlextValidationError",
    "FlextTypeError",
    "FlextAttributeError",
    "FlextOperationError",
    "FlextConfigurationError",
    "FlextConnectionError",
    "FlextAuthenticationError",
    "FlextPermissionError",
    "FlextNotFoundError",
    "FlextAlreadyExistsError",
    "FlextTimeoutError",
    "FlextProcessingError",
    "FlextCriticalError",
    "FlextExceptions",
    "create_module_exception_classes",
    "ERROR_CODES",
    # Core Configuration (50+ exports from core_config.py)
    "FlextConfig",
    "FlextSettings",
    "FlextConfigManager",
    "FlextConfigOps",
    "FlextConfigDefaults",
    "FlextConfigValidation",
    "FlextBaseConfigModel",
    "FlextDatabaseConfig",
    "FlextRedisConfig",
    "FlextJWTConfig",
    "FlextOracleConfig",
    "FlextLDAPConfig",
    "FlextSingerConfig",
    "FlextObservabilityConfig",
    "FlextConfigFactory",
    "merge_configs",
    "safe_get_env_var",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    # Core Mixins (50+ exports from core_mixins.py)
    "FlextCacheableMixin",
    "FlextComparableMixin",
    "FlextEntityMixin",
    "FlextFullMixin",
    "FlextIdentifiableMixin",
    "FlextLoggableMixin",
    "FlextSerializableMixin",
    "FlextServiceMixin",
    "FlextTimestampMixin",
    "FlextTimingMixin",
    "FlextValidatableMixin",
    "FlextValidators",
    "FlextValueObjectMixin",
    # =========================================================================
    # LEGACY BACKWARD COMPATIBILITY (MAINTAINED)
    # =========================================================================
    # Core Foundation Patterns
    "FlextResult",
    # Container System
    "FlextContainer",
    "FlextContainerUtils",
    "get_flext_container",
    "configure_flext_container",
    "create_module_container_utilities",
    # Constants
    "FlextConstants",
    "FlextEnvironment",
    "FlextLogLevel",
    "FlextConnectionType",
    "FlextDataFormat",
    "FlextFieldType",
    "FlextEntityStatus",
    "FlextOperationStatus",
    # Utilities
    "FlextUtilities",
    "FlextPerformance",
    "FlextConversions",
    "FlextTextProcessor",
    "FlextTimeUtils",
    "FlextIdGenerator",
    "FlextGenerators",
    "FlextTypeGuards",
    "FlextBaseFactory",
    "FlextGenericFactory",
    "safe_call",
    # Validation
    "FlextGuards",
    "ValidatedModel",
    "FlextValidation",
    "FlextBaseValidator",
    "FlextDomainValidator",
    # Domain Models
    "FlextModel",
    "FlextValue",
    "FlextEntity",
    "FlextFactory",
    "FlextEntityFactory",
    "FlextDomainEntity",
    "FlextDomainValueObject",
    "FlextAuth",
    "FlextData",
    "FlextObs",
    "FlextDatabaseModel",
    "FlextOperationModel",
    "FlextOracleModel",
    "FlextServiceModel",
    "FlextSingerStreamModel",
    "FlextValueObject",
    "FlextValueObjectFactory",
    "FlextAggregateRoot",
    "FlextDomainService",
    # Handlers
    "FlextBaseHandler",
    "FlextValidatingHandler",
    "FlextAuthorizingHandler",
    "FlextEventHandler",
    "FlextMetricsHandler",
    "FlextHandlerRegistry",
    "FlextHandlerChain",
    "FlextHandlers",
    # Commands
    "FlextCommands",
    "FlextCommandHandler",
    "FlextCommandBus",
    "FlextQueryHandler",
    # Decorators
    "FlextDecorators",
    "FlextDecoratorUtils",
    "FlextValidationDecorators",
    "FlextErrorHandlingDecorators",
    "FlextPerformanceDecorators",
    "FlextFunctionalDecorators",
    "FlextImmutabilityDecorators",
    "FlextLoggingDecorators",
    # Fields
    "FlextFieldCore",
    "FlextFieldMetadata",
    "FlextFieldRegistry",
    "FlextFields",
    # Payload/Events
    "FlextPayload",
    "FlextEvent",
    "FlextMessage",
    # Logging
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLogContextManager",
    "get_logger",
    # Protocols
    "FlextConnectionProtocol",
    "FlextService",
    "FlextMiddleware",
    "FlextAuthProtocol",
    "FlextObservabilityProtocol",
    "FlextValidatorProtocol",
    "FlextValidationRule",
    "FlextConfigurableProtocol",
    "FlextRepository",
    "FlextUnitOfWork",
    "FlextServiceFactory",
    "FlextHandler",
    "FlextMessageHandler",
    "FlextEventProcessor",
    "FlextMetricsCollector",
    "FlextLoggerProtocol",
    "FlextTracerProtocol",
    "FlextMetricsProtocol",
    "FlextPlugin",
    "FlextPluginContext",
    "FlextPluginLoader",
    "FlextPluginRegistry",
    # Observability
    "FlextConsoleLogger",
    "FlextNoOpTracer",
    "FlextInMemoryMetrics",
    "FlextSimpleObservability",
    "FlextMinimalObservability",
    "get_observability",
    "get_console_logger",
    "get_noop_tracer",
    "get_memory_metrics",
    # Legacy Compatibility
    "legacy",
    "_decorators_base",
    # Legacy Classes
    "LegacyBaseEntry",
    "LegacyBaseProcessor",
    "LegacyConsole",
    "DecoratedFunction",
    # Legacy Factory Helpers
    "create_database_config",
    "create_redis_config",
    "create_oracle_config",
    "create_ldap_config",
    "create_database_model",
    "create_oracle_model",
    "create_operation_model",
    "create_service_model",
    "create_singer_stream_model",
    "create_base_handler",
    "create_validating_handler",
    "create_authorizing_handler",
    "create_event_handler",
    "create_metrics_handler",
    "create_cross_service_message",
    "create_cross_service_event",
    "create_cache_decorator",
    "create_safe_decorator",
    "create_timing_decorator",
    "create_validation_decorator",
    "create_log_context",
]

# =============================================================================
# MIGRATION GUIDANCE
# =============================================================================


def get_migration_info() -> str:
    """Get information about the new consolidated structure."""
    return """
FLEXT Core Reorganization Complete!

NEW STRUCTURE (PREFERRED):
├── core_types.py     - ALL types and T* aliases (single source of truth)
├── core_exceptions.py - ALL exception patterns (single source of truth)
├── core_config.py    - ALL configuration models (single source of truth)
├── core_mixins.py    - ALL mixin patterns (single source of truth)
└── Legacy modules    - Maintained for backward compatibility

MIGRATION:
- ✅ All existing imports continue to work (100% backward compatible)
- 🆕 Prefer new core_* modules for new development
- 📦 Single consolidated files eliminate duplication
- 🚀 Better performance due to reduced circular imports

EXAMPLES:
# Old (still works):
from flext_core import FlextError, TAnyDict

# New (preferred):
from flext_core.core_types import FlextTypes, TAnyDict
from flext_core.core_exceptions import FlextError

# Best practice:
from flext_core import FlextError, FlextTypes  # Import from __init__.py
"""


__all__ += ["get_migration_info"]
