"""FLEXT Core foundation library.

Provides foundational patterns, types, and utilities for data integration.
All exports use FlextXXX naming convention for namespace safety.

Legacy Support:
    For deprecated functions, import from flext_core.legacy (with warnings).
    Migrate to proper Flext* prefixed classes for production use.

Architecture:
    This is a pure foundation library providing patterns used across 32+ FLEXT projects.
    All exports follow strict Flext* naming convention to avoid namespace collisions.
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
# CORE FOUNDATION PATTERNS - Layer 0: Essential patterns
# =============================================================================

from flext_core.result import FlextResult
from flext_core.exceptions import (
    FlextError,
    FlextValidationError,
    FlextTypeError,
    FlextAttributeError,
    FlextOperationError,
    FlextConfigurationError,
)

# =============================================================================
# TYPE SYSTEM - Modern Python 3.13+ type definitions
# =============================================================================

from flext_core.typings import (
    FlextTypes,
    TEntityId,
    FlextEntityId,
    TAnyObject,
    TConfigDict,
    # Frequently used legacy aliases (compatibility)
    TAnyDict,
    TErrorMessage,
    TLogMessage,
    TUserData,
)

# =============================================================================
# CONSTANTS - Single source of truth for entire ecosystem
# =============================================================================

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

# =============================================================================
# DEPENDENCY INJECTION - Container system for service location
# =============================================================================

from flext_core.container import (
    FlextContainer,
    FlextContainerUtils,
    configure_flext_container,
    create_module_container_utilities,
)

# Essential helper for container access (commonly used across ecosystem)
get_flext_container = FlextContainerUtils.get_flext_container

# =============================================================================
# CONFIGURATION - Clean config management system
# =============================================================================

from flext_core.config import (
    FlextConfig,
    FlextSettings,
    FlextConfigManager,
    FlextConfigOps,
    FlextConfigDefaults,
    FlextConfigValidation,
)
from flext_core.config import merge_configs as merge_configs
from flext_core.config_models import (
    FlextBaseConfigModel,
    FlextDatabaseConfig,
    FlextRedisConfig,
    FlextJWTConfig,
    FlextOracleConfig,
    FlextLDAPConfig,
    FlextSingerConfig,
    FlextObservabilityConfig,
    FlextApplicationConfig,
    FlextDataIntegrationConfig,
    FlextConfigFactory,
)

# =============================================================================
# UTILITIES - SOLID-organized utility classes
# =============================================================================

from flext_core.utilities import (
    FlextUtilities,
    FlextPerformance,
    FlextConversions,
    FlextTextProcessor,
    FlextTimeUtils,
    FlextIdGenerator,
    FlextTypeGuards,
    FlextBaseFactory,
    FlextGenericFactory,
)

# =============================================================================
# VALIDATION - Comprehensive validation system
# =============================================================================

from flext_core.guards import (
    FlextGuards,
    ValidatedModel,
)
from flext_core.validation import (
    FlextValidation,
    FlextBaseValidator,
    FlextDomainValidator,
)

# =============================================================================
# DOMAIN MODELS - Domain-Driven Design patterns
# =============================================================================

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
    # Legacy compatibility models (for backward compatibility only)
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

# =============================================================================
# HANDLERS - CQRS pattern implementations
# =============================================================================

from flext_core.handlers import (
    FlextBaseHandler,
    FlextValidatingHandler,
    FlextAuthorizingHandler,
    FlextEventHandler,
    FlextMetricsHandler,
    FlextHandlerRegistry,
    FlextHandlerChain,
    FlextHandlers,  # Legacy compatibility facade
)

# =============================================================================
# COMMANDS - CQRS command patterns
# =============================================================================

from flext_core.commands import FlextCommands

# =============================================================================
# DECORATORS - Enterprise decorator patterns
# =============================================================================

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

# Backward-compatibility: provide _decorators_base alias expected by tests
import flext_core.base_decorators as _decorators_base  # noqa: F401

# =============================================================================
# MIXINS - Reusable behavior mixins
# =============================================================================

from flext_core.mixins import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextEntityMixin,
    FlextFullMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextServiceMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextValidators,
    FlextValueObjectMixin,
)

# =============================================================================
# FIELDS - Field definition and metadata system
# =============================================================================

from flext_core.fields import (
    FlextFieldCore,
    FlextFieldMetadata,
    FlextFieldRegistry,
    FlextFields,
)

# =============================================================================
# PAYLOAD & EVENTS - Message and event patterns
# =============================================================================

from flext_core.payload import (
    FlextPayload,
    FlextEvent,
    FlextMessage,
)

# =============================================================================
# LOGGING - Structured logging system
# =============================================================================

from flext_core.loggings import (
    FlextLogger,
    FlextLoggerFactory,
    FlextLogContextManager,
)

# Essential helper for logging (commonly used across ecosystem)
get_logger = FlextLoggerFactory.get_logger

# =============================================================================
# PROTOCOLS - Interface definitions
# =============================================================================

from flext_core.protocols import (
    FlextConnectionProtocol,
    FlextAuthProtocol,
    FlextObservabilityProtocol,
    FlextValidator,
    FlextValidationRule,
    FlextConfigurable,
    FlextRepository,
    FlextUnitOfWork,
    FlextServiceFactory,
    FlextHandler,
    FlextMessageHandler,
    FlextValidatingHandler as FlextValidatingHandlerProtocol,
    FlextAuthorizingHandler as FlextAuthorizingHandlerProtocol,
    FlextEventProcessor,
    FlextMetricsCollector,
    FlextLoggerProtocol,
    FlextTracerProtocol,
    FlextMetricsProtocol,
)

# =============================================================================
# OBSERVABILITY - Foundation observability implementations
# =============================================================================

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
# LEGACY COMPATIBILITY - Lazy import for backward compatibility
# =============================================================================

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


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Only essential ones to avoid collisions
# =============================================================================

# These aliases are kept for critical backward compatibility only
# All new code should use proper Flext* prefixed names

# Command pattern compatibility (expected by many tests)
FlextCommandHandler = FlextCommands.Handler
FlextCommandBus = FlextCommands.Bus
FlextQueryHandler = FlextCommands.QueryHandler


# =============================================================================
# LEGACY CLASSES AND HELPERS - Direct exports for easy access
# =============================================================================

# Import legacy classes and helpers directly for convenient access
from flext_core.legacy import (
    # Legacy compatibility classes
    LegacyBaseEntry,
    LegacyBaseProcessor,
    LegacyConsole,
    DecoratedFunction,
    # Legacy factory helpers - Config creation
    create_database_config,
    create_redis_config,
    create_oracle_config,
    create_ldap_config,
    # Legacy factory helpers - Model creation
    create_database_model,
    create_oracle_model,
    create_operation_model,
    create_service_model,
    create_singer_stream_model,
    # Legacy factory helpers - Handler creation
    create_base_handler,
    create_validating_handler,
    create_authorizing_handler,
    create_event_handler,
    create_metrics_handler,
    # Legacy factory helpers - Message/Event creation
    create_cross_service_message,
    create_cross_service_event,
    # Legacy factory helpers - Decorator creation
    create_cache_decorator,
    create_safe_decorator,
    create_timing_decorator,
    create_validation_decorator,
    # Legacy factory helpers - Logging
    create_log_context,
)

# =============================================================================
# EXPORTS - Clean, collision-free public API
# =============================================================================

__all__: list[str] = [
    # Version Information
    "__version__",
    "FlextVersionInfo",
    "FlextCompatibilityResult",
    # Core Foundation Patterns
    "FlextResult",
    # Exception Hierarchy
    "FlextError",
    "FlextValidationError",
    "FlextTypeError",
    "FlextAttributeError",
    "FlextOperationError",
    "FlextConfigurationError",
    # Type System
    "FlextTypes",
    "TEntityId",
    "FlextEntityId",
    "TAnyObject",
    "TConfigDict",
    # Common legacy aliases used across tests/examples
    "TAnyDict",
    "TErrorMessage",
    "TLogMessage",
    "TUserData",
    # Constants
    "FlextConstants",
    "FlextEnvironment",
    "FlextLogLevel",
    "FlextConnectionType",
    "FlextDataFormat",
    "FlextFieldType",
    "FlextEntityStatus",
    "FlextOperationStatus",
    # Container System
    "FlextContainer",
    "FlextContainerUtils",
    "get_flext_container",
    "configure_flext_container",
    "create_module_container_utilities",
    # Configuration Management
    "FlextConfig",
    "FlextSettings",
    "FlextConfigManager",
    "FlextConfigOps",
    "FlextConfigDefaults",
    "FlextConfigValidation",
    "merge_configs",
    "FlextBaseConfigModel",
    "FlextDatabaseConfig",
    "FlextRedisConfig",
    "FlextJWTConfig",
    "FlextOracleConfig",
    "FlextLDAPConfig",
    "FlextSingerConfig",
    "FlextObservabilityConfig",
    "FlextApplicationConfig",
    "FlextDataIntegrationConfig",
    "FlextConfigFactory",
    # Utilities
    "FlextUtilities",
    "FlextPerformance",
    "FlextConversions",
    "FlextTextProcessor",
    "FlextTimeUtils",
    "FlextIdGenerator",
    "FlextTypeGuards",
    "FlextBaseFactory",
    "FlextGenericFactory",
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
    # Decorators
    "FlextDecorators",
    "FlextDecoratorUtils",
    "FlextValidationDecorators",
    "FlextErrorHandlingDecorators",
    "FlextPerformanceDecorators",
    "FlextFunctionalDecorators",
    "FlextImmutabilityDecorators",
    "FlextLoggingDecorators",
    # Mixins
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
    "FlextAuthProtocol",
    "FlextObservabilityProtocol",
    "FlextValidator",
    "FlextValidationRule",
    "FlextConfigurable",
    "FlextRepository",
    "FlextUnitOfWork",
    "FlextServiceFactory",
    "FlextHandler",
    "FlextMessageHandler",
    "FlextValidatingHandlerProtocol",
    "FlextAuthorizingHandlerProtocol",
    "FlextEventProcessor",
    "FlextMetricsCollector",
    "FlextLoggerProtocol",
    "FlextTracerProtocol",
    "FlextMetricsProtocol",
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
    # Legacy compatibility classes (deprecated - migrate to Flext* classes)
    "LegacyBaseEntry",
    "LegacyBaseProcessor",
    "LegacyConsole",
    "DecoratedFunction",
    # Legacy factory helpers - Config creation (deprecated)
    "create_database_config",
    "create_redis_config",
    "create_oracle_config",
    "create_ldap_config",
    # Legacy factory helpers - Model creation (deprecated)
    "create_database_model",
    "create_oracle_model",
    "create_operation_model",
    "create_service_model",
    "create_singer_stream_model",
    # Legacy factory helpers - Handler creation (deprecated)
    "create_base_handler",
    "create_validating_handler",
    "create_authorizing_handler",
    "create_event_handler",
    "create_metrics_handler",
    # Legacy factory helpers - Message/Event creation (deprecated)
    "create_cross_service_message",
    "create_cross_service_event",
    # Legacy factory helpers - Decorator creation (deprecated)
    "create_cache_decorator",
    "create_safe_decorator",
    "create_timing_decorator",
    "create_validation_decorator",
    # Legacy factory helpers - Logging (deprecated)
    "create_log_context",
    # Add backward-compatibility aliases to exports
    "FlextCommandHandler",
    "FlextCommandBus",
    "FlextQueryHandler",
]
