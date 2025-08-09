"""FLEXT Core foundation library.

Provides foundational patterns, types, and utilities for data integration.
All exports use FlextXXX naming convention for namespace safety.

Legacy Support:
    For deprecated functions, import from flext_core.legacy (with warnings).
    Migrate to proper Flext* prefixed classes for production use.
"""

# =============================================================================
# VERSION INFORMATION
# =============================================================================

from flext_core.__version__ import __version__

# =============================================================================
# CORE RESULT PATTERN - Foundation for error handling
# =============================================================================

from flext_core.result import FlextResult

# =============================================================================
# CONTAINER - Dependency injection system
# =============================================================================

from flext_core.container import (
    FlextContainer,
    FlextContainerUtils,
    configure_flext_container,
)

# Essential helper for container access (commonly used across ecosystem)
get_flext_container = FlextContainerUtils.get_flext_container

# =============================================================================
# CONFIGURATION - Clean config management
# =============================================================================

from flext_core.config import (
    FlextConfig,
    FlextSettings,
    FlextConfigManager,
    FlextConfigOps,
    FlextConfigDefaults,
    FlextConfigValidation,
)

# =============================================================================
# CONFIGURATION MODELS - Configuration specific models
# =============================================================================

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
    FlextConfigFactory,  # Factory class for config creation
)

# =============================================================================
# CONSTANTS - Single source of truth
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
# TYPES - Modern type system with FlextTypes
# =============================================================================

from flext_core.typings import (
    FlextTypes,
    TEntityId,
    FlextEntityId,
    TAnyObject,
    TConfigDict,
)

# =============================================================================
# UTILITIES - SOLID-organized utility classes only
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
# GUARDS - Validation and type guards (organized in static class)
# =============================================================================

from flext_core.guards import (
    FlextGuards,
    ValidatedModel,
)

# =============================================================================
# VALIDATION - Comprehensive validation system
# =============================================================================

from flext_core.validation import (
    FlextValidation,
    FlextBaseValidator,
    FlextDomainValidator,
)

# =============================================================================
# DOMAIN MODELS - DDD patterns
# =============================================================================

from flext_core.models import (
    FlextModel,
    FlextValue,
    FlextEntity,
    FlextFactory,
    FlextDomainEntity,
    FlextDomainValueObject,  # Domain value object
    FlextAuth,
    FlextData,
    FlextDatabaseModel,
    FlextObs,  # Observability model
    FlextOperationModel,  # Operation model
    FlextOracleModel,  # Oracle model
    FlextServiceModel,  # Service model
)

# FlextEntityFactory moved to models.py following SOLID consolidation
from flext_core.value_objects import (
    FlextValueObject,
    FlextValueObjectFactory,
)
from flext_core.aggregate_root import FlextAggregateRoot

# =============================================================================
# DOMAIN SERVICES
# =============================================================================

from flext_core.domain_services import (
    FlextDomainService,
)

# =============================================================================
# HANDLERS - Event and command handlers (classes only)
# =============================================================================

from flext_core.handlers import (
    FlextBaseHandler,
    FlextValidatingHandler,
    FlextAuthorizingHandler,
    FlextEventHandler,
    FlextMetricsHandler,
    FlextHandlerRegistry,
    FlextHandlerChain,
)
from flext_core.handlers import FlextHandlers

# =============================================================================
# COMMANDS - CQRS command patterns
# =============================================================================

from flext_core.commands import (
    FlextCommands,
)

# Backward-compatibility aliases expected by tests
FlextCommandHandler = FlextCommands.Handler
FlextCommandBus = FlextCommands.Bus

# =============================================================================
# DECORATORS - Enterprise decorator patterns (organized in static classes)
# =============================================================================

from flext_core.decorators import (
    FlextDecorators,
    FlextDecoratorUtils,
    FlextValidationDecorators,
    FlextErrorHandlingDecorators,
    FlextPerformanceDecorators,
    FlextLoggingDecorators,
)

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
# FIELDS - Field definition and metadata
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
# LOGGING - Structured logging (essential helper function)
# =============================================================================

from flext_core.loggings import (
    FlextLogger,
    FlextLoggerFactory,
    FlextLogContextManager,
)

# Essential helper for logging (commonly used across ecosystem)
get_logger = FlextLoggerFactory.get_logger

# =============================================================================
# EXCEPTIONS - Comprehensive error handling
# =============================================================================

from flext_core.exceptions import (
    FlextError,
    FlextValidationError,
    FlextTypeError,
    FlextAttributeError,
    FlextOperationError,
    FlextConfigurationError,
)

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
)

# =============================================================================
# INTERFACES - Legacy compatibility
# =============================================================================

from flext_core import interfaces

# =============================================================================
# VERSION AND COMPATIBILITY
# =============================================================================

from flext_core.__version__ import (
    FlextVersionInfo,
    FlextCompatibilityResult,
)

# =============================================================================
# EXPORTS - Clean professional API (NO facades, NO duplications)
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Core Foundation Patterns
    "FlextResult",
    # Type System
    "TAnyObject",
    "TConfigDict",
    # Container System (with essential helper)
    "FlextContainer",
    "FlextContainerUtils",
    "get_flext_container",  # Essential helper
    "configure_flext_container",
    # Configuration Management
    "FlextConfig",
    "FlextSettings",
    "FlextConfigManager",
    "FlextConfigOps",
    "FlextConfigDefaults",
    "FlextConfigValidation",
    # Configuration Models
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
    "FlextConfigFactory",  # Factory class for config creation
    # Constants
    "FlextConstants",
    "FlextEnvironment",
    "FlextLogLevel",
    "FlextConnectionType",
    "FlextDataFormat",
    "FlextFieldType",
    "FlextEntityStatus",
    "FlextOperationStatus",
    # Modern Type System
    "FlextTypes",
    "TEntityId",
    "FlextEntityId",
    # Utilities (SOLID-organized classes)
    "FlextUtilities",
    "FlextPerformance",
    "FlextConversions",
    "FlextTextProcessor",
    "FlextTimeUtils",
    "FlextIdGenerator",
    "FlextTypeGuards",
    "FlextBaseFactory",
    "FlextGenericFactory",
    # Guards and Validation
    "FlextGuards",
    "ValidatedModel",
    "FlextValidation",
    "FlextBaseValidator",
    "FlextDomainValidator",
    # Domain Models (DDD)
    "FlextModel",
    "FlextValue",
    "FlextEntity",
    "FlextFactory",
    "FlextDomainEntity",
    "FlextDomainValueObject",  # Domain value object
    "FlextAuth",
    "FlextData",
    "FlextDatabaseModel",
    "FlextObs",  # Observability model
    "FlextOperationModel",  # Operation model
    "FlextOracleModel",  # Oracle model
    "FlextServiceModel",  # Service model
    # "FlextEntityFactory", # Moved to models.py
    "FlextValueObject",
    "FlextValueObjectFactory",
    "FlextAggregateRoot",
    # Domain Services
    "FlextDomainService",
    # Handlers (classes only)
    "FlextBaseHandler",
    "FlextValidatingHandler",
    "FlextAuthorizingHandler",
    "FlextEventHandler",
    "FlextMetricsHandler",
    "FlextHandlerRegistry",
    "FlextHandlerChain",
    "FlextHandlers",  # Backward compatibility alias
    # Commands
    "FlextCommands",
    # Decorators
    "FlextDecorators",
    "FlextDecoratorUtils",
    "FlextValidationDecorators",
    "FlextErrorHandlingDecorators",
    "FlextPerformanceDecorators",
    "FlextLoggingDecorators",
    # Mixins
    "FlextValidators",
    "FlextTimestampMixin",
    "FlextIdentifiableMixin",
    "FlextLoggableMixin",
    "FlextTimingMixin",
    "FlextCacheableMixin",
    "FlextComparableMixin",
    "FlextValidatableMixin",
    "FlextSerializableMixin",
    # Fields
    "FlextFieldCore",
    "FlextFieldMetadata",
    "FlextFieldRegistry",
    "FlextFields",
    # Payload/Events
    "FlextPayload",
    "FlextEvent",
    "FlextMessage",
    # Logging (with essential helper)
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLogContextManager",
    "get_logger",  # Essential helper
    # Exceptions
    "FlextError",
    "FlextValidationError",
    "FlextTypeError",
    "FlextAttributeError",
    "FlextOperationError",
    "FlextConfigurationError",
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
    # Interfaces (legacy compatibility)
    "interfaces",
    # Version Information
    "FlextVersionInfo",
    "FlextCompatibilityResult",
]
