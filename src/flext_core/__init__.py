"""FLEXT Core - Enterprise data integration foundation library.

This module provides the architectural foundation for the FLEXT ecosystem with type-safe
error handling, dependency injection, domain modeling patterns, and enterprise-grade
functionality for data integration platforms.

Architecture:
    Foundation Layer: FlextResult, exceptions, constants, types
    Domain Layer: FlextEntity, FlextValueObject, FlextAggregateRoot, domain services
    Application Layer: Commands (CQRS), handlers, validation, guards
    Infrastructure Layer: Container (DI), config, logging, observability, context
    Support Layer: Decorators, mixins, utilities, fields, services, adapters

Core Components:
    FlextAggregates: DDD aggregate pattern with domain event management
    FlextCommands: CQRS command patterns with validation and result handling
    FlextConfig: Pydantic-based configuration with environment variable support
    FlextConstants: Application constants, enums, and configuration defaults
    FlextContainer: Type-safe dependency injection container with factory support
    FlextContext: Request/operation context management with correlation IDs
    FlextCore: Core functionality for versioning, validation, and type management
    FlextDecorators: Enterprise decorators for validation, caching, and metrics
    FlextDelegationSystem: Delegation system for method chaining and composition
    FlextDomainServices: Domain services for business logic and operations
    FlextExceptions: Hierarchical exception system with error codes and metrics
    FlextFields: Field validation and metadata for forms and schemas
    FlextGuards: Type guards and runtime validation utilities
    FlextHandlers: Request/response handlers with result composition
    FlextLogger: Structured logging with context propagation and JSON output
    FlextMixins: Reusable behavior patterns (timestamps, serialization, etc.)
    FlextObservability: Metrics, tracing, and monitoring abstractions
    FlextProcessors: Processing utilities for schema and data validation
    FlextProtocols: Interface definitions and contracts for dependency injection
    FlextResult[T]: Railway-oriented error handling with map/flat_map/bind operations
    FlextServices: Service layer abstractions and patterns
    FlextTypeAdapters: Type adapters for data transformation and validation
    FlextTypes: Type definitions, aliases, and generic type parameters
    FlextUtilities: Helper functions, generators, and type utilities
    FlextValidation: Composable validation framework with predicate logic


Examples:
    Railway-oriented programming:
    >>> result = (
    ...     FlextResult.ok({"user": "john"})
    ...     .map(lambda d: validate_user(d))
    ...     .flat_map(lambda u: save_user(u))
    ...     .map_error(lambda e: f"Operation failed: {e}")
    ... )
    >>> if result.success:
    ...     user = result.value

    Dependency injection:
    >>> container = get_flext_container()
    >>> container.register("database", DatabaseService())
    >>> db_result = container.get("database")
    >>> if db_result.success:
    ...     db = db_result.value

    Domain modeling:
    >>> class User(FlextEntity):
    ...     name: str
    ...     email: str
    ...
    ...     def activate(self) -> FlextResult[None]:
    ...         if self.is_active:
    ...             return FlextResult.fail("Already active")
    ...         self.is_active = True
    ...         return FlextResult.ok(None)

Notes:
    - All business operations should return FlextResult[T] for composability
    - Use the global container for dependency injection across the ecosystem
    - Domain entities should inherit from FlextEntity or FlextValueObject
    - Follow Clean Architecture patterns with layered imports
    - Leverage type safety with generic parameters and protocols

"""

from __future__ import annotations

# =============================================================================
# FOUNDATION LAYER - Import first, no dependencies on other modules
# =============================================================================

from flext_core.__version__ import *
from flext_core.constants import *
from flext_core.typings import *
from flext_core.result import *
from flext_core.exceptions import *
from flext_core.protocols import *

# =============================================================================
# DOMAIN LAYER - Depends only on Foundation layer
# =============================================================================

from flext_core.aggregate_root import *
from flext_core.domain_services import *
from flext_core.models import *

# =============================================================================
# APPLICATION LAYER - Depends on Domain + Foundation layers
# =============================================================================

from flext_core.commands import *
from flext_core.guards import *
from flext_core.handlers import *
from flext_core.validation import *

# =============================================================================
# INFRASTRUCTURE LAYER - Depends on Application + Domain + Foundation
# =============================================================================

# Infrastructure layer - explicit imports to avoid type conflicts
from flext_core.config import *  # type: ignore[assignment]
from flext_core.container import *  # type: ignore[assignment]
from flext_core.context import *
from flext_core.loggings import *
from flext_core.observability import *

# =============================================================================
# SUPPORT LAYER - Depends on layers as needed, imported last
# =============================================================================

from flext_core.decorators import *
from flext_core.delegation_system import *
from flext_core.fields import *
from flext_core.mixins import *
from flext_core.processors import *
from flext_core.services import *
from flext_core.type_adapters import *
from flext_core.utilities import *

# =============================================================================
# CORE FUNCTIONALITY - Main implementation exports
# =============================================================================

# Core functionality - ensure specific exports are accessible
# from flext_core.core import *


# =============================================================================
# LEGACY FUNCTIONALITY - Legacy implementation exports
# =============================================================================

# Legacy functionality - ensure specific exports are accessible
from flext_core.legacy import *  # type: ignore[assignment]


# =============================================================================
# CONSOLIDATED EXPORTS - Combine all __all__ from modules
# =============================================================================

# Combine all __all__ exports from imported modules
import flext_core.__version__ as _version
import flext_core.aggregate_root as _aggregate_root
import flext_core.commands as _commands
import flext_core.config as _config
import flext_core.constants as _constants
import flext_core.container as _container
import flext_core.context as _context

# import flext_core.core as _core
import flext_core.decorators as _decorators
import flext_core.delegation_system as _delegation_system
import flext_core.domain_services as _domain_services
import flext_core.exceptions as _exceptions
import flext_core.fields as _fields
import flext_core.guards as _guards
import flext_core.handlers as _handlers
import flext_core.loggings as _loggings
import flext_core.mixins as _mixins
import flext_core.models as _models
import flext_core.observability as _observability
import flext_core.protocols as _protocols
import flext_core.result as _result
import flext_core.processors as _processors
import flext_core.services as _services
import flext_core.type_adapters as _type_adapters
import flext_core.typings as _typings
import flext_core.utilities as _utilities
import flext_core.validation as _validation
import flext_core.legacy as _legacy

# Collect all __all__ exports from imported modules
_temp_exports: list[str] = []

for module in [
    _version,
    _constants,
    _typings,
    _result,
    _exceptions,
    _protocols,
    _models,
    _aggregate_root,
    _domain_services,
    _commands,
    _validation,
    _guards,
    _handlers,
    _container,
    _config,
    _loggings,
    _observability,
    _context,
    # _core,
    _mixins,
    _decorators,
    _utilities,
    _fields,
    _services,
    _delegation_system,
    _processors,
    _type_adapters,
    _legacy,
]:
    if hasattr(module, "__all__"):
        _temp_exports.extend(module.__all__)

# Remove duplicates and sort for consistent exports - build complete list first
_seen: set[str] = set()
_final_exports: list[str] = []
for item in _temp_exports:
    if item not in _seen:
        _seen.add(item)
        _final_exports.append(item)
_final_exports.sort()

# Define __all__ as literal list for linter compatibility
# This dynamic assignment is necessary for aggregating module exports
__all__: list[str] = _final_exports  # pyright: ignore[reportUnsupportedDunderAll] # noqa: PLE0605
