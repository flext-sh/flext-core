"""FLEXT Core - Foundation Library for FLEXT Ecosystem.

This module provides the complete FLEXT ecosystem foundation following
FLEXT architectural patterns with proper layered imports and zero circular
dependencies. All exports use wildcard imports from individual modules
following the hierarchical FlextXxx pattern.

Architecture Overview:
    This library follows Clean Architecture principles with proper dependency
    layering to avoid circular imports. The import order is carefully structured
    to respect dependencies:

    Foundation Layer (no dependencies):
        - __version__: Version information
        - constants: System-wide constants and enums
        - typings: Type definitions and generic patterns
        - result: Railway-oriented programming with FlextResult[T]
        - exceptions: Exception hierarchy and error handling
        - protocols: Interface definitions and contracts

    Domain Layer (depends on Foundation):
        - models: Domain models with Pydantic integration
        - entities: Entity patterns for DDD
        - aggregate_root: Aggregate root patterns with domain events
        - domain_services: Domain service patterns

    Application Layer (depends on Domain + Foundation):
        - commands: CQRS command patterns
        - handlers: Handler implementations and registry
        - validation: Validation framework with predicates
        - payload: Message/event patterns for integration
        - guards: Type guards and validation decorators

    Infrastructure Layer (depends on Application + Domain + Foundation):
        - container: Dependency injection container
        - config: Configuration management with Pydantic Settings
        - loggings: Structured logging with structlog integration
        - observability: Metrics, tracing, monitoring abstractions
        - context: Request/operation context management
        - core: Core framework integration layer

    Support Layer (depends on layers as needed, imported last):
        - mixins: Reusable behavior patterns (timestamps, etc.)
        - decorators: Enterprise decorator patterns
        - utilities: Helper functions and generators
        - fields: Field validation and metadata
        - services: Service layer abstractions
        - semantic: Semantic modeling and analysis
        - test helpers: Available separately when needed
        - delegation_system: Mixin delegation patterns
        - schema_processing: Schema validation and processing
        - type_adapters: Type adaptation utilities
        - root_models: Root model patterns for validation
        - legacy: Backward compatibility layer

Key Features:
    - FlextResult[T] for railway-oriented programming
    - Hierarchical FlextTypes system for comprehensive type definitions
    - FlextConstants for system-wide constants and configuration
    - FlextProtocols for centralized interface definitions
    - FlextContainer for dependency injection
    - FlextEntity, FlextValueObject, FlextAggregateRoot for DDD patterns
    - Professional enterprise patterns following SOLID principles

Examples:
    Basic usage with FlextResult pattern::

        from flext_core import FlextResult


        def validate_email(email: str) -> FlextResult[str]:
            if "@" not in email:
                return FlextResult[None].fail("Invalid email format")
            return FlextResult[None].ok(email)


        result = validate_email("user@example.com")
        if result.success:
            validated_email = result.unwrap()

    Using hierarchical types::

        from flext_core import FlextTypes

        # Type-safe declarations
        user_id: FlextTypes.Domain.EntityId = "user_123"
        config: FlextTypes.Config.ConfigDict = {"key": "value"}

    Dependency injection::

        from flext_core import get_flext_container

        container = get_flext_container()
        container.register("service", MyService())
        service_result = container.get("service")

Note:
    This module enforces Python 3.13+ requirements and follows FLEXT refactoring
    patterns with hierarchical organization, proper import layering to avoid
    circular dependencies, and centralized compatibility management through
    legacy.py facades.

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
from flext_core.payload import *
from flext_core.validation import *

# =============================================================================
# INFRASTRUCTURE LAYER - Depends on Application + Domain + Foundation
# =============================================================================

from flext_core.config import *
from flext_core.container import *
from flext_core.context import *
from flext_core.core import *
from flext_core.loggings import *
from flext_core.observability import *

# =============================================================================
# SUPPORT LAYER - Depends on layers as needed, imported last
# =============================================================================

from flext_core.decorators import *
from flext_core.delegation_system import *
from flext_core.fields import *
from flext_core.mixins import *
from flext_core.root_models import *
from flext_core.schema_processing import *
from flext_core.services import *
from flext_core.type_adapters import *
from flext_core.utilities import *

# =============================================================================
# LEGACY COMPATIBILITY LAYER - Import last for backward compatibility
# =============================================================================

from flext_core.legacy import *

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
import flext_core.core as _core
import flext_core.decorators as _decorators
import flext_core.delegation_system as _delegation_system
import flext_core.domain_services as _domain_services
import flext_core.exceptions as _exceptions
import flext_core.fields as _fields
import flext_core.guards as _guards
import flext_core.handlers as _handlers
import flext_core.legacy as _legacy
import flext_core.loggings as _loggings
import flext_core.mixins as _mixins
import flext_core.models as _models
import flext_core.observability as _observability
import flext_core.payload as _payload
import flext_core.protocols as _protocols
import flext_core.result as _result
import flext_core.root_models as _root_models
import flext_core.schema_processing as _schema_processing
import flext_core.services as _services
import flext_core.type_adapters as _type_adapters
import flext_core.typings as _typings
import flext_core.utilities as _utilities
import flext_core.validation as _validation

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
    _payload,
    _handlers,
    _container,
    _config,
    _loggings,
    _observability,
    _context,
    _core,
    _mixins,
    _decorators,
    _utilities,
    _fields,
    _services,
    # _testing_utilities removed - not available
    _delegation_system,
    _schema_processing,
    _type_adapters,
    _root_models,
    _legacy,  # Include legacy module in wildcard exports
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
