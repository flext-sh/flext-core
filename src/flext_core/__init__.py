"""FLEXT Core - Foundation framework for enterprise domain-driven applications.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

==============================================================================
ROOT MODULE - ECOSYSTEM INTEGRATION POINT
==============================================================================

This module provides the canonical public API for flext-core, a comprehensive
framework for building production-grade applications following domain-driven
design, clean architecture, and railway-oriented programming principles.

**Architecture Layer**: 0.5 (Root Module - Integration Bridge)

FLEXT Core provides the foundation for 32+ dependent projects through:

**Railway-Oriented Error Handling**:
  FlextResult[T] - Monadic error handling with composable operations
  - Structural typing: Satisfies FlextProtocols.Result via method signatures
  - Methods: ok(), fail(), is_success, is_failure, unwrap(), unwrap_error()
  - Operations: map, flat_map, map_error, alt, filter, pipeline
  - Primary access: .value (raises on failure), .unwrap(), .unwrap_or(default)

**Dependency Injection & Configuration**:
  FlextContainer - Type-safe singleton dependency injection
  - Structural typing: Satisfies FlextProtocols.ServiceLocator via methods
  - Methods: with_service(), get(), with_factory(), get_typed()
  - Features: Singleton pattern, factory pattern, type-safe retrieval
  - Integration: Global container accessible from any layer

**CQRS & Event-Driven Architecture**:
  FlextDispatcher - Unified command/query dispatch orchestration (Layer 1-3)
  - Layer 1: CQRS routing with command/query handler registration
  - Layer 2: Reliability patterns (circuit breaker, rate limiting, retry, timeout)
  - Layer 3: Advanced processing (batch, parallel, fallback execution)
  - Structural typing: Satisfies FlextProtocols.CommandDispatcher
  - Methods: execute(), dispatch(), register_handler(), process_batch(),
    process_parallel()

  FlextHandlers - CQRS handler registry (Layer 3)
  - Structural typing: Satisfies FlextProtocols.HandlerRegistry
  - Methods: register_handler(), execute(), get_handler()

**Domain Modeling**:
  FlextModels - DDD patterns for entities and value objects (Layer 2)
  - Structural typing: Satisfies FlextProtocols.Model via methods
  - Entity - Has identity (id), typically mutable
  - Value - Immutable, compared by value
  - AggregateRoot - Consistency boundary with invariant enforcement
  - Features: Pydantic v2 integration, validation, serialization

**Structured Logging**:
  FlextLogger - Context-aware structured logging (Layer 4)
  - Structural typing: Satisfies FlextProtocols.Logger via methods
  - Methods: info(), debug(), warning(), error(), critical()
  - Features: Context propagation, correlation IDs, performance tracking
  - Integration: Automatic structlog context management

**Configuration Management**:
  FlextConfig - Pydantic v2 settings with validation (Layer 4)
  - Structural typing: Satisfies FlextProtocols.Configurable
  - Features: Environment variable substitution, validation, type safety
  - Integration: Settings with hierarchical overrides

**Context Management**:
  FlextContext - Hierarchical request/operation context (Layer 4)
  - Structural typing: Satisfies FlextProtocols.ContextManager
  - Features: Contextvars, correlation IDs, service identification
  - Namespaces: Variables, Correlation, Service, Request, Performance

**Structured Exception Handling**:
  FlextExceptions - Exception hierarchy with error codes (Layer 1)
  - Structural typing: Satisfies FlextProtocols.Exception via methods
  - Exception types: 14+ specialized error categories
  - Features: Error codes, correlation IDs, structured logging
  - Integration: Automatic metrics and monitoring

**Protocol Definitions**:
  FlextProtocols - @runtime_checkable protocol interfaces (Layer 0)
  - 12+ domain-specific protocols using structural typing
  - Protocols: Service, Repository, UseCase, CommandHandler, QueryHandler
  - Features: Type-safe interfaces, duck typing verification
  - Integration: All classes satisfy protocols through method signatures

**Type System & Utilities**:
  FlextTypes - 40+ TypeVars and type aliases (Layer 0)
  - Variance: Covariance (_co), Contravariance (_contra), Invariant
  - Aliases: Command, Query, Event, Message, ResultT
  - Features: Generic type constraints for type safety

  FlextUtilities - ID generation and data normalization helpers (Layer 2)
  - Structural typing: Satisfies FlextProtocols.Utility
  - Methods: validate_pipeline(), Generators (14+ ID/timestamp methods)
  - Features: Validator composition, deterministic ID generation

**Service Base Class**:
  FlextService - Domain service base with context enrichment (Layer 2)
  - Structural typing: Satisfies FlextProtocols.Service
  - Features: Logger injection, context management, operation tracking
  - Methods: execute(), validate_business_rules(), get_service_info()

**Decorators for Cross-Cutting Concerns**:
  FlextDecorators - Automation decorators (Layer 3)
  - @inject - Dependency injection
  - @log_operation - Operation logging
  - @track_performance - Performance tracking
  - @railway - Railway-oriented error handling
  - @retry, @timeout, @with_correlation - Reliability patterns
  - Structural typing: Satisfies FlextProtocols.Decorator

**Constants & Runtime Integration**:
  FlextConstants - 50+ error codes and patterns (Layer 0)
  - 30+ nested namespaces for organization
  - Integration: Used throughout for error classification

  FlextRuntime - External library integration (Layer 0.5)
  - Bridge between FLEXT protocols and third-party libraries
  - Features: Automatic adaptation, compatibility layer

==============================================================================
STRUCTURAL TYPING (DUCK TYPING) - CORE DESIGN PRINCIPLE
==============================================================================

All FLEXT classes implement FlextProtocols through **structural typing**, not
inheritance. This means:

1. **Method Signatures**: Classes satisfy protocols by implementing required
   methods with correct signatures
2. **No Inheritance**: Classes do NOT inherit from @runtime_checkable protocols
3. **isinstance() Works**: isinstance(obj, FlextProtocols.Service) validates
   structural compliance
4. **Duck Typing**: "If it walks like a duck and quacks like a duck, it's a duck"
5. **Type Safety**: Full type checking with MyPy strict mode
6. **Flexibility**: Classes can satisfy multiple protocols without conflict

Example of structural typing:
  class UserService:
      '''Satisfies FlextProtocols.Service through method signatures.'''
      def execute(self, command: Command) -> FlextResult:
          '''Required method - protocol compliance verified.'''
          pass

  # No inheritance from FlextProtocols.Service!
  service = UserService()
  assert isinstance(service, FlextProtocols.Service)  # ✅ True (duck typing)

==============================================================================
ECOSYSTEM ARCHITECTURE
==============================================================================

Layer Hierarchy (Clean Architecture):

  Layer 4: Infrastructure
    ├─ FlextConfig (Pydantic Settings)
    ├─ FlextLogger (Structured Logging)
    └─ FlextContext (Request Context)

  Layer 3: Application
    ├─ FlextDispatcher (Unified CQRS Dispatch - Layers 1-3)
    ├─ FlextHandlers (Handler Registry)
    └─ FlextDecorators (Cross-Cutting Concerns)

  Layer 2: Domain
    ├─ FlextModels (DDD Models)
    ├─ FlextService (Domain Services)
    ├─ FlextMixins (Reusable Behaviors)
    └─ FlextUtilities (Validation & Conversion)

  Layer 1: Foundation
    ├─ FlextResult[T] (Railway Pattern)
    ├─ FlextContainer (Dependency Injection)
    └─ FlextExceptions (Error Hierarchy)

  Layer 0.5: Integration Bridge
    └─ FlextRuntime (Third-Party Integration)

  Layer 0: Pure Constants
    ├─ FlextConstants (50+ Error Codes)
    ├─ FlextTypes (40+ TypeVars)
    ├─ FlextProtocols (12+ Protocol Definitions)
    └─ FlextRegistry (Handler Registry)

==============================================================================
QUALITY GATES & VALIDATION
==============================================================================

Production Readiness (All Mandatory):
  ✅ Type Safety: Pyrefly strict mode (zero type errors)
  ✅ Linting: Ruff with zero violations
  ✅ Testing: Comprehensive test coverage (80%+)
  ✅ Security: Bandit security scanning
  ✅ Documentation: Complete API documentation
  ✅ Type-Safe APIs: Full generic type preservation and strict typing

This module re-exports all public APIs from 12+ sub-modules while maintaining
strict layer hierarchy and protocol compliance.

==============================================================================
USAGE EXAMPLES
==============================================================================

**Example 1: Railway-Oriented Error Handling**:
    >>> from flext_core import FlextResult
    >>> from pydantic import EmailStr
    >>> def validate_user_email(email: EmailStr) -> FlextResult[str]:
    ...     # Pydantic v2 EmailStr validates format natively
    ...     return FlextResult[str].ok(email)
    >>> result = validate_user_email("user@example.com")
    >>> if result.is_success:
    ...     email = result.unwrap()

**Example 2: Dependency Injection (Fluent Interface)**:
    >>> from flext_core import FlextContainer, FlextLogger
    >>> container = FlextContainer().with_service("logger", FlextLogger(__name__))
    >>> logger_result = container.get("logger")
    >>> if logger_result.is_success:
    ...     retrieved_logger = logger_result.unwrap()

**Example 3: Domain-Driven Design**:
    >>> from flext_core import FlextModels
    >>> class Email(FlextModels.Value):
    ...     address: str
    >>> class User(FlextModels.Entity):
    ...     name: str
    ...     email: Email
    >>> user = User(id="1", name="John", email=Email(address="john@example.com"))

**Example 4: CQRS Pattern with Dispatcher**:
    >>> from flext_core import FlextDispatcher, FlextResult
    >>> dispatcher = FlextDispatcher()
    >>> class CreateUserCommand:
    ...     name: str
    >>> def handle_create_user(cmd: CreateUserCommand) -> FlextResult[dict]:
    ...     return FlextResult[dict].ok({"user_id": "123"})
    >>> dispatcher.register_handler(CreateUserCommand, handle_create_user)
    >>> result = dispatcher.dispatch(CreateUserCommand(name="Alice"))

**Example 5: Service Base Class**:
    >>> from flext_core import FlextService, FlextResult
    >>> class UserService(FlextService[dict]):
    ...     def execute(self, cmd: dict) -> FlextResult[dict]:
    ...         self.logger.info("Processing user command")
    ...         return FlextResult[dict].ok({"status": "ok"})

**Example 6: Structured Logging with Context**:
    >>> from flext_core import FlextLogger, FlextContext
    >>> logger = FlextLogger(__name__)
    >>> with FlextContext():
    ...     logger.info("user_created", user_id="123", email="user@example.com")

**Example 7: Configuration Management**:
    >>> from flext_core import FlextConfig
    >>> class AppConfig(FlextConfig):
    ...     database_url: str
    ...     debug: bool = False
    >>> config = AppConfig(database_url="postgres://localhost")

**Example 8: Decorator for Dependency Injection**:
    >>> from flext_core import FlextDecorators, FlextContainer, FlextResult
    >>> @FlextDecorators.inject
    ... def process_user(container: FlextContainer) -> FlextResult:
    ...     return FlextResult[bool].ok(True)

**Example 9: Decorator for Performance Tracking**:
    >>> from flext_core import FlextDecorators
    >>> @FlextDecorators.track_performance
    ... def expensive_operation() -> None:
    ...     pass
    >>> expensive_operation()  # Automatically tracks execution time

**Example 10: Full Integration Example**:
    >>> from flext_core import (
    ...     FlextResult,
    ...     FlextContainer,
    ...     FlextLogger,
    ...     FlextService,
    ...     FlextModels,
    ...     FlextExceptions,
    ... )
    >>> class CreateUserService(FlextService[dict]):
    ...     def execute(self, cmd: dict) -> FlextResult[dict]:
    ...         container = FlextContainer.get_global()
    ...         db_result = container.get("database")
    ...         if db_result.is_failure:
    ...             error = db_result.unwrap_error()
    ...             self.logger.error("db_unavailable", error=str(error))
    ...             return FlextResult[dict].fail("Database unavailable")
    ...         return FlextResult[dict].ok({"user_id": "123"})

==============================================================================
ROOT IMPORT PATTERN (ECOSYSTEM STANDARD)
==============================================================================

✅ CORRECT - Always use root imports (this module):
    from flext_core import FlextResult, FlextContainer, FlextModels

❌ FORBIDDEN - Never use internal module imports:
    from flext_core.result import FlextResult       # ❌ Breaks ecosystem
    from flext_core.container import FlextContainer # ❌ Breaks ecosystem
    from flext_core.models import FlextModels       # ❌ Breaks ecosystem

Why: 32+ dependent projects rely on root imports. Internal imports break the
entire ecosystem by creating circular dependencies and import order issues.

==============================================================================
TYPE SAFETY & GENERIC CONSTRAINTS
==============================================================================

Complete TypeVar system with proper variance:

**Covariance** (_co suffix):
  T_co - Generic output type (can be subtype in covariant contexts)
  TResult_co - Result output type
  TEntity_co - Entity domain type
  T1_co, T2_co, T3_co - Multi-parameter covariant types

**Contravariance** (_contra suffix):
  T_contra - Generic input type (can be supertype in contravariant contexts)
  TCommand_contra - Command input handler type
  TEvent_contra - Event subscriber type
  TCacheKey_contra - Cache key type

**Invariant** (no suffix):
  T - Generic invariant type (must match exactly)
  K - Key type
  V - Value type
  P, R, E, F, U, W - Placeholder types for specific contexts

These ensure type safety throughout the ecosystem while preventing variance
errors during composition and inheritance.

For detailed documentation, see the README.md file in this directory.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from beartype import BeartypeConf, BeartypeStrategy

from flext_core.__version__ import __version__, __version_info__
from flext_core._models.collections import FlextModelsCollections
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger, FlextLoggerResultAdapter
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import (
    CallableInputT,
    CallableOutputT,
    E,
    F,
    FactoryT,
    FlextTypes,
    K,
    MessageT_contra,
    P,
    R,
    ResultT,
    T,
    T1_co,
    T2_co,
    T3_co,
    T_co,
    T_Config,
    T_contra,
    T_Namespace,
    TAggregate_co,
    TCacheKey_contra,
    TCacheValue_co,
    TCommand_contra,
    TConfigKey_contra,
    TDomainEvent_co,
    TEntity_co,
    TEvent_contra,
    TInput_contra,
    TInput_Handler_contra,
    TInput_Handler_Protocol_contra,
    TItem_contra,
    TQuery_contra,
    TResult_co,
    TResult_contra,
    TResult_Handler_co,
    TResult_Handler_Protocol,
    TState_co,
    TUtil_contra,
    TValue_co,
    TValueObject_co,
    U,
    V,
    W,
)
from flext_core.utilities import FlextUtilities

# Type aliases for nested FlextConfig attributes (pyrefly compatibility)
AutoConfig = FlextConfig.AutoConfig
auto_register = FlextConfig.auto_register

# =============================================================================
# RUNTIME TYPE CHECKING - Python 3.13 Strict Typing Enforcement
# =============================================================================
# beartype provides RUNTIME type validation in addition to static checking.
#
# ENABLED via FlextRuntime.enable_runtime_checking() for package-wide validation.
# Critical methods can also use @beartype decorator individually.
#
# Beartype provides O(log n) runtime validation with minimal overhead.
# Static type checking (pyright strict mode) is ALWAYS active.
# Documentation: https://beartype.readthedocs.io/en/stable/
# =============================================================================

# Beartype configuration for runtime type checking (available for your use)
BEARTYPE_CONF = BeartypeConf(
    strategy=BeartypeStrategy.Ologn,  # O(log n) - thorough with acceptable overhead
    is_color=True,  # Colored error messages
    claw_is_pep526=False,  # Disable variable annotation checking
    warning_cls_on_decorator_exception=UserWarning,  # Warnings on decorator failures
)

# =============================================================================

__all__ = [
    "BEARTYPE_CONF",
    "AutoConfig",
    "CallableInputT",
    "CallableOutputT",
    "E",
    "F",
    "FactoryT",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLogger",
    "FlextLoggerResultAdapter",
    "FlextMixins",
    "FlextModels",
    "FlextModelsCollections",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextTypes",
    "FlextUtilities",
    "K",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "T",
    "T1_co",
    "T2_co",
    "T3_co",
    "TAggregate_co",
    "TCacheKey_contra",
    "TCacheValue_co",
    "TCommand_contra",
    "TConfigKey_contra",
    "TDomainEvent_co",
    "TEntity_co",
    "TEvent_contra",
    "TInput_Handler_Protocol_contra",
    "TInput_Handler_contra",
    "TInput_contra",
    "TItem_contra",
    "TQuery_contra",
    "TResult_Handler_Protocol",
    "TResult_Handler_co",
    "TResult_co",
    "TResult_contra",
    "TState_co",
    "TUtil_contra",
    "TValueObject_co",
    "TValue_co",
    "T_Config",
    "T_Namespace",
    "T_co",
    "T_contra",
    "U",
    "V",
    "W",
    "__version__",
    "__version_info__",
    "auto_register",
]
