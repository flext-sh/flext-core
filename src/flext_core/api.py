"""FlextCore - Unified facade for complete flext-core ecosystem.

This module provides the main thin facade exposing ALL flext-core functionality
through a single unified entry point. Following FLEXT domain library standards,
this is the recommended way to access flext-core components with proper
integration of all foundation patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from flext_core.__version__ import __version__, __version_info__
from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processors import FlextProcessors
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextCore:
    """Unified facade for complete flext-core ecosystem integration.

    This is the single recommended entry point for accessing ALL flext-core
    functionality with proper component integration and modern patterns.

    **UNIFIED FACADE PATTERN**: One class providing access to entire ecosystem.
    **COMPLETE INTEGRATION**: All 20+ flext-core components accessible.

    **Components Available**:

    - **FlextResult**: Railway pattern for error handling
    - **FlextConfig**: Pydantic 2.11+ configuration management
    - **FlextContainer**: Dependency injection container
    - **FlextModels**: DDD base classes (Entity, Value, AggregateRoot)
    - **FlextLogger**: Structured logging with context
    - **FlextBus**: Event messaging and pub/sub
    - **FlextContext**: Request/operation context management
    - **FlextHandlers**: CQRS handler registration
    - **FlextRegistry**: Component registration
    - **FlextDispatcher**: Message routing
    - **FlextMixins**: Reusable behaviors
    - **FlextUtilities**: Helper functions
    - **FlextService**: Service base class
    - **FlextConstants**: System constants
    - **FlextTypes**: Type system (40+ TypeVars)
    - **FlextExceptions**: Exception hierarchy
    - **FlextProtocols**: Interface definitions

    **Single-Import Usage Pattern (Recommended for Examples)**:

    ```python
    # ✅ SINGLE IMPORT - Complete framework access
    from flext_core import FlextCore

    # Railway pattern - direct class access
    result = FlextCore.Result[str].ok("success")
    error = FlextCore.Result[str].fail("error")

    # Components via namespace (zero additional imports)
    logger = FlextCore.Logger(__name__)
    container = FlextCore.Container.get_global()
    entity = FlextCore.Models.Entity(id="123")

    # Constants and types
    timeout = FlextCore.Constants.Defaults.TIMEOUT
    http_ok = FlextCore.Constants.Http.Status.OK


    # Decorators - TWO ACCESS PATTERNS
    @FlextCore.railway  # Direct access (NEW in v0.9.9+)
    def operation() -> FlextCore.Result[str]:
        return FlextCore.Result[str].ok("done")


    @FlextCore.Decorators.railway  # Namespace access (also supported)
    def operation2() -> FlextCore.Result[str]:
        return FlextCore.Result[str].ok("done")


    # Runtime management
    runtime = FlextCore.Runtime()

    # Validation utilities
    is_valid = FlextCore.validate.is_not_empty("test")

    # Version information
    print(f"Version: {FlextCore.version}")  # e.g., "0.9.9"
    ```

    **Instance-Based Usage** (for stateful operations):

    ```python
    # Create unified core instance
    core = FlextCore()

    # Access components through properties
    config = core.config
    container = core.container
    logger = core.logger
    ```

    **Architecture**:

    - **Foundation Library**: Core patterns for entire FLEXT ecosystem
    - **Thin Facade**: NO business logic, pure delegation
    - **Complete Access**: All 20+ components available
    - **Modern Patterns**: Railway, DI, CQRS, DDD
    - **Type Safety**: Full generic type support

    **Design Principles**:

    - Thin facade: NO logic, pure component access
    - Complete integration: ALL flext-core components
    - Railway pattern: FlextResult throughout
    - Zero duplication: Direct access to components
    - Ecosystem foundation: Sets standards for all projects
    """

    # =================================================================
    # CLASS-LEVEL TYPE ALIASES
    # =================================================================
    # Direct class references for maintaining method access patterns
    # This allows both FlextCore.Result[T].ok() and isinstance() checks to work

    # Direct class reference (not a type alias) to maintain method access
    # FlextCore.Result.ok() and FlextCore.Result[T].ok() both work
    Result = FlextResult

    class Types(FlextTypes):
        """Type system namespace providing type definitions."""

    class Models(FlextModels):
        """Domain modeling namespace providing DDD patterns."""

    class Constants(FlextConstants):
        """System constants and configuration values."""

    class Exceptions(FlextExceptions):
        """Exception hierarchy for error handling."""

    class Protocols(FlextProtocols):
        """Protocol definitions and interfaces."""

    class Config(FlextConfig):
        """Configuration management with Pydantic."""

    class Container(FlextContainer):
        """Dependency injection container with facade-compatible type returns.

        Overrides parent classmethods to return FlextCore.Container type
        instead of FlextContainer for proper type compatibility in examples.
        """

        @classmethod
        def get_global(cls) -> FlextCore.Container:
            """Get the global container instance.

            Returns:
                FlextCore.Container: The global container instance with proper facade typing.

            Note:
                This override ensures FlextCore.Container.get_global() returns
                FlextCore.Container type instead of base FlextContainer type.

            """
            # Call parent implementation but cast to correct facade type
            # The parent _ensure_global_instance() creates FlextContainer instance,
            # but since FlextCore.Container inherits from FlextContainer,
            # the instance is compatible - we just need to satisfy the type checker
            return cast("FlextCore.Container", super().get_global())

    class Logger(FlextLogger):
        """Structured logging with context."""

    class Bus(FlextBus):
        """Event messaging and pub/sub."""

    class Context(FlextContext):
        """Request and operation context management."""

    class Registry(FlextRegistry):
        """Component registration and discovery."""

    class Dispatcher(FlextDispatcher):
        """Message routing and dispatching."""

    class Handlers[MessageT_contra, ResultT](FlextHandlers[MessageT_contra, ResultT]):
        """CQRS handler registration and execution with facade-compatible type returns.

        Overrides parent classmethods to return FlextCore.Handlers type
        instead of FlextHandlers for proper type compatibility in examples.
        """

        @classmethod
        def from_callable(
            cls,
            callable_func: Callable[..., object],
            handler_name: str | None = None,
            handler_type: Literal["command", "query"] = "command",
            mode: str | None = None,
            handler_config: FlextModels.CqrsConfig.Handler
            | FlextTypes.Dict
            | None = None,
        ) -> FlextCore.Handlers[object, object]:
            """Create a handler from a callable function.

            Args:
                callable_func: The callable function to wrap
                handler_name: Name for the handler (defaults to function name)
                handler_type: Type of handler (command, query, etc.)
                mode: Handler mode (for compatibility)
                handler_config: Optional handler configuration

            Returns:
                FlextCore.Handlers: Handler instance with proper facade typing.

            Note:
                This override ensures FlextCore.Handlers.from_callable() returns
                FlextCore.Handlers type instead of base FlextHandlers type.

            """
            # Call parent implementation but cast to correct facade type
            return cast(
                "FlextCore.Handlers[object, object]",
                super().from_callable(
                    callable_func=callable_func,
                    handler_name=handler_name,
                    handler_type=handler_type,
                    mode=mode,
                    handler_config=handler_config,
                ),
            )

    class Processors(FlextProcessors):
        """Processing utilities and orchestration."""

    class Service[TDomainResult](FlextService[TDomainResult]):
        """Base service class with infrastructure."""

    class Mixins(FlextMixins):
        """Reusable behavior mixins."""

    class Utilities(FlextUtilities):
        """Helper functions and utilities."""

    class Runtime(FlextRuntime):
        """Runtime management and lifecycle."""

    class Decorators(FlextDecorators):
        """Decorator utilities namespace.

        Provides access to all flext-core decorators through unified facade.
        Inherits from FlextDecorators for single-class pattern consistency.

        Example:
            >>> @FlextCore.Decorators.railway()
            ... def operation() -> FlextResult[str]:
            ...     return FlextResult[str].ok("success")
            >>>
            >>> @FlextCore.Decorators.inject(logger=FlextLogger)
            ... def process(*, logger: FlextLogger) -> None:
            ...     logger.info("Processing...")

        """

    # =================================================================
    # VERSION INFORMATION (v0.9.9+ Enhancement)
    # =================================================================
    # Direct access to version information through FlextCore facade

    version: str = __version__
    version_info: tuple[int | str, ...] = __version_info__

    # =================================================================
    # INSTANCE INITIALIZATION
    # =================================================================
    # Note: PEP 695 type aliases (above) automatically support generic subscripting
    # like FlextCore.Result[T] without needing __class_getitem__ method

    def __init__(self) -> None:
        """Initialize the unified core facade.

        Args:
            None

        """
        super().__init__()


__all__ = [
    "FlextCore",
]
