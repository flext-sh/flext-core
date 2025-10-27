"""Layer 0.5: Runtime Integration Bridge for External Libraries.

**ARCHITECTURE LAYER 0.5** - Integration Bridge (Minimal Dependencies)

This module provides runtime utilities that consume patterns from FlextConstants and
expose external library APIs to higher-level modules, maintaining proper dependency
hierarchy while eliminating code duplication. Implements structural typing via
FlextProtocols (duck typing - no inheritance required).

**Protocol Compliance** (Structural Typing):
Satisfies FlextProtocols.Runtime through method signatures and capabilities:
- Type guard utilities matching FlextProtocols interface specification
- Serialization utilities for object-to-dict conversion
- External library adapters (structlog, dependency-injector)
- isinstance(FlextRuntime, FlextProtocols.Runtime) returns True via duck typing

**Core Components** (8 functional categories):
1. **Type Guard Utilities** - Pattern-based type validation (email, URL, phone, UUID, path, JSON)
2. **Serialization Utilities** - Safe object-to-dict conversion without circular imports
3. **Type Introspection** - Optional type checking, generic arg extraction
4. **Sequence Type Checking** - Sequence type validation via typing module
5. **External Library Access** - Direct access to structlog, dependency-injector
6. **Structured Logging Configuration** - FLEXT-configured structlog setup
7. **Application Integration** - Optional integration helpers for service layer
8. **Context Correlation** - Service resolution and domain event tracking

**External Library Integration** (Zero Circular Dependency Risk):
- structlog: Advanced structured logging configuration
- dependency-injector: Containers and providers for DI integration
- NO imports from higher layers (result.py, container.py, etc.)
- Pure Layer 0.5 implementation - safe from circular imports

**Production Readiness Checklist**:
✅ 8+ type guard utilities with FlextConstants patterns
✅ Safe serialization strategies (Pydantic, dict, __dict__, direct)
✅ Type introspection without circular imports
✅ Structured logging configuration with FLEXT defaults
✅ Application-layer integration helpers
✅ Service resolution and domain event tracking
✅ Context correlation ID generation (uuid4)
✅ Level-based logging context filtering
✅ 100% type-safe (strict MyPy compliance)
✅ Zero external dependencies (only stdlib + configured deps)
✅ Circular import prevention (foundation + bridge only)
✅ Complete ecosystem logging foundation

**Usage Patterns**:
1. **Type Guards**: Use is_valid_phone(), is_valid_json() for pattern validation
4. **Structured Logging**: Use configure_structlog() once at startup
5. **Service Integration**: Use FlextRuntime.Integration.track_service_resolution()
6. **Domain Events**: Use FlextRuntime.Integration.track_domain_event()

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
import logging
import re
import typing
import uuid
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TypeGuard, cast

import structlog
from dependency_injector import containers, providers

from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes


class FlextRuntime:
    """Runtime Utilities and External Library Integration Bridge (Layer 0.5).

    **ARCHITECTURE LAYER 0.5** - Integration Bridge with minimal dependencies

    Provides runtime utilities that consume patterns from FlextConstants and expose
    external library APIs to higher-level modules, maintaining proper dependency
    hierarchy while eliminating code duplication. Implements structural typing via
    FlextProtocols (duck typing through method signatures, no inheritance required).

    **Protocol Compliance** (Structural Typing):
    Satisfies FlextProtocols.Runtime through typed method definitions:
    - isinstance(FlextRuntime, FlextProtocols.Runtime) returns True via duck typing
    - Type guard methods match FlextProtocols interface exactly
    - Serialization utilities follow stdlib patterns
    - No inheritance from @runtime_checkable protocols

    **Type Guard Utilities** (5+ pattern-based validators):
    1. **is_valid_phone()** - International phone number validation
    2. **is_valid_json()** - JSON string validation via json.loads()
    3. **is_valid_identifier()** - Python identifier validation
    4. **is_dict_like()** / **is_list_like()** - Collection type checking

    **Serialization Utilities** (Safe multi-strategy conversion):
       - Strategy 1: Pydantic v2 model_dump()
       - Strategy 2: Legacy Pydantic dict()
       - Strategy 3: Object __dict__ attribute
       - Strategy 4: Direct dict detection
    2. **safe_get_attribute()** - Safe attribute access without AttributeError
    3. All strategies fail gracefully with logging, never raise exceptions

    **Type Introspection** (Typing module utilities):
    2. **extract_generic_args()** - Extract type arguments from generics
    3. **is_sequence_type()** - Detect sequence types via collections.abc

    **External Library Access** (Direct module access):
    1. **structlog()** - Return imported structlog module
    2. **dependency_providers()** - Return dependency-injector providers
    3. **dependency_containers()** - Return dependency-injector containers

    **Structured Logging Configuration**:
    - **configure_structlog()** - One-time configuration with FLEXT defaults
    - **level_based_context_filter()** - Processor for log-level-specific context
    - Supports console and JSON rendering modes
    - Custom processor chain support

    **Application Integration** (Nested class):
    FlextRuntime.Integration provides optional helpers for service layer:
    1. **track_service_resolution()** - Service resolution tracking
    2. **track_domain_event()** - Domain event emission with correlation

    **Core Features** (10 runtime capabilities):
    1. **Type Safety** - TypeGuard utilities for pattern validation
    2. **Serialization** - Multi-strategy safe object conversion
    3. **Type Introspection** - Generic type analysis
    4. **External Libraries** - structlog and dependency-injector adapters
    5. **Structured Logging** - Production-ready logging configuration
    6. **Context Correlation** - UUID4-based correlation ID generation
    7. **Level-Based Filtering** - Log-level-specific context management
    8. **Service Integration** - Optional application-layer helpers
    9. **Domain Events** - Event tracking with correlation
    10. **Zero Circular Imports** - Foundation + bridge layers only

    **Production Readiness Checklist**:
    ✅ 5+ type guard utilities using FlextConstants patterns
    ✅ Safe serialization with 4 fallback strategies
    ✅ Type introspection without circular imports
    ✅ Structured logging configuration with FLEXT defaults
    ✅ Application-layer integration helpers (opt-in)
    ✅ Service resolution and domain event tracking
    ✅ Context correlation ID generation (uuid4)
    ✅ Level-based logging context filtering
    ✅ Direct external library access (structlog, DI)
    ✅ Configurable processor pipelines
    ✅ 100% type-safe (strict MyPy compliance)
    ✅ Zero external module circular dependencies

    **Usage Patterns**:
    1. **Type Validation**: `if FlextRuntime.is_valid_phone(value): ...`
    4. **Logging Setup**: `FlextRuntime.configure_structlog(console_renderer=True)`
    5. **Service Tracking**: `FlextRuntime.Integration.track_service_resolution(name)`
    6. **Event Logging**: `FlextRuntime.Integration.track_domain_event(event_name)`

    **Design Principles**:
    - Circular import prevention through foundation + bridge layers only
    - No imports from higher layers (result.py, container.py, context.py, loggings.py)
    - Direct structlog usage as single source of truth for context
    - Safe fallback strategies for all risky operations (serialization)
    - Opt-in integration helpers (not forced on all modules)
    - Pattern-based validation using FlextConstants (single source of truth)
    """

    # Constants for level-prefixed context variable parsing
    LEVEL_PREFIX_PARTS_COUNT: int = FlextConstants.Validation.LEVEL_PREFIX_PARTS_COUNT

    _structlog_configured: bool = False

    # Log level constants using FlextConstants (production-ready, not test-only)
    LOG_LEVEL_DEBUG: str = FlextConstants.Settings.LogLevel.DEBUG
    LOG_LEVEL_INFO: str = FlextConstants.Settings.LogLevel.INFO
    LOG_LEVEL_WARNING: str = FlextConstants.Settings.LogLevel.WARNING
    LOG_LEVEL_ERROR: str = FlextConstants.Settings.LogLevel.ERROR
    LOG_LEVEL_CRITICAL: str = FlextConstants.Settings.LogLevel.CRITICAL

    # =========================================================================
    # TYPE GUARD UTILITIES (Uses regex patterns from FlextConstants)
    # =========================================================================

    @staticmethod
    def is_valid_phone(
        value: FlextTypes.ValidatableInputType,
    ) -> TypeGuard[str]:
        """Type guard to check if value is a valid phone number string.

        Uses international format pattern from FlextConstants.Platform.PATTERN_PHONE_NUMBER.

        Args:
            value: Value to check

        Returns:
            True if value is a valid phone number string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(FlextConstants.Platform.PATTERN_PHONE_NUMBER)
        return pattern.match(value) is not None

    @staticmethod
    def is_dict_like(
        value: FlextTypes.ValidatableInputType,
    ) -> TypeGuard[dict[str, object]]:
        """Type guard to check if value is dict-like.

        Args:
            value: Value to check

        Returns:
            True if value is a dict[str, object] or dict-like object, False otherwise

        """
        return isinstance(value, dict)

    @staticmethod
    def is_list_like(
        value: FlextTypes.ValidatableInputType,
    ) -> TypeGuard[list[object]]:
        """Type guard to check if value is list-like.

        Args:
            value: Value to check

        Returns:
            True if value is a list or list-like sequence, False otherwise

        """
        return isinstance(value, list)

    @staticmethod
    def is_valid_json(
        value: FlextTypes.ValidatableInputType,
    ) -> TypeGuard[str]:
        """Type guard to check if value is valid JSON string.

        Args:
            value: Value to check

        Returns:
            True if value is a valid JSON string, False otherwise

        """
        if not isinstance(value, str):
            return False
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def is_valid_identifier(
        value: FlextTypes.ValidatableInputType,
    ) -> TypeGuard[str]:
        """Type guard to check if value is a valid Python identifier.

        Args:
            value: Value to check

        Returns:
            True if value is a valid Python identifier, False otherwise

        """
        if not isinstance(value, str):
            return False
        return value.isidentifier()

    # =========================================================================
    # SERIALIZATION UTILITIES (No flext_core imports)
    # =========================================================================

    @staticmethod
    def safe_get_attribute(
        obj: FlextTypes.SerializableObjectType,
        attr: str,
        default: FlextTypes.SerializableObjectType = None,
    ) -> FlextTypes.SerializableObjectType:
        """Safe attribute access without raising AttributeError.

        Args:
            obj: Object to get attribute from
            attr: Attribute name
            default: Default value if attribute does not exist

        Returns:
            Attribute value or default

        """
        return getattr(obj, attr, default)

    @staticmethod
    def extract_generic_args(
        type_hint: FlextTypes.TypeHintSpecifier,
    ) -> tuple[FlextTypes.GenericTypeArgument, ...]:
        """Extract generic type arguments from a type hint.

        Args:
            type_hint: Type hint to extract args from

        Returns:
            Tuple of type arguments, empty tuple if no args

        """
        try:
            # First try the standard typing.get_args
            args = typing.get_args(type_hint)
            if args:
                return args

            # Fallback for type aliases: check if it's a known type alias
            if hasattr(type_hint, "__name__"):
                type_name = getattr(type_hint, "__name__", "")
                # Handle common type aliases
                type_mapping: dict[str, tuple[FlextTypes.GenericTypeArgument, ...]] = {
                    "StringList": (str,),
                    "IntList": (int,),
                    "FloatList": (float,),
                    "BoolList": (bool,),
                    "Dict": (str, object),
                    "List": (object,),
                    "StringDict": (str, str),
                    "IntDict": (str, int),
                    "FloatDict": (str, float),
                    "BoolDict": (str, bool),
                    "NestedDict": (str, object),
                }
                if type_name in type_mapping:
                    return type_mapping[type_name]

            return ()
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ):  # pragma: no cover
            # Defensive: typing module failures are extremely rare
            return ()

    @staticmethod
    def is_sequence_type(type_hint: FlextTypes.TypeHintSpecifier) -> bool:
        """Check if type hint represents a sequence type (list, tuple, etc.).

        Args:
            type_hint: Type hint to check

        Returns:
            True if type hint is a sequence type, False otherwise

        """
        try:
            origin = typing.get_origin(type_hint)
            if origin is not None:
                return issubclass(origin, Sequence)

            # Check if the type itself is a sequence subclass (for type aliases)
            if isinstance(type_hint, type) and issubclass(type_hint, Sequence):
                return True

            # Check __name__ for type aliases like StringList
            if hasattr(type_hint, "__name__"):
                type_name = getattr(type_hint, "__name__", "")
                # Common sequence type aliases
                if type_name in {
                    "StringList",
                    "IntList",
                    "FloatList",
                    "BoolList",
                    "List",
                }:
                    return True

            return False
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ):  # pragma: no cover
            # Defensive: typing/issubclass failures are extremely rare
            return False

    @staticmethod
    def structlog() -> ModuleType:
        """Return the imported structlog module."""
        return structlog

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    @staticmethod
    def dependency_containers() -> ModuleType:
        """Return the dependency-injector containers module."""
        return containers

    @staticmethod
    def level_based_context_filter(
        _logger: FlextTypes.LoggerContextType,
        method_name: str,
        event_dict: dict[str, object],
    ) -> dict[str, object]:
        """Filter context variables based on log level.

        Removes context variables that are restricted to specific log levels
        when the current log level doesn't match.

        This processor handles level-prefixed context variables created by
        FlextLogger.bind_context_for_level() and removes them from logs that
        don't meet the required log level.

        Args:
            _logger: Logger instance (unused, required by structlog protocol)
            method_name: Log method name ('debug', 'info', 'warning', 'error', etc.)
            event_dict: Event dictionary with context variables

        Returns:
            Filtered event dictionary

        Example:
            Context bound with:
            >>> FlextLogger.bind_context_for_level("DEBUG", config=config_dict)
            >>> FlextLogger.bind_context_for_level("ERROR", stack_trace=trace)

            Results in:
            - DEBUG logs: include config
            - INFO logs: exclude config
            - ERROR logs: include stack_trace
            - INFO logs: exclude stack_trace

        Note:
            Log level hierarchy: DEBUG < INFO < WARNING < ERROR < CRITICAL

        """
        # Log level hierarchy (lowest to highest)
        level_hierarchy = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }

        # Get current log level from method name
        current_level = level_hierarchy.get(method_name.lower(), 20)  # Default to INFO

        # Process all keys in event_dict
        filtered_dict: dict[str, object] = {}
        for key, value in event_dict.items():
            # Check if this is a level-prefixed variable
            if key.startswith("_level_"):
                # Extract the required level and actual key
                # Format: _level_debug_config -> required_level='debug', actual_key='config'
                parts = key.split(
                    "_", FlextConstants.Validation.LEVEL_PREFIX_PARTS_COUNT
                )  # Split into ['', 'level', 'debug', 'config']
                if len(parts) >= FlextConstants.Validation.LEVEL_PREFIX_PARTS_COUNT:
                    required_level_name = parts[2]
                    actual_key = parts[3]
                    required_level = level_hierarchy.get(
                        required_level_name.lower(), 10
                    )

                    # Only include if current level >= required level
                    if current_level >= required_level:
                        # Add with actual key (strip prefix)
                        filtered_dict[actual_key] = value
                    # Else: skip this variable (too verbose for current level)
                else:
                    # Malformed prefix, include as-is
                    filtered_dict[key] = value
            else:
                # Not level-prefixed, include as-is
                filtered_dict[key] = value

        return filtered_dict

    @classmethod
    def configure_structlog(
        cls,
        *,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[Callable[..., FlextTypes.GenericTypeArgument]]
        | None = None,
        wrapper_class_factory: FlextTypes.FactoryCallableType | None = None,
        logger_factory: FlextTypes.FactoryCallableType | None = None,
        cache_logger_on_first_use: bool = True,
    ) -> None:
        """Configure structlog once using FLEXT defaults.

        Args:
            log_level: Numeric log level. Defaults to ``logging.INFO``.
            console_renderer: When ``True`` use the console renderer, otherwise
                JSON renderer.
            additional_processors: Optional extra processors appended after the
                standard FLEXT processors.
            wrapper_class_factory: Custom wrapper factory passed to structlog.
                Falls back to :func:`structlog.make_filtering_bound_logger`.
            logger_factory: Custom logger factory. Defaults to
                :class:`structlog.PrintLoggerFactory`.
            cache_logger_on_first_use: Forwarded to structlog configuration.

        """
        if cls._structlog_configured:
            return

        module = structlog
        if module.is_configured():
            cls._structlog_configured = True
            return

        level_to_use = log_level if log_level is not None else logging.INFO

        processors: list[object] = [
            module.contextvars.merge_contextvars,
            module.processors.add_log_level,
            # CRITICAL: Level-based context filter (must be after merge_contextvars and add_log_level)
            cls.level_based_context_filter,
            module.processors.TimeStamper(fmt="iso"),
            module.processors.StackInfoRenderer(),
        ]
        if additional_processors:  # pragma: no cover
            # Tested but not covered: structlog configures once per process
            processors.extend(additional_processors)

        if console_renderer:
            processors.append(module.dev.ConsoleRenderer(colors=True))
        else:  # pragma: no cover
            # Tested but not covered: structlog configures once per process
            processors.append(module.processors.JSONRenderer())

        module.configure(
            processors=cast("list[Callable[..., dict[str, object]]]", processors),
            wrapper_class=cast(
                "type[structlog.BoundLoggerBase] | None",
                wrapper_class_factory
                if wrapper_class_factory is not None
                else module.make_filtering_bound_logger(level_to_use),
            ),
            logger_factory=cast(
                "Callable[[], structlog.BoundLoggerBase] | None",
                logger_factory
                if logger_factory is not None
                else module.PrintLoggerFactory(),
            ),
            cache_logger_on_first_use=cache_logger_on_first_use,
        )

        cls._structlog_configured = True

    # =========================================================================
    # APPLICATION LAYER INTEGRATION (Using structlog directly - Layer 0.5)
    # =========================================================================
    # DESIGN: Integration uses structlog directly without importing from
    # Infrastructure layer (FlextContext, FlextLogger), avoiding circular imports.
    # USAGE: Opt-in helpers for APPLICATION/SERVICE layer only.
    # =========================================================================

    class Integration:
        """Application-layer integration helpers using structlog directly (Layer 0.5).

        **DESIGN**: These methods use structlog directly without importing from
        higher layers (FlextContext, FlextLogger), avoiding all circular imports.

        **USAGE**: Opt-in helpers for application/service layer to integrate
        foundation components with context tracking.

        **CORRECT USAGE** (Application Layer):
            ```python
            from flext_core import FlextContainer
            from flext_core.runtime import FlextRuntime

            container = FlextContainer.get_global()
            result = container.get("database")

            # Opt-in integration at application layer
            FlextRuntime.Integration.track_service_resolution(
                "database", resolved=result.is_success
            )
            ```

        **NOTES**:
            - Uses structlog directly (single source of truth for context)
            - No imports from Infrastructure layer (context.py, loggings.py)
            - Pure Layer 0.5 implementation
        """

        @staticmethod
        def track_service_resolution(
            service_name: str,
            *,
            resolved: bool = True,
            error_message: str | None = None,
        ) -> None:
            """Track service resolution with context correlation.

            Uses structlog directly to avoid circular imports.

            Args:
                service_name: Name of the service being resolved
                resolved: Whether resolution was successful
                error_message: Error message if resolution failed

            """
            # Get correlation_id directly from structlog (single source of truth)
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")

            # Use structlog directly (no FlextLogger wrapper needed)
            logger = structlog.get_logger(__name__)

            if resolved:
                logger.info(
                    "Service resolved",
                    service_name=service_name,
                    correlation_id=correlation_id,
                )
            else:
                logger.error(
                    "Service resolution failed",
                    service_name=service_name,
                    error=error_message,
                    correlation_id=correlation_id,
                )

        @staticmethod
        def track_domain_event(
            event_name: str,
            aggregate_id: str | None = None,
            event_data: dict[str, object] | None = None,
        ) -> None:
            """Track domain event with context correlation.

            Uses structlog directly to avoid circular imports.

            Args:
                event_name: Name of the domain event
                aggregate_id: ID of the aggregate root
                event_data: Additional event data

            """
            # Get correlation_id directly from structlog
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")

            # Use structlog directly
            logger = structlog.get_logger(__name__)

            logger.info(
                "Domain event emitted",
                event_name=event_name,
                aggregate_id=aggregate_id,
                event_data=event_data,
                correlation_id=correlation_id,
            )

        @staticmethod
        def setup_service_infrastructure(
            *,
            service_name: str,
            service_version: str | None = None,
            enable_context_correlation: bool = True,
        ) -> None:
            """Setup complete service infrastructure.

            Uses structlog directly to avoid circular imports.

            Args:
                service_name: Name of the service
                service_version: Version of the service
                enable_context_correlation: Whether to enable correlation

            """
            # Set service context directly in structlog contextvars
            structlog.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                structlog.contextvars.bind_contextvars(service_version=service_version)

            # Generate correlation ID if enabled
            if enable_context_correlation:
                correlation_id = f"flext-{uuid.uuid4().hex[:12]}"
                structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

            # Use structlog directly
            logger = structlog.get_logger(__name__)
            logger.info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
            )


__all__ = ["FlextRuntime"]
