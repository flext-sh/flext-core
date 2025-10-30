# ruff: disable=E402
"""Domain service base class with dependency injection and validation.

This module provides FlextService[T], a base class for implementing domain
services with comprehensive infrastructure support including dependency
injection, context management, logging, and validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import inspect
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Union, cast, get_args, get_origin, override

from pydantic import computed_field

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult

# =========================================================================
# SERVICE FACTORY DECORATOR - Phase 3 Enhancement
# =========================================================================


def service_factory(factory: Callable[..., Any]) -> Callable[[type], type]:
    """Decorator to register custom service factory.

    Allows services to provide custom factory logic instead of relying on
    automatic dependency detection. Useful for services with complex
    initialization requirements.

    **Usage:**

        >>> @service_factory(lambda container: MyService(custom_init=True))
        ... class MyService(FlextService[Result]):
        ...     def execute(self) -> FlextResult[Result]:
        ...         return FlextResult[Result].ok(Result())

    Args:
        factory: Callable that takes container and returns service instance

    Returns:
        Decorator function for service class

    """

    def decorator(service_class: type) -> type:
        """Mark service class with custom factory."""
        # Store custom factory on the class for __init_subclass__ to use
        # SLF001: Intentional private attribute assignment for decorator pattern
        # Cast to Any to allow dynamic attribute assignment on type objects
        cast("Any", service_class)._custom_factory = factory
        return service_class

    return decorator


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins,
    ABC,
):
    """Base class for domain services with dependency injection and validation.

    Implements FlextProtocols.Service through structural typing (duck typing).
    All service subclasses automatically satisfy the Service protocol by
    implementing required methods: execute(), validate_business_rules(),
    validate_config(), is_valid(), and get_service_info().

    **INTERFACE SEGREGATION - Component Responsibilities:**

    This class achieves interface segregation by inheriting from three focused bases:

    1. **FlextModels.ArbitraryTypesModel** (Data Layer)
       - Pydantic v2 validation
       - Type-safe field declarations
       - Serialization support (model_dump, model_dump_json)

    2. **FlextMixins** (Infrastructure Layer)
       Provides transparent access to framework infrastructure:

       Properties (mixin-provided):
       - container: FlextContainer - Global DI singleton
       - context: FlextContext - Request/operation context
       - logger: FlextLogger - Structured logging
       - config: FlextConfig - Global configuration
       - track() → Iterator[dict] - Operation performance tracking

       Private Methods (mixin-provided):
       - _register_in_container() - Service auto-registration
       - _propagate_context() - Context inheritance
       - _get_correlation_id() - Distributed tracing
       - _with_operation_context() - Scoped context management
       - _clear_operation_context() - Cleanup automation

    3. **ABC** (Abstract Protocol)
       - Abstract execute() method enforcement
       - Structural typing via FlextProtocols.Service

    **PROTOCOL IMPLEMENTATION - Methods This Class Provides:**
    ✅ STRUCTURAL TYPING: Implements FlextProtocols.Service interface
    - execute() [abstract] - Domain operation (implement in subclass)
    - validate_business_rules() - Business logic validation
    - validate_config() - Configuration validation
    - is_valid() - Combined validity check
    - get_service_info() - Service metadata
    - project_config - Auto-resolve project configuration
    - project_models - Auto-resolve domain models namespace

    **AUTO-REGISTRATION & DEPENDENCY INJECTION:**
    Services are automatically registered in the DI container via __init_subclass__.
    Constructor parameters are inspected for dependency injection.

    Usage:
        >>> from flext_core.service import FlextService
        >>> from flext_core.result import FlextResult
        >>> from flext_core.protocols import FlextProtocols
        >>>
        >>> class UserService(FlextService[User]):
        ...     # Container and logger automatically available via FlextMixins
        ...     def execute(self) -> FlextResult[User]:
        ...         # Use infrastructure transparently
        ...         self.logger.info("Creating user")
        ...         return FlextResult[User].ok(User(name="John"))
        >>>
        >>> service = UserService()  # Auto-registered in container
        >>> # Satisfies FlextProtocols.Service structurally
        >>> assert isinstance(service, FlextProtocols.Service)
    """

    # Mixin-provided infrastructure properties (explicit type documentation)
    # These are declared in FlextMixins and available on all service instances
    # container: FlextContainer - Singleton DI container access
    # context: FlextContext - Scoped request/operation context
    # logger: FlextLogger - Structured logging with context
    # config: FlextConfig - Global configuration instance
    # track: Method[Iterator] - Context manager for operation tracking

    _bus: object | None = None  # FlextBus type to avoid circular import

    @computed_field  # Pydantic v2 computed_field already provides property behavior
    def service_config(self) -> FlextConfig:
        """Automatic config binding via Pydantic v2 computed_field.

        Pure Pydantic v2 pattern - no wrappers, no descriptors, no boilerplate.
        Config is transparently available via computed property.

        Example:
            >>> class OrderService(FlextService[Order]):
            ...     def execute(self) -> FlextResult[Order]:
            ...         # Config automatically available
            ...         timeout = self.service_config.timeout
            ...         return FlextResult[Order].ok(Order())

        Returns:
            FlextConfig: Global configuration instance

        """
        return FlextConfig.get_global_instance()

    @property
    def project_config(self) -> FlextConfig:
        """Auto-resolve project-specific configuration by naming convention.

        Attempts to resolve configuration using naming convention:
        - Service class name: FlextCliCore → FlextCliConfig
        - Looks up in global container
        - Falls back to FlextConfig.get_global_instance()

        This property enables dependency-free configuration access:
        - No manual PrivateAttr declarations needed
        - Convention-based auto-resolution
        - Type-safe configuration access

        Example:
            >>> class FlextCliCore(FlextService[CliDataDict]):
            ...     def execute(self) -> FlextResult[CliDataDict]:
            ...         # Automatically resolves FlextCliConfig
            ...         debug = self.project_config.debug
            ...         return FlextResult[CliDataDict].ok({})

        Returns:
            FlextConfig: Project-specific configuration instance

        """
        try:
            # Extract project name from service class: FlextCliCore → FlextCli
            service_class_name = self.__class__.__name__
            # Try to find matching config class
            # Pattern: FlextXyzService → FlextXyzConfig
            config_class_name = service_class_name.replace("Service", "Config")

            container = self.container
            config_result = container.get(config_class_name)

            if config_result.is_success:
                return config_result.unwrap()
        except Exception:
            # Fall back to global config if resolution fails
            pass

        # Fall back to global config
        return FlextConfig.get_global_instance()

    @property
    def project_models(self) -> type:
        """Auto-resolve project-specific models namespace by naming convention.

        Attempts to resolve models using naming convention:
        - Service class name: FlextCliCore → FlextCliModels
        - Looks up in global container
        - Returns empty namespace if not found

        This property enables model-free service implementation:
        - No manual models imports needed
        - Convention-based auto-resolution
        - Type namespace access via property

        Example:
            >>> class FlextCliCore(FlextService[CliDataDict]):
            ...     def execute(self) -> FlextResult[CliDataDict]:
            ...         # Automatically resolves FlextCliModels
            ...         config_model = self.project_models.Configuration
            ...         return FlextResult[CliDataDict].ok({})

        Returns:
            type: Project models namespace (typically a class with nested types)

        """
        try:
            # Extract project name from service class: FlextCliCore → FlextCli
            service_class_name = self.__class__.__name__
            # Try to find matching models class
            # Pattern: FlextXyzService → FlextXyzModels
            models_class_name = service_class_name.replace("Service", "Models")

            container = self.container
            models_result = container.get(models_class_name)

            if models_result.is_success:
                models_obj = models_result.unwrap()
                if isinstance(models_obj, type):
                    return models_obj
        except Exception:
            # Return default namespace if resolution fails
            pass

        # Return a minimal namespace type if not found
        return type("ModelsNamespace", (), {})

    def __init_subclass__(cls) -> None:
        """Automatic service registration with enhanced dependency detection.

        Pure Python 3.13+ pattern - no wrappers, no helpers, no boilerplate.
        Services are transparently registered in global container when class is defined.

        **Phase 3 Enhancements**:
        - Custom factory support via @service_factory decorator
        - Improved type detection: handles Optional[T], Union[T1, T2], generics
        - Better error messages for missing dependencies
        - Complex dependency graph support (transitive dependencies)

        **Phase 2 Features**:
        - Scans service class __init__ method for constructor parameters
        - Detects required dependencies (parameters excluding 'self' and 'config')
        - Creates smart factory that auto-injects dependencies from container
        - Falls back to simple instantiation if dependencies can't be resolved

        Example:
            >>> class UserService(FlextService[User]):  # ← Auto-registered
            ...     def __init__(self, database: Database, cache: Cache):
            ...         # Dependencies auto-injected from container!
            ...         super().__init__()
            ...         self.database = database
            ...         self.cache = cache
            ...
            ...     def execute(self) -> FlextResult[User]:
            ...         return FlextResult[User].ok(User(name="John"))

            >>> # Service already registered - no manual calls needed
            >>> container = FlextContainer.get_global()
            >>> service_result = container.get("UserService")
            >>> assert service_result.is_success  # ← Database and Cache injected!

        **Custom Factory Example**:
            >>> @service_factory(lambda container: UserService(db=special_db))
            ... class UserService(FlextService[User]):
            ...     def __init__(self, db: Database):
            ...         super().__init__()
            ...         self.db = db

        **Backward Compatibility**: Services without special dependencies work unchanged

        """
        super().__init_subclass__()

        service_name = cls.__name__
        container = FlextContainer.get_global()

        # Check for custom factory first (Phase 3 enhancement)
        custom_factory = getattr(cls, "_custom_factory", None)
        if custom_factory is not None:
            # Use custom factory directly
            container.register_factory(service_name, lambda: custom_factory(container))
            return

        # Enhanced dependency detection with support for Optional, Union types
        try:
            init_signature = inspect.signature(cls.__init__)
            dependencies: dict[str, object] = {}
            optional_dependencies: set[str] = set()

            # Extract parameters and detect types
            for param_name, param in init_signature.parameters.items():
                if param_name == "self":
                    continue

                # Skip config parameter (handled separately)
                if param_name == "config":
                    continue

                # Skip **kwargs and *args parameters
                if param.kind in {
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                }:
                    continue

                # Skip generic 'data' parameter (Pydantic initialization)
                if param_name == "data":
                    continue

                # Store dependency name and type
                if param.annotation != inspect.Parameter.empty:
                    dep_type = param.annotation

                    # Skip generic object type (too broad)
                    if dep_type is object:
                        continue

                    dependencies[param_name] = dep_type

                    # Detect optional types (Optional[T] = Union[T, None])
                    origin = get_origin(dep_type)
                    if origin is Union:
                        # Check if None is in the union (making it optional)
                        args = get_args(dep_type)
                        if type(None) in args:
                            optional_dependencies.add(param_name)

            # Create smart factory with enhanced dependency resolution
            if dependencies:

                def smart_factory(
                    deps: dict[str, object] = dependencies,
                    optional: set[str] = optional_dependencies,
                ) -> object:
                    """Factory with automatic dependency injection (Phase 3)."""
                    resolved_deps: dict[str, object] = {}
                    unresolved_required: list[tuple[str, str]] = []

                    # Attempt to resolve each dependency from container
                    for dep_name, dep_type in deps.items():
                        # Try to get from container by parameter name first
                        dep_result = container.get(dep_name)

                        if dep_result.is_success:
                            resolved_deps[dep_name] = dep_result.unwrap()
                            continue

                        # Try to get by type name (extract from complex types)
                        base_type = dep_type
                        origin = get_origin(dep_type)

                        # Extract base type from Optional, Union, etc.
                        if origin is Union:
                            args = get_args(dep_type)
                            # Use first non-None type
                            base_type = next(
                                (a for a in args if a is not type(None)), dep_type
                            )

                        type_name = getattr(base_type, "__name__", str(base_type))
                        type_result = container.get(type_name)

                        if type_result.is_success:
                            resolved_deps[dep_name] = type_result.unwrap()
                        elif dep_name not in optional:
                            # Track unresolved required dependencies
                            unresolved_required.append((dep_name, str(base_type)))

                    # Better error messages for missing required dependencies
                    if unresolved_required:
                        missing = ", ".join(
                            f"{name}({type_})" for name, type_ in unresolved_required
                        )
                        msg = (
                            f"Cannot create {service_name}: "
                            f"unresolved required dependencies: {missing}. "
                            f"Register them in container or use @service_factory."
                        )
                        raise RuntimeError(msg)

                    # Create instance with resolved dependencies
                    return cls(**resolved_deps)

                # Register with smart factory
                container.register_factory(service_name, smart_factory)
            else:
                # No detected dependencies - use simple factory
                container.register_factory(service_name, cls)

        except (ValueError, TypeError) as e:
            # If signature inspection fails, try simple registration
            try:
                container.register_factory(service_name, cls)
            except Exception as inner_e:
                # Last resort: log and continue
                msg = (
                    f"Failed to register {service_name}: "
                    f"signature inspection failed ({e!s})"
                )
                raise RuntimeError(msg) from inner_e

    @override
    def __init__(self, **data: object) -> None:
        """Initialize domain service with Pydantic validation and infrastructure.

        Automatic infrastructure provided transparently:
        - Service registration: via __init_subclass__ (class definition time)
        - Container access: via FlextMixins.container property
        - Logger access: via FlextMixins.logger property
        - Context access: via FlextMixins.context property
        - Config access: via FlextMixins.config property

        No manual setup needed - pure Python 3.13+ patterns.
        """
        super().__init__(**data)
        # AUTOMATIC: All infrastructure via properties (zero boilerplate)

    # =============================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses (Domain.Service protocol)
    # =============================================================================

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain operation (Domain.Service protocol).

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure with error

        """

    def execute_with_context_cleanup(self) -> FlextResult[TDomainResult]:
        """Execute operation with automatic scoped context cleanup.

        This method wraps execute() with automatic cleanup of operation-scoped
        logging context, preventing context accumulation while preserving
        request and application-level context.

        Returns:
            FlextResult[TDomainResult]: Result from execute() with guaranteed context cleanup

        Usage:
            >>> service = MyService()
            >>> result = service.execute_with_context_cleanup()
            >>> # Operation context cleared, request context (correlation_id) preserved

        Note:
            - Recommended for calling services from CLI/API boundaries
            - Clears operation scope only (preserves request and application scopes)
            - Request-level context (correlation_id) persists across service calls
            - Application-level context (app name, version) persists for lifetime

        """
        try:
            # Execute the service operation
            return self.execute()
        finally:
            # CRITICAL: Clean up operation-scoped context to prevent accumulation
            # Preserves request context (correlation_id) and application context
            self._clear_operation_context()

    # =============================================================================
    # VALIDATION METHODS (Domain.Service protocol)
    # =============================================================================

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for the domain service (Domain.Service protocol).

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        return FlextResult[None].ok(None)

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration (Domain.Service protocol).

        Returns:
            FlextResult[None]: Success if configuration is valid, failure with error details

        """
        return FlextResult[None].ok(None)

    def is_valid(self) -> bool:
        """Check if the domain service is in a valid state (Domain.Service protocol).

        Returns:
            bool: True if the service is valid and ready for operations, False otherwise

        """
        # Check business rules and configuration
        try:
            business_rules = self.validate_business_rules()
            config = self.validate_config()
            return business_rules.is_success and config.is_success
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            # If validation raises an exception, the service is not valid
            return False

    def get_service_info(self) -> dict[str, object]:
        """Get service information and metadata (Domain.Service protocol).

        Returns:
            dict[str, object]: Service information dictionary with basic service type info.

        """
        return {
            "service_type": self.__class__.__name__,
        }

    # =============================================================================
    # OPERATION EXECUTION METHODS (Domain.Service protocol)
    # =============================================================================

    def execute_operation(
        self,
        request: FlextModels.OperationExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute operation with validation, timeout, retry, and monitoring (Domain.Service protocol).

        Validates business rules and configuration before executing the operation.

        Args:
            request: Operation execution request with callable, arguments, and configuration

        Returns:
            FlextResult[TDomainResult]: Success with operation result or failure with validation/execution error

        """
        operation_name = request.operation_name or "unnamed_operation"
        with self.track(operation_name):
            self._propagate_context(operation_name)

            self.logger.info(
                f"Executing operation: {request.operation_name}",
                extra={
                    "timeout_seconds": request.timeout_seconds,
                    "has_retry_config": bool(request.retry_config),
                    "correlation_id": self._get_correlation_id(),
                },
            )

            # Validate business rules before execution (Domain.Service protocol)
            business_rules_result = self.validate_business_rules()
            if business_rules_result.is_failure:
                self.logger.error(
                    f"Business rules validation failed for operation: {request.operation_name}",
                    extra={"error": business_rules_result.error},
                )
                return FlextResult[TDomainResult].fail(
                    f"Business rules validation failed: {business_rules_result.error}"
                )

            # Validate configuration (Domain.Service protocol)
            config_result = self.validate_config()
            if config_result.is_failure:
                self.logger.error(
                    f"Configuration validation failed for operation: {request.operation_name}",
                    extra={"error": config_result.error},
                )
                return FlextResult[TDomainResult].fail(
                    f"Configuration validation failed: {config_result.error}"
                )

            # Validate keyword_arguments is a dict
            if not isinstance(request.keyword_arguments, dict):
                return FlextResult[TDomainResult].fail(
                    f"Invalid keyword arguments: expected dict, got {type(request.keyword_arguments).__name__}"
                )

            # Execute with retry logic if configured
            retry_config = request.retry_config or {}

            # Validate retry config types
            max_attempts_raw = retry_config.get("max_attempts", 1) or 1
            if not isinstance(max_attempts_raw, int):
                return FlextResult[TDomainResult].fail(
                    f"Invalid retry configuration: max_attempts must be an integer, got {type(max_attempts_raw).__name__}"
                )

            initial_delay_raw = retry_config.get("initial_delay_seconds", 0.1) or 0.1
            if not isinstance(initial_delay_raw, (int, float)):
                return FlextResult[TDomainResult].fail(
                    f"Invalid retry configuration: initial_delay_seconds must be numeric, got {type(initial_delay_raw).__name__}"
                )

            max_delay_raw = retry_config.get("max_delay_seconds", 60.0) or 60.0
            if not isinstance(max_delay_raw, (int, float)):
                return FlextResult[TDomainResult].fail(
                    f"Invalid retry configuration: max_delay_seconds must be numeric, got {type(max_delay_raw).__name__}"
                )

            # Validate backoff_multiplier if present
            backoff_multiplier_raw = retry_config.get("backoff_multiplier")
            if backoff_multiplier_raw is not None:
                if not isinstance(backoff_multiplier_raw, (int, float)):
                    return FlextResult[TDomainResult].fail(
                        f"Invalid retry configuration: backoff_multiplier must be numeric, got {type(backoff_multiplier_raw).__name__}"
                    )
                if backoff_multiplier_raw < 1.0:
                    return FlextResult[TDomainResult].fail(
                        "Invalid retry configuration: backoff_multiplier must be >= 1.0"
                    )

            max_attempts: int = max_attempts_raw
            initial_delay: float = cast("float", initial_delay_raw)
            max_delay: float = cast("float", max_delay_raw)
            exponential_backoff: bool = cast(
                "bool", retry_config.get("exponential_backoff", False)
            )
            retry_on_exceptions_raw = retry_config.get(
                "retry_on_exceptions", [Exception]
            )
            retry_on_exceptions: list[type[Exception]] = cast(
                "list[type[Exception]]", retry_on_exceptions_raw or [Exception]
            )

            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    # Filter out None values from arguments
                    filtered_args = [
                        v for v in request.arguments.values() if v is not None
                    ]

                    # Apply timeout if specified
                    if request.timeout_seconds and request.timeout_seconds > 0:
                        if not callable(request.operation_callable):
                            return FlextResult[TDomainResult].fail(
                                f"operation_callable must be callable, got {type(request.operation_callable)}"
                            )
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as executor:
                            future = executor.submit(
                                request.operation_callable,
                                *filtered_args,
                                **request.keyword_arguments,
                            )
                            # Let TimeoutError and other exceptions propagate to outer except
                            # so retry logic can handle them
                            result = future.result(timeout=request.timeout_seconds)
                    else:
                        # Execute the operation without timeout
                        if not callable(request.operation_callable):
                            return FlextResult[TDomainResult].fail(
                                f"operation_callable must be callable, got {type(request.operation_callable)}"
                            )
                        result = request.operation_callable(
                            *filtered_args, **request.keyword_arguments
                        )

                    self.logger.info(
                        f"Operation completed successfully: {request.operation_name}"
                    )

                    # If result is already a FlextResult, return it directly
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[TDomainResult]", result)

                    result_value: TDomainResult = cast("TDomainResult", result)
                    return FlextResult[TDomainResult].ok(result_value)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = any(
                        isinstance(e, exc_type) for exc_type in retry_on_exceptions
                    )

                    if not should_retry or attempt >= max_attempts - 1:
                        # Create better error message based on exception type
                        error_msg = str(e) or ""
                        if isinstance(
                            e,
                            (TimeoutError, concurrent.futures.TimeoutError),
                        ):
                            error_msg = (
                                "Operation timed out"
                                f" after {request.timeout_seconds} seconds"
                            )
                        elif not error_msg:
                            error_msg = type(e).__name__

                        self.logger.exception(
                            f"Operation execution failed: {request.operation_name}",
                            extra={
                                "error": error_msg,
                                "error_type": type(e).__name__,
                            },
                        )
                        return FlextResult[TDomainResult].fail(
                            f"Operation {request.operation_name} failed: {error_msg}"
                        )

                    # Calculate delay for retry
                    if exponential_backoff:
                        delay = min(initial_delay * (2**attempt), max_delay)
                    else:
                        delay = min(initial_delay, max_delay)

                    self.logger.warning(
                        f"Operation {request.operation_name} failed (attempt {attempt + 1}/{max_attempts}), retrying in {delay}s",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

                    time.sleep(delay)

            # Should not reach here, but handle it
            if last_exception:
                return FlextResult[TDomainResult].fail(
                    f"Operation {request.operation_name} failed: {last_exception}"
                )
            return FlextResult[TDomainResult].fail(
                f"Operation {request.operation_name} failed after {max_attempts} attempts"
            )

    def execute_with_full_validation(
        self, _request: FlextModels.DomainServiceExecutionRequest
    ) -> FlextResult[TDomainResult]:
        """Execute operation with full validation including business rules and config.

        Args:
            _request: Domain service execution request (unused, for protocol compatibility)

        Returns:
            FlextResult[TDomainResult]: Success with operation result or failure with validation/execution error

        """
        # Full validation: business rules + config + execution
        business_rules_result = self.validate_business_rules()
        if business_rules_result.is_failure:
            return FlextResult[TDomainResult].fail(
                f"Business rules validation failed: {business_rules_result.error}"
            )

        config_result = self.validate_config()
        if config_result.is_failure:
            return FlextResult[TDomainResult].fail(
                f"Configuration validation failed: {config_result.error}"
            )

        # Execute the operation and cast to object result type for API compatibility
        return self.execute()

    def execute_conditionally(
        self, request: FlextModels.ConditionalExecutionRequest
    ) -> FlextResult[TDomainResult]:
        """Execute operation conditionally based on the provided condition.

        Args:
            request: Conditional execution request

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure

        """
        # Evaluate condition
        try:
            condition_met = bool(request.condition(self))
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[TDomainResult].fail(f"Condition evaluation failed: {e}")

        if not condition_met:
            # Condition not met, check if there's a false action
            if hasattr(request, "false_action") and request.false_action is not None:
                try:
                    false_result: object = None
                    if callable(request.false_action):
                        false_result = request.false_action(self)
                        # If the action returns a FlextResult, return it directly
                        if isinstance(false_result, FlextResult):
                            false_flext_result: FlextResult[TDomainResult] = (
                                false_result
                            )
                            return false_flext_result
                        false_result_value: TDomainResult = cast(
                            "TDomainResult", false_result
                        )
                        return FlextResult[TDomainResult].ok(false_result_value)
                    false_action_value: TDomainResult = cast(
                        "TDomainResult", request.false_action
                    )
                    return FlextResult[TDomainResult].ok(false_action_value)
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    return FlextResult[TDomainResult].fail(
                        f"False action execution failed: {e}"
                    )
            else:
                return FlextResult[TDomainResult].fail("Condition not met")

        # Condition met, check if there's a true action
        if hasattr(request, "true_action") and request.true_action is not None:
            try:
                if callable(request.true_action):
                    true_result = request.true_action(self)
                    # If the action returns a FlextResult, return it directly
                    if isinstance(true_result, FlextResult):
                        true_flext_result: FlextResult[TDomainResult] = true_result
                        return true_flext_result
                    true_result_value: TDomainResult = cast(
                        "TDomainResult", true_result
                    )
                    return FlextResult[TDomainResult].ok(true_result_value)
                true_action_value: TDomainResult = cast(
                    "TDomainResult", request.true_action
                )
                return FlextResult[TDomainResult].ok(true_action_value)
            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return FlextResult[TDomainResult].fail(
                    f"True action execution failed: {e}"
                )

        # No specific action, execute the default operation
        return self.execute()

    def execute_with_timeout(
        self, timeout_seconds: float
    ) -> FlextResult[TDomainResult]:
        """Execute operation with timeout handling.

        Args:
            timeout_seconds: Maximum execution time in seconds

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with timeout error

        """

        def timeout_handler(_signum: object, _frame: object) -> None:
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg)

        # Set up the timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            return self.execute()
        except TimeoutError as e:
            return FlextResult[TDomainResult].fail(str(e))
        finally:
            # Restore the old handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    # Helper classes for advanced service operations
    class _ExecutionHelper:
        """Helper class for execution-related operations utilities."""

        @staticmethod
        def prepare_execution_context(
            service: FlextService[
                object
            ],  # Generic service - works with any TDomainResult
        ) -> dict[str, object]:
            """Prepare execution context for a service."""
            context: dict[str, object] = {
                "service_type": service.__class__.__name__,
                "service_name": getattr(service, "_service_name", None),
                "timestamp": datetime.now(UTC),
            }
            return context

        @staticmethod
        def cleanup_execution_context(
            service: FlextService[object],
            context: dict[str, object],  # Generic service
        ) -> None:
            """Clean up execution context after operation."""
            # Basic cleanup - could be extended for more complex operations

    class _MetadataHelper:
        """Helper class for metadata extraction and formatting utilities."""

        @staticmethod
        def extract_service_metadata(
            service: FlextService[object],
            *,
            include_timestamps: bool = True,  # Generic service
        ) -> dict[str, object]:
            """Extract metadata from a service instance."""
            metadata: dict[str, object] = {
                "service_class": service.__class__.__name__,
                "service_name": getattr(service, "_service_name", None),
                "service_module": service.__class__.__module__,
            }

            if include_timestamps:
                now = datetime.now(UTC)
                metadata["created_at"] = now
                metadata["extracted_at"] = now

            return metadata

        @staticmethod
        def format_service_info(
            _service: FlextService[object],
            metadata: dict[str, object],  # Generic service
        ) -> str:
            """Format service information for display."""
            return f"Service: {metadata.get('service_type', 'Unknown')} ({metadata.get('service_name', 'unnamed')})"

    # =========================================================================
    # Protocol Implementation: ExecutableService[T]
    # =========================================================================

    def execute_service(self) -> FlextResult[TDomainResult]:
        """Execute service (ExecutableService protocol).

        Part of ExecutableService[T] protocol implementation.
        Delegates to execute() method.

        Returns:
            FlextResult[T]: Service execution result

        """
        return self.execute()

    # =========================================================================
    # Protocol Implementation: ContextAware
    # =========================================================================

    def set_context(self, context: dict[str, object]) -> FlextResult[None]:
        """Set context (ContextAware protocol).

        Part of ContextAware protocol implementation.
        Sets the execution context for the service.

        Args:
            context: Context dictionary

        Returns:
            FlextResult[None]: Success or context setting error

        """
        try:
            self._context = context
            return FlextResult[None].ok(None)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[None].fail(
                f"Context setting failed: {e}",
                error_code="CONTEXT_ERROR",
                error_data={"exception": str(e)},
            )

    # =========================================================================
    # Protocol Implementation: TimeoutSupport
    # =========================================================================

    def with_timeout(self, timeout_seconds: float) -> FlextResult[TDomainResult]:
        """Execute with timeout (TimeoutSupport protocol).

        Part of TimeoutSupport protocol implementation.
        Delegates to execute_with_timeout() method.

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            FlextResult[T]: Service execution result

        """
        return self.execute_with_timeout(timeout_seconds)

    def get_timeout(self) -> FlextResult[float]:
        """Get current timeout (TimeoutSupport protocol).

        Part of TimeoutSupport protocol implementation.

        Returns:
            FlextResult[float]: Current timeout value or error

        """
        try:
            timeout = getattr(self, "_timeout", 30.0)
            return FlextResult[float].ok(timeout)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[float].fail(
                f"Timeout retrieval failed: {e}",
                error_code="TIMEOUT_ERROR",
                error_data={"exception": str(e)},
            )


__all__: list[str] = [
    "FlextService",
]
