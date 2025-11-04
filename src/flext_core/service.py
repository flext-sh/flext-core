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
from datetime import UTC, datetime
from typing import ClassVar, Self, Union, cast, get_args, get_origin, override

from pydantic import computed_field

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult

# =========================================================================
# FLEXT SERVICE - Domain Service Base Class
# =========================================================================


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins,
    ABC,
):
    """Base class for domain services (Type-Safe & Zero Ceremony).

    **CONTROL EXECUTION:**
    Set `auto_execute = True` in subclass to enable zero-ceremony instantiation:

        >>> class UserService(FlextService[User]):
        ...     auto_execute = True  # Enable auto-execution
        ...     user_id: str
        ...
        ...     def execute(self) -> FlextResult[User]:
        ...         return FlextResult.ok(User(id=self.user_id))
        >>>
        >>> # V2 Auto: Direct result (4 chars!)
        >>> user = UserService(user_id="123")  # Returns User directly!
        >>> print(user.name)  # ✅ Type-safe!

    **USAGE PATTERN V2 Manual (Property):**
    - Service(**params).result → Returns TDomainResult directly
    - Type-safe: IDE autocomplete + type checkers work perfectly
    - 68% less code than V1
    - Pydantic @computed_field (native, no hacks)

    **USAGE PATTERN V1 (Explicit - Still Supported):**
    - Service(**params).execute() → Returns FlextResult[TDomainResult]
    - result.unwrap() → Returns TDomainResult or raises
    - Use when you need FlextResult for railway pattern composition

    **EXAMPLE V2 AUTO (Zero Ceremony - Recommended):**

        >>> class UserService(FlextService[User]):
        ...     auto_execute = True  # ← Enable auto-execution
        ...     user_id: str
        ...
        ...     def execute(self) -> FlextResult[User]:
        ...         return FlextResult.ok(User(id=self.user_id))
        >>>
        >>> # Just instantiate - it returns User directly!
        >>> user = UserService(user_id="123")
        >>> print(user.name)  # ✅ Type-safe, IDE autocomplete works!
        >>> print(user.id)  # ✅ 'id' is now available for domain models!
        >>>
        >>> # Error handling via try/except (Pythonic)
        >>> try:
        ...     user = UserService(user_id="123")
        ...     print(user.name)
        >>> except FlextExceptions.BaseError as e:
        ...     print(f"Error: {e}")

    **EXAMPLE V2 MANUAL (Property Pattern):**

        >>> class UserService(FlextService[User]):
        ...     # auto_execute defaults to False
        ...     user_id: str
        ...
        ...     def execute(self) -> FlextResult[User]:
        ...         return FlextResult.ok(User(id=self.user_id))
        >>>
        >>> # Access .result property
        >>> user = UserService(user_id="123").result
        >>> print(user.name)  # ✅ Type-safe!

    **EXAMPLE V1 (Explicit FlextResult):**

        >>> # V1: Explicit FlextResult handling (19 chars)
        >>> result = UserService(user_id="123").execute()
        >>>
        >>> # V1: Handle success/failure
        >>> if result.is_success:
        ...     user = result.unwrap()
        ...     print(user.id)
        >>> else:
        ...     print(f"Error: {result.error}")
        >>>
        >>> # V1: Monadic composition
        >>> result = (
        ...     UserService(user_id="123").execute()
        ...     .map(lambda u: u.name.upper())
        ...     .and_then(lambda name: save_user(name))
        ... )

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
    """

    # Mixin-provided infrastructure properties (explicit type documentation)
    # These are declared in FlextMixins and available on all service instances
    # container: FlextContainer - Singleton DI container access
    # context: FlextContext - Scoped request/operation context
    # logger: FlextLogger - Structured logging with context
    # config: FlextConfig - Global configuration instance
    # track: Method[Iterator] - Context manager for operation tracking

    # =========================================================================
    # CLASS CONTROL: Auto-Execution
    # =========================================================================

    auto_execute: ClassVar[bool] = False  # Default: manual execution
    """Control automatic execution on instantiation.

    Set to True in subclasses to enable zero-ceremony pattern where
    instantiation directly returns the unwrapped domain result.

    Example:
        >>> class AutoUserService(FlextService[User]):
        ...     auto_execute = True
        ...     user_id: str
        ...     def execute(self) -> FlextResult[User]:
        ...         return FlextResult.ok(User(id=self.user_id))
        >>>
        >>> # Returns User directly, not service instance!
        >>> user = AutoUserService(user_id="123")
        >>> assert isinstance(user, User)
    """

    # =========================================================================
    # V2 OVERRIDE: Zero-Ceremony Instantiation
    # =========================================================================

    def __new__(cls, **kwargs: object) -> Self:
        """Control execution flow based on auto_execute class attribute.

        If auto_execute=True: Returns unwrapped domain result (cast to Self)
        If auto_execute=False: Returns service instance (default)

        Args:
            **kwargs: Service initialization parameters

        Returns:
            Service instance OR unwrapped domain result (cast to Self for type safety)

        Note:
            When auto_execute=True, the actual runtime value is TDomainResult,
            but it's cast to Self for type checker compatibility. Callers should
            type-annotate with the domain result type for auto_execute services.

        Example:
            >>> class AutoService(FlextService[User]):
            ...     auto_execute = True
            ...     user_id: str
            ...     def execute(self) -> FlextResult[User]:
            ...         return FlextResult.ok(User(id=self.user_id))
            >>>
            >>> user: User = AutoService(user_id="123")  # Type as User, not AutoService

        """
        instance = cast("Self", super().__new__(cls))
        # Call __init__ via type() to avoid mypy "unsound" warning
        type(instance).__init__(instance, **kwargs)

        if cls.auto_execute:
            # Auto-execute and return unwrapped result (cast for type safety)
            return cast("Self", instance.execute().unwrap())

        # Return service instance (default behavior)
        return instance

    # =========================================================================
    # V2 PROPERTY: Zero-Ceremony Access Pattern
    # =========================================================================

    @computed_field  # Pydantic 2 native API - acts as property
    def result(self) -> TDomainResult:
        """Auto-execute and unwrap shorthand (V2 pattern).

        Zero-ceremony access to domain result. Type-safe alternative to
        .execute().unwrap() pattern with 68% less code.

        Property name 'result' chosen to avoid conflicts with common field names
        like 'value', 'data', 'id', 'name', etc.

        Returns:
            TDomainResult: Unwrapped domain result from execute()

        Raises:
            FlextExceptions.BaseError: If execute() fails

        Example:
            >>> # V2: Zero ceremony (7 chars)
            >>> user = UserService(user_id="123").result
            >>> print(user.name)  # Type-safe, IDE autocomplete works!
            >>>
            >>> # V1: Still works (19 chars)
            >>> user = UserService(user_id="123").execute().unwrap()
            >>>
            >>> # Type inference works perfectly
            >>> user: User = UserService(user_id="123").result  # ✅ Mypy happy

        Note:
            This is a Pydantic @computed_field, meaning:
            - Lazy evaluation (only executes when accessed)
            - Type-safe (type checkers infer TDomainResult automatically)
            - Serializable (included in model_dump if configured)
            - Zero performance overhead vs manual .execute().unwrap()

        """
        return self.execute().unwrap()

    @classmethod
    def _extract_dependencies_from_signature(cls) -> dict[str, type]:
        """Extract dependencies from __init__ signature.

        Returns:
            Dict mapping parameter names to their types

        """
        sig = inspect.signature(cls.__init__)
        deps = {}

        for name, param in sig.parameters.items():
            # Skip special parameters
            if name in {"self", "config", "data"} or param.kind in {
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            }:
                continue

            # Only include typed parameters
            if param.annotation != inspect.Parameter.empty and param.annotation is not object:
                # Extract base type from Optional/Union
                dep_type = param.annotation
                origin = get_origin(dep_type)
                if origin is Union:
                    # Get first non-None type
                    args = get_args(dep_type)
                    dep_type = next((a for a in args if a is not type(None)), dep_type)
                deps[name] = dep_type

        return deps

    @classmethod
    def _resolve_dependencies(
        cls, dependencies: dict[str, type], container: FlextContainer, service_name: str
    ) -> dict[str, object]:
        """Resolve dependencies from container.

        Args:
            dependencies: Map of param_name -> type
            container: DI container
            service_name: Service name for error messages

        Returns:
            Dict of resolved dependencies

        Raises:
            FlextExceptions.ConfigurationError: If required dependencies cannot be resolved

        """
        resolved = {}
        missing = []

        for param_name, param_type in dependencies.items():
            # Try by name first, then by type name
            result = container.get(param_name)
            if result.is_failure:
                type_name = getattr(param_type, "__name__", str(param_type))
                result = container.get(type_name)

            if result.is_success:
                resolved[param_name] = result.unwrap()
            else:
                missing.append(f"{param_name}({param_type})")

        if missing:
            raise FlextExceptions.ConfigurationError(
                message=f"Cannot create {service_name}: unresolved dependencies: {', '.join(missing)}",
                error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
                config_key=service_name,
            )

        return resolved

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
                config = config_result.unwrap()
                if isinstance(config, FlextConfig):
                    return config
        except Exception as e:
            # Fall back to global config if resolution fails
            self.logger.debug(f"Config resolution failed, using global instance: {e}")

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
        except Exception as e:
            # Return default namespace if resolution fails
            self.logger.debug(f"Models resolution failed, using default namespace: {e}")

        # Return a minimal namespace type if not found
        return type("ModelsNamespace", (), {})

    def __init_subclass__(cls) -> None:
        """Automatic service registration with dependency injection.

        Services are automatically registered in FlextContainer with smart factories
        that auto-inject dependencies based on constructor parameters.

        **Features**:
        - Auto-detects constructor dependencies via type hints
        - Handles Optional[T] and Union types
        - Provides clear error messages for missing dependencies
        - Falls back to simple registration if DI fails
        - Adds _flext_v1_mode to subclass __init__ signature for type checkers

        **Example - Auto DI**:
            >>> class UserService(FlextService[User]):
            ...     def __init__(self, database: Database, cache: Cache):
            ...         super().__init__()
            ...         self.database = database  # Auto-injected
            ...         self.cache = cache  # Auto-injected

        **Example - No DI**:
            >>> class SimpleService(FlextService[str]):
            ...     def execute(self) -> FlextResult[str]:
            ...         return FlextResult[str].ok("simple")

        """
        super().__init_subclass__()

        service_name = cls.__name__
        container = FlextContainer.get_global()

        # Auto-detect and inject dependencies
        try:
            deps = cls._extract_dependencies_from_signature()

            if deps:
                # Factory with auto-DI
                def factory() -> object:
                    return cls(**cls._resolve_dependencies(deps, container, service_name))
                container.register_factory(service_name, factory)
            else:
                # No deps - simple registration
                container.register_factory(service_name, cls)

        except (ValueError, TypeError):
            # Fallback: register without DI
            try:
                container.register_factory(service_name, cls)
            except Exception as inner_e:
                raise FlextExceptions.ConfigurationError(
                    message=f"Failed to register {service_name}",
                    error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
                    config_key=service_name,
                ) from inner_e

    @override
    def __init__(self, **data: object) -> None:
        """Initialize domain service with Pydantic validation and infrastructure.

        Args:
            **data: Pydantic model fields

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

    def _validate_pre_execution(
        self, request: FlextModels.OperationExecutionRequest
    ) -> FlextResult[None]:
        """Validate business rules, config, and request before execution.

        Args:
            request: Operation execution request

        Returns:
            FlextResult[None]: Success or failure with validation error

        """
        # Validate business rules
        business_rules_result = self.validate_business_rules()
        if business_rules_result.is_failure:
            self.logger.error(
                f"Business rules validation failed for operation: {request.operation_name}",
                extra={"error": business_rules_result.error},
            )
            return FlextResult[None].fail(
                f"Business rules validation failed: {business_rules_result.error}"
            )

        # Validate configuration
        config_result = self.validate_config()
        if config_result.is_failure:
            self.logger.error(
                f"Configuration validation failed for operation: {request.operation_name}",
                extra={"error": config_result.error},
            )
            return FlextResult[None].fail(
                f"Configuration validation failed: {config_result.error}"
            )

        # Validate keyword_arguments is a dict
        if not isinstance(request.keyword_arguments, dict):
            return FlextResult[None].fail(
                f"Invalid keyword arguments: expected dict, got {type(request.keyword_arguments).__name__}"
            )

        return FlextResult[None].ok(None)

    def _execute_callable_once(
        self, request: FlextModels.OperationExecutionRequest
    ) -> TDomainResult:
        """Execute operation callable once (with timeout if specified).

        Args:
            request: Operation execution request

        Returns:
            TDomainResult: Result from the operation

        Raises:
            Exception: Any exception from the operation execution

        """
        if not callable(request.operation_callable):
            raise FlextExceptions.ValidationError(
                message=f"operation_callable must be callable, got {type(request.operation_callable)}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
                field="operation_callable",
            )

        # Filter out None values from arguments
        filtered_args = [v for v in request.arguments.values() if v is not None]

        # Execute with timeout if specified
        if request.timeout_seconds and request.timeout_seconds > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    request.operation_callable,
                    *filtered_args,
                    **request.keyword_arguments,
                )
                result = future.result(timeout=request.timeout_seconds)
            return cast("TDomainResult", result)

        # Execute without timeout
        result = request.operation_callable(*filtered_args, **request.keyword_arguments)
        return cast("TDomainResult", result)

    def _retry_loop(
        self, request: FlextModels.OperationExecutionRequest, retry_config: dict[str, object]
    ) -> FlextResult[TDomainResult]:
        """Execute operation with retry logic.

        Args:
            request: Operation execution request
            retry_config: Retry configuration dict

        Returns:
            FlextResult[TDomainResult]: Result of execution with retry

        """
        # Extract and validate retry parameters
        try:
            max_attempts = int(cast("int", retry_config.get("max_attempts", 1) or 1))
            initial_delay = float(cast("float", retry_config.get("initial_delay_seconds", 0.1) or 0.1))
            max_delay = float(cast("float", retry_config.get("max_delay_seconds", 60.0) or 60.0))
            exponential_backoff = bool(retry_config.get("exponential_backoff"))
            retry_on_exceptions = cast(
                "list[type[Exception]]",
                retry_config.get("retry_on_exceptions") or [Exception],
            )

            # Validate backoff_multiplier if present
            if "backoff_multiplier" in retry_config:
                backoff_mult = float(cast("float", retry_config["backoff_multiplier"]))
                if backoff_mult < 1.0:
                    msg = "Invalid retry configuration: backoff_multiplier must be >= 1.0"
                    return FlextResult.fail(
                        msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
        except (ValueError, TypeError) as e:
            return FlextResult.fail(
                f"Invalid retry configuration: {e}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        last_exception: Exception | None = None

        for attempt in range(max_attempts):
            try:
                result = self._execute_callable_once(request)
                self.logger.info(
                    f"Operation completed successfully: {request.operation_name}"
                )

                # Wrap result if not already a FlextResult
                if isinstance(result, FlextResult):
                    return cast("FlextResult[TDomainResult]", result)
                return FlextResult[TDomainResult].ok(cast("TDomainResult", result))

            except Exception as e:
                last_exception = e

                # Check if we should retry this exception
                should_retry = any(
                    isinstance(e, exc_type) for exc_type in retry_on_exceptions
                )

                if not should_retry or attempt >= max_attempts - 1:
                    # Final failure
                    error_msg = str(e) or type(e).__name__
                    if isinstance(e, (TimeoutError, concurrent.futures.TimeoutError)):
                        error_msg = f"Operation timed out after {request.timeout_seconds} seconds"

                    self.logger.exception(
                        f"Operation execution failed: {request.operation_name}",
                        extra={"error": error_msg, "error_type": type(e).__name__},
                    )
                    return FlextResult[TDomainResult].fail(
                        f"Operation {request.operation_name} failed: {error_msg}"
                    )

                # Calculate delay for retry
                delay = (
                    min(initial_delay * (2**attempt), max_delay)
                    if exponential_backoff
                    else min(initial_delay, max_delay)
                )

                self.logger.warning(
                    f"Operation {request.operation_name} failed (attempt {attempt + 1}/{max_attempts}), retrying in {delay}s",
                    extra={"error": str(e), "error_type": type(e).__name__},
                )
                time.sleep(delay)

        # Fallback (should not reach here)
        return FlextResult[TDomainResult].fail(
            f"Operation {request.operation_name} failed after {max_attempts} attempts: {last_exception}"
        )

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

            # Validate pre-execution requirements
            validation_result = self._validate_pre_execution(request)
            if validation_result.is_failure:
                return FlextResult[TDomainResult].fail(validation_result.error)

            # Execute with retry logic (delegated to private method)
            return self._retry_loop(request, request.retry_config or {})

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

    def _execute_action(
        self, action: object, action_name: str
    ) -> FlextResult[TDomainResult]:
        """Execute a conditional action (true or false).

        Args:
            action: Action to execute (callable or value)
            action_name: Name for error messages ("true" or "false")

        Returns:
            FlextResult[TDomainResult]: Result of action execution

        """
        try:
            if callable(action):
                result = action(self)
                # If the action returns a FlextResult, return it directly
                if isinstance(result, FlextResult):
                    return cast("FlextResult[TDomainResult]", result)
                return FlextResult[TDomainResult].ok(cast("TDomainResult", result))
            return FlextResult[TDomainResult].ok(cast("TDomainResult", action))
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[TDomainResult].fail(
                f"{action_name.capitalize()} action execution failed: {e}"
            )

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

        # Execute false action if condition not met
        if not condition_met:
            if hasattr(request, "false_action") and request.false_action is not None:
                return self._execute_action(request.false_action, "false")
                return FlextResult[TDomainResult].fail("Condition not met")

        # Execute true action if condition met
        if hasattr(request, "true_action") and request.true_action is not None:
            return self._execute_action(request.true_action, "true")

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
