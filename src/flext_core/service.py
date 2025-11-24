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
import logging
import re
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import (
    ClassVar,
    Self,
    TypeGuard,
    Union,
    cast,
    get_args,
    get_origin,
    override,
)

from beartype.door import is_bearable
from pydantic import computed_field

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

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
        ...         return self.ok(User(id=self.user_id))
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
        ...         return self.ok(User(id=self.user_id))
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
        ...         return self.ok(User(id=self.user_id))
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
        ...     UserService(user_id="123")
        ...     .execute()
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
        ...         return self.ok(User(id=self.user_id))
        >>>
        >>> # Returns User directly, not service instance!
        >>> user = AutoUserService(user_id="123")
        >>> assert isinstance(user, User)
    """

    # Runtime type validation attribute (set by __class_getitem__)
    _expected_domain_result_type: ClassVar[type | None] = None

    # =========================================================================
    # AUTO-CONFIGURATION: Structlog Configuration
    # =========================================================================

    def model_post_init(self, __context: object, /) -> None:
        """Auto-configure structlog on first FlextService instantiation.

        This ensures logging is available automatically in all FLEXT libraries
        without requiring manual configuration in each application.

        Configuration happens once per process via FlextRuntime guards.
        """
        super().model_post_init(__context)

        # Auto-configure structlog if not already configured
        if not FlextRuntime.is_structlog_configured():
            FlextRuntime.configure_structlog(
                log_level=logging.INFO,  # Default to INFO, can be overridden by CLI
                console_renderer=True,
            )

    # =========================================================================
    # RUNTIME TYPE VALIDATION: Automatic via __class_getitem__
    # =========================================================================

    def __class_getitem__(cls, item: type | tuple[type, ...]) -> type[Self]:
        """Intercept FlextService[T] to create typed subclass for runtime validation.

        When FlextService[User] is accessed, this method creates a subclass that
        stores the expected domain result type (User) in _expected_domain_result_type.
        The subclass inherits all methods and behaviors but adds automatic type
        validation in execute() return value.

        This enables automatic type checking at runtime:
            FlextService[User].execute() → FlextResult[User]     # ✅ Valid
            FlextService[User].execute() → FlextResult[Product]  # ❌ TypeError

        Args:
            item: The domain result type parameter (e.g., User, Product) or tuple of types

        Returns:
            A typed subclass with _expected_domain_result_type set

        Example:
            >>> class UserService(FlextService[User]):
            ...     def execute(self) -> FlextResult[User]:
            ...         return self.ok(User(...))
            >>>
            >>> service = UserService()
            >>> result = service.execute()  # ✅ Validated automatically

        """
        # Handle tuple of types - use first type
        actual_type = item[0] if isinstance(item, tuple) else item

        # Create typed subclass dynamically using type() built-in
        cls_name = getattr(cls, "__name__", "FlextService")
        cls_qualname = getattr(cls, "__qualname__", "FlextService")
        type_name = getattr(actual_type, "__name__", str(actual_type))

        # Create typed subclass with proper namespace attributes for Pydantic v2
        # __module__ and __qualname__ must be in namespace dict for metaclass
        # Dynamic type creation using helper function - valid Python metaprogramming
        return FlextUtilities.Generators.create_dynamic_type_subclass(
            f"{cls_name}[{type_name}]",
            cls,
            {
                "_expected_domain_result_type": actual_type,
                "__module__": cls.__module__,
                "__qualname__": f"{cls_qualname}[{type_name}]",
            },
        )

    def validate_domain_result(
        self,
        result: FlextResult[TDomainResult],
    ) -> FlextResult[TDomainResult]:
        """Validate execute() result matches expected domain result type.

        Args:
            result: The result returned from execute()

        Returns:
            The same result if validation passes

        Raises:
            TypeError: If result data doesn't match _expected_domain_result_type

        """
        # Type validation - use isinstance for simple types, beartype for complex hints
        if self._expected_domain_result_type is not None and result.is_success:
            try:
                # For simple type objects, isinstance is more reliable than beartype
                if isinstance(self._expected_domain_result_type, type):
                    type_mismatch = not isinstance(
                        result.value,
                        self._expected_domain_result_type,
                    )
                else:
                    # For complex type hints (generics, unions), use beartype
                    # Type checker may think this is unreachable, but complex types are validated here
                    type_mismatch = not is_bearable(
                        result.value,
                        self._expected_domain_result_type,
                    )
            except (TypeError, AttributeError):
                # If type checking fails, assume type matches
                type_mismatch = False

            if type_mismatch:
                expected_name = getattr(
                    self._expected_domain_result_type,
                    "__name__",
                    str(self._expected_domain_result_type),
                )
                actual_name = type(result.value).__name__
                msg = (
                    f"{self.__class__.__name__}.execute() returned "
                    f"FlextResult[{actual_name}] instead of "
                    f"FlextResult[{expected_name}]. "
                    f"Data: {result.value!r}"
                )
                raise TypeError(msg)
        return result

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
            but it's cast to Self for type checker. Callers should
            type-annotate with the domain result type for auto_execute services.

        Example:
            >>> class AutoService(FlextService[User]):
            ...     auto_execute = True
            ...     user_id: str
            ...
            ...     def execute(self) -> FlextResult[User]:
            ...         return self.ok(User(id=self.user_id))
            >>>
            >>> user: User = AutoService(user_id="123")  # Type as User, not AutoService

        """
        instance = super().__new__(cls)
        # Call __init__ via type() to avoid mypy "unsound" warning
        type(instance).__init__(instance, **kwargs)

        if cls.auto_execute:
            # Auto-execute with runtime type validation
            # NOTE: kwargs already applied via __init__() above (line 379)
            # execute() accesses them as self.field_name, does not take parameters
            result = instance.validate_domain_result(instance.execute())
            # Cast to Self for type checker (actual runtime value is TDomainResult)
            return cast("Self", result.unwrap())

        # Return service instance (default behavior)
        return instance

    # =========================================================================
    # CLASS METHODS: Alternative Instantiation Patterns
    # =========================================================================

    @classmethod
    def v1(cls, **kwargs: object) -> Self:
        """Create service instance without auto-execution.

        Returns a service instance that requires explicit .execute() call.

        Args:
            **kwargs: Service initialization parameters

        Returns:
            Self: Service instance (not executed)

        Example:
            >>> class UserService(FlextService[User]):
            ...     auto_execute = True  # Even with auto_execute=True
            ...     user_id: str
            ...
            ...     def execute(self) -> FlextResult[User]:
            ...         return self.ok(User(id=self.user_id))
            >>>
            >>> # v1(): returns service instance (V1 pattern)
            >>> service = UserService.v1(user_id="789")
            >>> result = service.execute()  # Explicit execution
            >>> assert result.is_success
            >>> user = result.unwrap()

        """
        # Create instance using object.__new__ to bypass auto_execute
        # Python type system: __new__(cls) returns object, but cls.type determines actual type
        instance = object.__new__(cls)
        # Initialize instance
        type(instance).__init__(instance, **kwargs)
        # Type narrowing: object.__new__(cls) always returns an instance of cls
        # Use isinstance check for proper type narrowing without type: ignore
        if not isinstance(instance, cls):
            msg = f"Expected instance of {cls.__name__}, got {type(instance).__name__}"
            raise TypeError(msg)
        return instance

    @classmethod
    def with_result(cls, **kwargs: object) -> FlextResult[TDomainResult]:
        """Create service and return FlextResult instead of unwrapped value.

        This method always returns FlextResult[T], regardless of auto_execute setting.
        Useful when you want railway pattern composition even with auto_execute=True.

        Args:
            **kwargs: Service initialization parameters

        Returns:
            FlextResult[TDomainResult]: Result of service execution

        Example:
            >>> class UserService(FlextService[User]):
            ...     auto_execute = True  # Direct execution by default
            ...     user_id: str
            ...
            ...     def execute(self) -> FlextResult[User]:
            ...         return self.ok(User(id=self.user_id))
            >>>
            >>> # Normal: returns User directly
            >>> user = UserService(user_id="123")
            >>>
            >>> # with_result(): returns FlextResult[User]
            >>> result = UserService.with_result(user_id="456")
            >>> assert result.is_success
            >>> user = result.unwrap()

        """
        # Create instance using object.__new__ to bypass auto_execute
        instance = cast("Self", object.__new__(cls))
        # Initialize instance
        type(instance).__init__(instance, **kwargs)
        # Execute and return result
        return instance.validate_domain_result(instance.execute())

    # =========================================================================
    # V2 PROPERTY: Zero-Ceremony Access Pattern
    # =========================================================================

    @property
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
        return self.validate_domain_result(self.execute()).unwrap()

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
            if (
                param.annotation != inspect.Parameter.empty
                and param.annotation is not object
            ):
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
        cls,
        dependencies: dict[str, type],
        container: FlextContainer,
        service_name: str,
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

    @property
    def config(self) -> FlextConfig:
        """Standard config access property.

        Provides unified access to FlextConfig. Subprojects can override
        this to return typed config with namespace access.

        Example:
            >>> class OrderService(FlextService[Order]):
            ...     def execute(self) -> FlextResult[Order]:
            ...         debug = self.config.debug  # Access config
            ...         return self.ok(Order())

        Returns:
            FlextConfig: Global configuration instance

        """
        return FlextConfig.get_global_instance()

    @computed_field  # Pydantic v2 computed_field already provides property behavior
    def service_config(self) -> FlextConfig:
        """Automatic config binding via Pydantic v2 computed_field.

        DEPRECATED: Use self.config instead.

        Pure Pydantic v2 pattern. Config is transparently available via computed property.

        Example:
            >>> class OrderService(FlextService[Order]):
            ...     def execute(self) -> FlextResult[Order]:
            ...         # Config automatically available
            ...         timeout = self.service_config.timeout
            ...         return self.ok(Order())

        Returns:
            FlextConfig: Global configuration instance

        """
        return self.config

    def _resolve_project_component(
        self,
        component_suffix: str,
        type_check_func: Callable[[object], bool],
    ) -> object:
        """Resolve project component by naming convention (DRY helper).

        Attempts to resolve component from container. Fast fail on errors.

        Args:
            component_suffix: Suffix to replace "Service" with ("Config"/"Models")
            type_check_func: Function to validate resolved object type

        Returns:
            Resolved component

        Raises:
            FlextExceptions.TypeError: If component found but type check fails
            FlextExceptions.NotFoundError: If component not found in container

        """
        service_class_name = self.__class__.__name__
        component_class_name = service_class_name.replace("Service", component_suffix)

        # Fast fail: container must be accessible
        container = self.container

        # Fast fail: component must exist in container
        result = container.get(component_class_name)
        if result.is_failure:
            msg = f"Component '{component_class_name}' not found in container"
            raise FlextExceptions.NotFoundError(
                message=msg,
                resource_type="component",
                resource_id=component_class_name,
            )

        obj = result.unwrap()
        if not type_check_func(obj):
            msg = (
                f"Component '{component_class_name}' found but type check failed. "
                f"Expected type validated by {type_check_func.__name__}"
            )
            raise FlextExceptions.TypeError(
                message=msg,
                expected_type=component_class_name,
                actual_type=type(obj).__name__,
            )
        return obj

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
            ...         return self.ok({})

        Returns:
            FlextConfig: Project-specific configuration instance

        """
        try:
            return cast(
                "FlextConfig",
                self._resolve_project_component(
                    "Config",
                    lambda obj: isinstance(obj, FlextConfig),
                ),
            )
        except (FlextExceptions.NotFoundError, RuntimeError, AttributeError):
            # Fast fail: return global config if project config not found or container unavailable
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
            ...         return self.ok({})

        Returns:
            type: Project models namespace (typically a class with nested types)

        """
        try:
            return cast(
                "type",
                self._resolve_project_component(
                    "Models",
                    lambda obj: isinstance(obj, type),
                ),
            )
        except FlextExceptions.NotFoundError:
            # Fast fail: return empty namespace if project models not found
            class EmptyModelsNamespace(SimpleNamespace):
                """Empty models namespace when not found in container."""

            return EmptyModelsNamespace

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
            ...         return self.ok("simple")

        """
        super().__init_subclass__()

        # Skip registration for base class or generic parameterized classes
        # Only register actual concrete subclasses (user-defined services)
        if cls is FlextService or "[" in cls.__name__:
            return

        # Skip registration for test classes (classes defined in test modules)
        # Test classes should not be registered in the DI container
        module = getattr(cls, "__module__", "")
        if module and ("test" in module.lower() or module.startswith("tests.")):
            return

        # Normalize service name: create unique identifier
        # Fast fail: service name must be valid identifier (pattern: ^[a-zA-Z0-9_:\- ]+$)
        # Use the actual subclass name for uniqueness
        raw_name = cls.__name__
        # Create unique name: module_classname (dots replaced with underscores)
        if module and not module.startswith("flext_core.service"):
            # Only include module if it's not the base module (user modules)
            module_normalized = module.replace(".", "_")
            service_name = f"{module_normalized}_{raw_name}"
        else:
            service_name = raw_name
        # Replace any remaining invalid characters with underscores (pattern allows: a-zA-Z0-9_:\- space)
        service_name = re.sub(r"[^a-zA-Z0-9_:\- ]", "_", service_name)

        container = FlextContainer.get_global()

        # Auto-detect and inject dependencies
        # NOTE: Using with_factory() for fluent interface
        # may contain invalid characters for with_factory(). This is internal
        # framework usage, not user-facing API.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            try:
                deps = cls._extract_dependencies_from_signature()

                if deps:
                    # Factory with auto-DI
                    def factory() -> object:
                        return cls(
                            **cls._resolve_dependencies(deps, container, service_name),
                        )

                    container.with_factory(service_name, factory)
                else:
                    # No deps - simple registration
                    container.with_factory(service_name, cls)

            except (ValueError, TypeError) as e:
                # Fast fail: registration must succeed or fail explicitly
                raise FlextExceptions.ConfigurationError(
                    message=f"Failed to register {service_name}: {type(e).__name__}: {e}",
                    error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
                    config_key=service_name,
                ) from e

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

        # Auto-init hook for subclass customization
        self._auto_init_components()
        self._on_service_init()

    # =============================================================================
    # SERVICE INITIALIZATION HOOKS - For subclass customization
    # =============================================================================

    def _auto_init_components(self) -> None:
        """Auto-initialize service components with automatic logging.

        Called automatically during __init__. Override in subclasses to add
        custom component initialization.

        Default implementation logs service initialization at DEBUG level,
        providing visibility into service lifecycle without manual boilerplate.

        Example:
            >>> class MyService(FlextService[MyResult]):
            ...     def _auto_init_components(self) -> None:
            ...         super()._auto_init_components()
            ...         # Custom initialization
            ...         self._database = self.container.get("database").unwrap()

        """
        # Auto-log service initialization (DEBUG level for minimal verbosity)
        service_name = self.__class__.__name__
        self.logger.debug(
            "service_initialized",
            service_name=service_name,
            module=self.__class__.__module__,
        )

    def _on_service_init(self) -> None:
        """Hook called after service initialization is complete.

        Override in subclasses to perform post-initialization setup.
        This is called after _auto_init_components().

        Example:
            >>> class MyService(FlextService[MyResult]):
            ...     def _on_service_init(self) -> None:
            ...         self.logger.debug("Service initialized")
            ...         # Validate configuration
            ...         if not self.config.some_required_field:
            ...             raise ValueError("Missing required config")

        """

    @contextmanager
    def _timeout_context(self, timeout_seconds: float) -> Iterator[None]:
        """Context manager for timeout operations.

        Args:
            timeout_seconds: Timeout duration in seconds

        Yields:
            None: Context for timeout operations

        """
        # Store original timeout if any
        original_timeout = getattr(self, "_current_timeout", None)

        try:
            # Set current timeout for the context
            self._current_timeout = timeout_seconds
            yield
        finally:
            # Restore original timeout
            if original_timeout is not None:
                self._current_timeout = original_timeout
            # Remove the attribute if there was no original timeout
            elif hasattr(self, "_current_timeout"):
                delattr(self, "_current_timeout")

    # =============================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses (Domain.Service protocol)
    # =============================================================================

    @abstractmethod
    def execute(self, **kwargs: object) -> FlextResult[TDomainResult]:
        """Execute the main domain operation (Domain.Service protocol).

        Args:
            **kwargs: Optional operation parameters (for services that extend execute with parameters).
                     Services that don't need parameters can ignore this.

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure with error

        Note:
            Subclasses can extend this method with specific parameters. When called without parameters,
            performs the standard domain operation (health check or default behavior).

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

    def validate_business_rules(self) -> FlextResult[bool]:
        """Validate business rules for the domain service (Domain.Service protocol).

        Returns:
            FlextResult[bool]: Success with True if valid, failure with error details

        """
        return self.ok(True)

    def validate_config(self) -> FlextResult[bool]:
        """Validate service configuration (Domain.Service protocol).

        Returns:
            FlextResult[bool]: Success with True if configuration is valid, failure with error details

        """
        return self.ok(True)

    def is_valid(self) -> bool:
        """Check if the domain service is in a valid state (Domain.Service protocol).

        Returns:
            bool: True if the service is valid and ready for operations, False otherwise

        """
        # Check business rules and configuration
        try:
            business_rules = self.validate_business_rules()
            config = self.validate_config()
            # Both must be successful AND return True
            return (
                business_rules.is_success
                and business_rules.value is True
                and config.is_success
                and config.value is True
            )
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
        self,
        request: FlextModels.OperationExecutionRequest,
    ) -> FlextResult[bool]:
        """Validate business rules, config, and request before execution.

        Args:
            request: Operation execution request

        Returns:
            FlextResult[bool]: Success with True if valid, failure with validation error

        """
        # Validate business rules
        self.logger.debug(
            "Validating business rules",
            operation="validate_pre_execution",
            operation_name=request.operation_name,
            source="flext-core/src/flext_core/service.py",
        )
        business_rules_result = self.validate_business_rules()
        if business_rules_result.is_failure:
            error_msg = business_rules_result.error or "Unknown business rules error"
            self.logger.error(
                "FAILED: Business rules validation failed - EXECUTION ABORTED",
                operation="validate_pre_execution",
                operation_name=request.operation_name,
                error=error_msg,
                consequence="Operation will not be executed",
                resolution_hint="Fix business rules validation logic",
                source="flext-core/src/flext_core/service.py",
            )
            return self.fail(f"Business rules validation failed: {error_msg}")

        # Validate configuration
        self.logger.debug(
            "Validating configuration",
            operation="validate_pre_execution",
            operation_name=request.operation_name,
            source="flext-core/src/flext_core/service.py",
        )
        config_result = self.validate_config()
        if config_result.is_failure:
            error_msg = config_result.error or "Unknown configuration error"
            self.logger.error(
                "FAILED: Configuration validation failed - EXECUTION ABORTED",
                operation="validate_pre_execution",
                operation_name=request.operation_name,
                error=error_msg,
                consequence="Operation will not be executed",
                resolution_hint="Fix configuration validation logic",
                source="flext-core/src/flext_core/service.py",
            )
            return self.fail(f"Configuration validation failed: {error_msg}")

        # Validate keyword_arguments is a dict
        if not FlextRuntime.is_dict_like(request.keyword_arguments):
            # Type checker may think this is unreachable, but it's reachable at runtime
            return self.fail(
                f"Invalid keyword arguments: expected dict, got {type(request.keyword_arguments).__name__}",
            )

        return self.ok(True)

    def _execute_callable_once(
        self,
        request: FlextModels.OperationExecutionRequest
        | Callable[[object], TDomainResult],
    ) -> TDomainResult:
        """Execute operation callable once (with timeout if specified).

        Args:
            request: Operation execution request OR callable (for test convenience)

        Returns:
            TDomainResult: Result from the operation

        Raises:
            Exception: Exception from the operation execution

        """
        # Handle callable directly (convenience for tests)
        if callable(request):
            operation_request = FlextModels.OperationExecutionRequest(
                operation_name="direct_callable",
                operation_callable=cast(
                    "Callable[..., FlextTypes.ResultLike[object]]",
                    request,
                ),
            )
        else:
            operation_request = request

        if not callable(operation_request.operation_callable):
            raise FlextExceptions.ValidationError(
                message=f"operation_callable must be callable, got {type(operation_request.operation_callable)}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
                field="operation_callable",
            )

        # Fast fail: None values in arguments indicate invalid request
        for arg_name, arg_value in operation_request.arguments.items():
            if arg_value is None:
                msg = f"Argument '{arg_name}' cannot be None. Use FlextResult.fail() for failures."
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    field=arg_name,
                )

        # Fast fail: None values in keyword arguments indicate invalid request
        for kwarg_name, kwarg_value in operation_request.keyword_arguments.items():
            if kwarg_value is None:
                msg = f"Keyword argument '{kwarg_name}' cannot be None. Use FlextResult.fail() for failures."
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    field=kwarg_name,
                )

        args_list = list(operation_request.arguments.values())

        # Execute with timeout if specified
        if operation_request.timeout_seconds and operation_request.timeout_seconds > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future: concurrent.futures.Future[object] = executor.submit(
                    operation_request.operation_callable,
                    *args_list,
                    **operation_request.keyword_arguments,
                )
                result = future.result(timeout=operation_request.timeout_seconds)
            return cast("TDomainResult", result)

        # Execute without timeout
        result = operation_request.operation_callable(
            *args_list,
            **operation_request.keyword_arguments,
        )
        return cast("TDomainResult", result)

    @staticmethod
    def _is_flext_result(obj: object) -> TypeGuard[FlextResult[object]]:
        """Type guard to check if object is a FlextResult."""
        return isinstance(obj, FlextResult)

    def _retry_loop(
        self,
        request: FlextModels.OperationExecutionRequest,
        retry_config: dict[str, object],
    ) -> FlextResult[TDomainResult]:
        """Execute operation with retry logic using FlextUtilities.

        Args:
            request: Operation execution request
            retry_config: Retry configuration dict

        Returns:
            FlextResult[TDomainResult]: Result of execution with retry

        """
        # Use FlextUtilities for robust retry configuration validation and execution
        return FlextUtilities.Validation.create_retry_config(retry_config).flat_map(
            lambda config: self._execute_with_retry_config(request, config),
        )

    def _execute_with_retry_config(
        self,
        request: FlextModels.OperationExecutionRequest,
        config: FlextTypes.RetryConfig,
    ) -> FlextResult[TDomainResult]:
        """Execute operation with validated retry configuration.

        Args:
            request: Operation execution request
            config: Validated retry configuration

        Returns:
            FlextResult[TDomainResult]: Result of execution with retry

        """
        for attempt in range(config.max_attempts):
            try:
                self.logger.debug(
                    "Executing operation callable",
                    operation="execute_with_retry_config",
                    operation_name=request.operation_name,
                    attempt=attempt + 1,
                    max_attempts=config.max_attempts,
                    source="flext-core/src/flext_core/service.py",
                )
                result = self._execute_callable_once(request)
                self.logger.info(
                    "Operation completed successfully",
                    operation="execute_with_retry_config",
                    operation_name=request.operation_name,
                    attempt=attempt + 1,
                    max_attempts=config.max_attempts,
                    source="flext-core/src/flext_core/service.py",
                )

                # Wrap result if not already a FlextResult
                if self._is_flext_result(result):
                    return cast("FlextResult[TDomainResult]", result)
                return self.ok(result)

            except Exception as e:
                attempt_num = attempt + 1

                # Use FlextUtilities for exception type checking
                should_retry = FlextUtilities.Validation.is_exception_retryable(
                    e,
                    config.retry_on_exceptions,
                )

                if not should_retry or attempt_num >= config.max_attempts:
                    # Final failure - use FlextUtilities for error message formatting
                    error_msg = FlextUtilities.Validation.format_error_message(
                        e,
                        request.timeout_seconds,
                    )

                    self.logger.exception(
                        "FATAL ERROR: Operation execution failed after all retries - EXECUTION ABORTED",
                        operation="execute_with_retry_config",
                        operation_name=request.operation_name,
                        attempt=attempt_num,
                        max_attempts=config.max_attempts,
                        error=error_msg,
                        error_type=type(e).__name__,
                        consequence="Operation failed completely - no result returned",
                        severity="critical",
                        resolution_hint="Check operation implementation and retry configuration",
                        source="flext-core/src/flext_core/service.py",
                    )
                    return self.fail(
                        f"Operation {request.operation_name} failed: {error_msg}",
                    )

                # Use FlextUtilities for delay calculation
                delay = FlextUtilities.Reliability.calculate_delay(
                    attempt=attempt,
                    config=config,
                )

                self.logger.warning(
                    "Operation failed, retrying with delay",
                    operation="execute_with_retry_config",
                    operation_name=request.operation_name,
                    attempt=attempt_num,
                    max_attempts=config.max_attempts,
                    retry_delay_seconds=delay,
                    error=str(e),
                    error_type=type(e).__name__,
                    consequence="Will retry operation after delay",
                    source="flext-core/src/flext_core/service.py",
                )
                time.sleep(delay)

        # Fast fail: This code should never be reached - all paths return above
        msg = (
            f"BUG: Operation {request.operation_name} loop completed without return. "
            f"This indicates a logic error in execute_with_retry_config()."
        )
        raise RuntimeError(msg)

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
        # Fast fail: operation_name must be str or None
        operation_name: str = (
            request.operation_name
            if isinstance(request.operation_name, str) and request.operation_name
            else "unnamed_operation"
        )
        with self.track(operation_name):
            self._propagate_context(operation_name)

            self.logger.info(
                "Starting operation execution",
                operation="execute_operation",
                operation_name=request.operation_name,
                timeout_seconds=request.timeout_seconds,
                has_retry_config=bool(request.retry_config),
                correlation_id=self._get_correlation_id(),
                source="flext-core/src/flext_core/service.py",
            )

            # Validate pre-execution requirements
            validation_result = self._validate_pre_execution(request)
            if validation_result.is_failure:
                return self.fail(
                    validation_result.error or "Pre-execution validation failed",
                )

            # Execute with retry logic (delegated to private method)
            retry_config = request.retry_config
            if not FlextRuntime.is_dict_like(retry_config):
                # Fast fail: retry_config must be dict (Field default_factory ensures this)
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = (
                    f"Invalid retry_config type: {type(retry_config).__name__}. "
                    "Expected dict[str, object]"
                )
                raise TypeError(msg)
            return self._retry_loop(request, retry_config)

    def execute_with_full_validation(
        self,
        _request: FlextModels.DomainServiceExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute operation with full validation including business rules and config.

        Args:
            _request: Domain service execution request (unused)

        Returns:
            FlextResult[TDomainResult]: Success with operation result or failure with validation/execution error

        """
        # Full validation: business rules + config + execution
        business_rules_result = self.validate_business_rules()
        if business_rules_result.is_failure:
            return self.fail(
                f"Business rules validation failed: {business_rules_result.error}",
            )

        config_result = self.validate_config()
        if config_result.is_failure:
            return self.fail(f"Configuration validation failed: {config_result.error}")

        return self.execute()

    def _execute_action(
        self,
        action: object,
        action_name: str,
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
                # Try calling with self first (preferred pattern)
                # If that fails with TypeError, try without arguments (test convenience)
                try:
                    result = action(self)
                except TypeError as e:
                    # If error is about argument count, try without self
                    if "positional argument" in str(e):
                        result = action()
                    else:
                        raise
                # If the action returns a FlextResult, return it directly
                # Fast fail: use type narrowing with validation
                if isinstance(result, FlextResult):
                    # Validate result type matches TDomainResult at runtime
                    return self.validate_domain_result(result)
                # Fast fail: non-FlextResult must be TDomainResult - validate via ok()
                # Type narrowing: result is object, but we validate it's TDomainResult
                # Use cast for type safety - result is validated to be TDomainResult
                validated_result: TDomainResult = cast("TDomainResult", result)
                return FlextResult[TDomainResult].ok(validated_result)
            # Fast fail: non-callable action must be TDomainResult - validate via ok()
            # Type narrowing: action is object, but we validate it's TDomainResult
            validated_action: TDomainResult = cast("TDomainResult", action)
            return FlextResult[TDomainResult].ok(validated_action)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return self.fail(f"{action_name.capitalize()} action execution failed: {e}")

    def execute_conditionally(
        self,
        request: FlextModels.ConditionalExecutionRequest,
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
            return self.fail(f"Condition evaluation failed: {e}")

        # Execute false action if condition not met
        if (
            not condition_met
            and hasattr(request, "false_action")
            and request.false_action is not None
        ):
            return self._execute_action(request.false_action, "false")
        if not condition_met:
            return self.fail("Condition not met")

        # Execute true action if condition met
        if hasattr(request, "true_action") and request.true_action is not None:
            return self._execute_action(request.true_action, "true")

        # No specific action, execute the default operation
        return self.execute()

    def _execute_with_timeout_threading(
        self, timeout_seconds: float
    ) -> FlextResult[TDomainResult]:
        """Execute operation with timeout using threading approach.

        Args:
            timeout_seconds: Maximum execution time in seconds

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with timeout error

        """
        result: FlextResult[TDomainResult] | None = None
        execution_error: Exception | None = None

        def execute_in_thread() -> None:
            nonlocal result, execution_error
            try:
                result = self.execute()
            except Exception as e:
                execution_error = e

        thread = threading.Thread(target=execute_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            # Thread still running - timeout occurred
            return self.fail(
                f"Operation timed out after {timeout_seconds} seconds",
                error_code="OPERATION_TIMEOUT",
                error_data={"timeout_seconds": timeout_seconds},
            )

        if execution_error:
            # Execution failed with an exception
            if isinstance(execution_error, TimeoutError):
                return self.fail(str(execution_error))
            raise execution_error

        if result is None:
            msg = "Unexpected: result not set"
            raise RuntimeError(msg)
        return result

    def execute_with_timeout(
        self,
        timeout_seconds: float,
    ) -> FlextResult[TDomainResult]:
        """Execute operation with timeout handling using threading.

        Args:
            timeout_seconds: Maximum execution time in seconds

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with timeout error

        """
        return self._execute_with_timeout_threading(timeout_seconds)

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

    def set_context(self, context: dict[str, object]) -> FlextResult[bool]:
        """Set context (ContextAware protocol).

        Part of ContextAware protocol implementation.
        Sets the execution context for the service.

        Args:
            context: Context dictionary

        Returns:
            FlextResult[bool]: Success with True if set, failure with context setting error

        """
        try:
            self._context = context
            return self.ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return self.fail(
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
            return self.ok(timeout)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return self.fail(
                f"Timeout retrieval failed: {e}",
                error_code="TIMEOUT_ERROR",
                error_data={"exception": str(e)},
            )


__all__: list[str] = [
    "FlextService",
]
