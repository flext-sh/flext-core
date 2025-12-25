"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services that participate in CQRS flows. It relies
on structural typing to satisfy ``p.Service`` and aligns with the
dispatcher-centric architecture.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import Self, cast, override

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    computed_field,
)
from pydantic.fields import ModelPrivateAttr

from flext_core._dispatcher import CircuitBreakerManager, RateLimiterManager
from flext_core._models.base import FlextModelsBase
from flext_core.constants import c
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins as x, require_initialized
from flext_core.models import m
from flext_core.protocols import p
from flext_core.registry import FlextRegistry
from flext_core.result import r
from flext_core.settings import FlextSettings
from flext_core.typings import t
from flext_core.utilities import u


class FlextService[TDomainResult](
    m.ArbitraryTypesModel,
    x,
    ABC,
):
    """Base class for domain services used in CQRS flows.

    Subclasses implement ``execute`` to run business logic and return
    ``FlextResult`` values. The base inherits :class:`FlextMixins` (which extends
    :class:`FlextRuntime`) so services can reuse runtime automation for creating
    scoped config/context/container triples via :meth:`create_service_runtime`
    while remaining protocol compliant via structural typing.

    **V2 Auto-Execute Pattern**:
    Services can use the auto-execute pattern by setting ``auto_execute = True``
    as a class variable. When enabled, instantiating the service automatically
    executes it and returns the domain result directly instead of the service instance.

    Example:
        class UserService(FlextService[User]):
            auto_execute = True  # Returns User directly, not service instance

            def execute(self) -> r[User]:
                return r.ok(User(id="123", name="Alice"))

        # Usage: user = UserService()  # Returns User object directly

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    def __new__(
        cls,
        **kwargs: t.GeneralValueType,
    ) -> Self:
        """Create service instance.

        For services with auto_execute = True, returns the execution result directly.
        For services without auto_execute, returns the service instance.

        Args:
            **kwargs: Configuration parameters for service initialization

        Returns:
            Self | TDomainResult: Service instance or execution result

        Note:
            This method uses a union return type to support the auto-execute pattern.
            When auto_execute=True, the method returns TDomainResult directly.
            When auto_execute=False, the method returns Self (the service instance).

        """
        instance = super().__new__(cls)
        # Check for auto_execute ClassVar
        auto_execute = getattr(cls, "auto_execute", False)
        if auto_execute:
            # Verify class is concrete (not abstract)
            if inspect.isabstract(cls):
                msg = (
                    f"Class {cls.__name__} has auto_execute=True but is still abstract. "
                    "Implement all abstract methods in the concrete class."
                )
                raise TypeError(msg)
            # Initialize the instance
            # Use type(instance) to avoid mypy error about accessing __init__ on instance
            type(instance).__init__(instance, **kwargs)
            # Execute and get result (V2 Auto pattern)
            # Concrete classes with auto_execute=True must implement execute()
            # After isabstract check, we know execute() exists on concrete class
            execute_method = getattr(instance, c.Mixins.METHOD_EXECUTE, None)
            if not callable(execute_method):
                msg = f"Class {cls.__name__} must implement execute() method"
                raise TypeError(msg)
            # Type narrowing: execute_method is callable and returns r[TDomainResult]
            # execute() is abstract method that returns r[TDomainResult] per class definition
            result_raw = execute_method()
            if not isinstance(result_raw, r):
                msg = f"execute() must return r, got {type(result_raw).__name__}"
                raise TypeError(msg)
            # Type narrowing: result_raw is r, and execute() signature guarantees r[TDomainResult]
            # isinstance() check confirms type but doesn't preserve generic type parameter
            result: r[TDomainResult] = result_raw  # type: ignore[assignment]
            # For auto_execute=True, return the result value directly (V2 Auto pattern)
            # This allows: user = AutoGetUserService(user_id="123") to get User object
            if result.is_failure:
                error_msg = result.error or "Service execution failed"
                raise FlextExceptions.BaseError(error_msg)
            # Return the unwrapped value directly (breaks static typing but is intended behavior)
            # Type narrowing: result.value is TDomainResult, which may be Self in some cases
            # This is a runtime pattern where TDomainResult can be the service instance itself
            # Type narrowing: result.is_success confirmed above, so result.value is TDomainResult
            return result.value  # type: ignore[return-value]
        # For auto_execute=False, return instance (normal pattern)
        # Pydantic BaseModel calls __init__ automatically after __new__,
        # so we don't need to call it manually here
        return instance

    @property
    def result(self) -> TDomainResult:
        """Get the execution result, raising exception on failure.

        TODO(docs/FLEXT_SERVICE_ARCHITECTURE.md#zero-ceremony): reassess
        migrating to ``@computed_field`` once we confirm that Pydantic service
        serialisation needs standardised dumps.
        """
        if not hasattr(self, "_execution_result"):
            # Lazy execution for services without auto_execute
            execution_result = self.execute()
            self._execution_result = execution_result

        result = self._execution_result
        if result.is_success:
            return result.value
        # On failure, raise exception
        raise FlextExceptions.BaseError(result.error or "Service execution failed")

    @override
    def __init__(
        self,
        **data: t.GeneralValueType,
    ) -> None:
        """Initialize service with configuration data.

        Sets up the service instance with optional configuration parameters
        passed through **data. Delegates to parent classes for proper
        initialization of mixins, models, and infrastructure components.

        Auto-discovery of handler-decorated methods enables zero-config handler
        registration: developers can mark methods with @h.handler() and they are
        automatically discovered during initialization.

        Args:
            **data: Configuration parameters for service initialization

        """
        runtime = self._create_initial_runtime()
        # Type narrowing: runtime.context is FlextContext
        # Use isinstance check for type narrowing since FlextContext is concrete class
        if not isinstance(runtime.context, FlextContext):
            msg = f"Expected FlextContext, got {type(runtime.context).__name__}"
            raise TypeError(msg)
        context = runtime.context

        with context.Service.service_context(
            self.__class__.__name__,
            runtime.config.version,
        ):
            super().__init__(**data)

        # Set attributes directly - PrivateAttr allows assignment without validation
        self._context = runtime.context
        # Type narrowing: runtime.config is p.Config, but we need FlextSettings
        # All implementations of p.Config in FLEXT are FlextSettings or subclasses
        # Use isinstance check for type narrowing since FlextSettings is concrete class
        if not isinstance(runtime.config, FlextSettings):
            msg = f"Expected FlextSettings, got {type(runtime.config).__name__}"
            raise TypeError(msg)
        self._config = runtime.config
        self._container = runtime.container
        self._runtime = runtime

        # Auto-discovery of handler-decorated methods for zero-config handler setup
        # Discovers all methods marked with @h.handler() decorator
        # Makes them available for dispatcher routing without explicit registration
        self._discovered_handlers = (
            FlextHandlers.Discovery.scan_class(self.__class__)
            if FlextHandlers.Discovery.has_handlers(self.__class__)
            else []
        )

    # Use PrivateAttr for private attributes (Pydantic v2 pattern)
    # PrivateAttr allows setting attributes without validation and bypasses __setattr__
    # Type annotations using PrivateAttr with explicit type hints
    _context: p.Ctx | None = PrivateAttr(default=None)
    _config: FlextSettings | None = PrivateAttr(default=None)
    _container: p.DI | None = PrivateAttr(default=None)
    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)
    _discovered_handlers: list[tuple[str, m.HandlerDecoratorConfig]] = PrivateAttr(
        default_factory=list,
    )
    _auto_result: TDomainResult | None = None

    @classmethod
    def _get_service_config_type(cls) -> type[FlextSettings]:
        """Get the config type for this service class.

        Services can override this method to specify their specific config type.
        Defaults to FlextSettings for generic services.

        Returns:
            type[FlextSettings]: The config class to use for this service

        """
        return FlextSettings  # Runtime return needs concrete class

    @classmethod
    def _create_runtime(
        cls,
        *,
        config_type: type[FlextSettings] | None = None,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        context: p.Ctx | None = None,
        subproject: str | None = None,
        services: Mapping[
            str,
            t.GeneralValueType | BaseModel | p.VariadicCallable[t.GeneralValueType],
        ]
        | None = None,
        factories: Mapping[
            str,
            Callable[
                [],
                (t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue]),
            ],
        ]
        | None = None,
        resources: Mapping[str, Callable[[], t.GeneralValueType]] | None = None,
        container_overrides: Mapping[str, t.FlexibleValue] | None = None,
        wire_modules: Sequence[ModuleType] | None = None,
        wire_packages: Sequence[str] | None = None,
        wire_classes: Sequence[type] | None = None,
    ) -> m.ServiceRuntime:
        """Materialize config, context, and container with DI wiring in one call.

        This method provides the same parameterized automation previously found in
        ``FlextRuntime.create_service_runtime`` but uses the factory methods of each
        class directly (Clean Architecture - each class knows how to instantiate itself).

        All runtime components are created via the bootstrap options pattern:
        1. Config materialization with overrides
        2. Context creation with initial data
        3. Container creation with registrations
        4. Dispatcher creation with auto-discovery
        5. Registry creation

        All parameters are optional and allow callers to:
        - Clone or materialize configuration models with optional overrides.
        - Seed containers with services/factories/resources without additional
          registration calls.
        - Apply container configuration overrides before wiring modules, packages,
          or classes for ``@inject`` usage.

        This method is called by :meth:`_create_initial_runtime` which uses
        :meth:`_runtime_bootstrap_options` to get the configuration options.
        """
        # 1. Config materialization with overrides
        config_cls = config_type or FlextSettings
        # Pydantic v2: Use model_validate for proper validation with overrides
        runtime_config = config_cls.model_validate(config_overrides or {})

        # 2. Context creation with initial data
        # FlextContext implements Ctx protocol structurally
        # Assign to p.Ctx type - mypy verifies protocol conformance
        runtime_context_typed: p.Ctx
        if context is not None:
            runtime_context_typed = context
        else:
            # FlextContext.create() returns FlextContext which implements p.Ctx
            runtime_context_typed = FlextContext.create()

        # 3. Container creation with registrations
        # runtime_config is FlextSettings which implements p.Config structurally
        # No cast needed - FlextSettings implements p.Config protocol
        runtime_config_typed: p.Config = runtime_config
        runtime_container = FlextContainer.create().scoped(
            config=runtime_config_typed,
            context=runtime_context_typed,
            subproject=subproject,
            services=services,
            factories=factories,
            resources=resources,
        )

        if container_overrides:
            runtime_container.configure(container_overrides)

        if wire_modules or wire_packages or wire_classes:
            runtime_container.wire_modules(
                modules=wire_modules,
                packages=wire_packages,
                classes=wire_classes,
            )

        # 4. Dispatcher creation with auto-discovery
        runtime_dispatcher = FlextDispatcher.create(auto_discover_handlers=True)

        # 5. Registry creation
        runtime_registry = FlextRegistry.create(auto_discover_handlers=True)

        return m.ServiceRuntime.model_construct(
            config=runtime_config,
            context=runtime_context_typed,
            container=runtime_container,
            dispatcher=runtime_dispatcher,  # type: ignore[arg-type]
            registry=runtime_registry,
        )

    @classmethod
    def _create_initial_runtime(cls) -> m.ServiceRuntime:
        """Build the initial runtime triple for a new service instance."""
        config_type = cls._get_service_config_type()
        options = cls._runtime_bootstrap_options()
        # Delegate to _create_runtime with options from _runtime_bootstrap_options
        # Extract values from options using u.mapper()
        config_type_raw = (
            u.mapper().get(options, "config_type") if "config_type" in options else None
        )
        # Type narrowing: Check if config_type_raw is a type subclass of FlextSettings
        config_type_val: type[FlextSettings] | None
        if (
            config_type_raw is not None
            and isinstance(config_type_raw, type)
            and issubclass(config_type_raw, FlextSettings)
        ):
            config_type_val = config_type_raw
        else:
            config_type_val = config_type

        config_overrides_raw = u.mapper().get(options, "config_overrides")
        # Type narrowing: Check if config_overrides_raw is a Mapping
        # isinstance() confirms type, use type: ignore for Mapping check
        if config_overrides_raw is not None and isinstance(
            config_overrides_raw, Mapping
        ):
            config_overrides_val: Mapping[str, t.FlexibleValue] | None = (
                config_overrides_raw  # type: ignore[assignment]
            )
        else:
            config_overrides_val = None

        context_raw = (
            u.mapper().get(options, "context") if "context" in options else None
        )
        # Type narrowing: context_raw should implement p.Ctx protocol if present
        # Since we can't check protocol conformance at runtime, use type: ignore
        if context_raw is not None:
            context_val: p.Ctx | None = context_raw  # type: ignore[assignment]
        else:
            context_val = None

        # Type narrowing: Check if subproject is a string
        subproject_raw = u.mapper().get(options, "subproject")
        subproject_val: str | None = (
            subproject_raw if isinstance(subproject_raw, str) else None
        )

        services_raw = options.get("services") if "services" in options else None
        # Cast needed: isinstance doesn't narrow to specific Mapping type signature
        services_val = (
            cast(
                "Mapping[str, t.GeneralValueType | BaseModel | p.VariadicCallable[t.GeneralValueType]] | None",
                services_raw,
            )
            if services_raw is not None and isinstance(services_raw, Mapping)
            else None
        )

        factories_raw = options.get("factories")
        # Cast needed: isinstance doesn't narrow to specific Mapping type signature
        factories_val = (
            cast(
                "Mapping[str, Callable[[], t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue]]] | None",
                factories_raw,
            )
            if factories_raw is not None and isinstance(factories_raw, Mapping)
            else None
        )

        resources_raw = u.mapper().get(options, "resources")
        # Cast needed: isinstance doesn't narrow to specific Mapping type signature
        resources_val = (
            cast(
                "Mapping[str, Callable[[], t.GeneralValueType]] | None",
                resources_raw,
            )
            if resources_raw is not None and isinstance(resources_raw, Mapping)
            else None
        )

        container_overrides_raw = u.mapper().get(options, "container_overrides")
        # Cast needed: isinstance doesn't narrow to specific Mapping type signature
        container_overrides_val = (
            cast(
                "Mapping[str, t.FlexibleValue] | None",
                container_overrides_raw,
            )
            if container_overrides_raw is not None
            and isinstance(container_overrides_raw, Mapping)
            else None
        )

        wire_modules_raw = u.mapper().get(options, "wire_modules")
        # Cast needed: isinstance doesn't narrow to specific Sequence type signature
        wire_modules_val = (
            cast(
                "Sequence[ModuleType] | None",
                wire_modules_raw,
            )
            if wire_modules_raw is not None and isinstance(wire_modules_raw, Sequence)
            else None
        )

        wire_packages_raw = u.mapper().get(options, "wire_packages")
        # Cast needed: isinstance doesn't narrow to specific Sequence type signature
        wire_packages_val = (
            cast(
                "Sequence[str] | None",
                wire_packages_raw,
            )
            if wire_packages_raw is not None and isinstance(wire_packages_raw, Sequence)
            else None
        )

        wire_classes_raw = u.mapper().get(options, "wire_classes")
        # Cast needed: isinstance doesn't narrow to specific Sequence type signature
        wire_classes_val = (
            cast(
                "Sequence[type] | None",
                wire_classes_raw,
            )
            if wire_classes_raw is not None and isinstance(wire_classes_raw, Sequence)
            else None
        )

        return cls._create_runtime(
            config_type=config_type_val,  # Already typed as type[FlextSettings] | None
            config_overrides=config_overrides_val,
            context=context_val,
            subproject=subproject_val,
            services=services_val,
            factories=factories_val,
            resources=resources_val,
            container_overrides=container_overrides_val,
            wire_modules=wire_modules_val,
            wire_packages=wire_packages_val,
            wire_classes=wire_classes_val,
        )

    @classmethod
    def _runtime_bootstrap_options(cls) -> t.RuntimeBootstrapOptions:
        """Hook for subclasses to parametrize runtime automation.

        Override to customize:
        - config_overrides: Dict of config values to override
        - services: Dict[str, object] to register as singletons
        - factories: Dict[str, Callable] to register as factories
        - resources: Dict[str, Callable] to register as resources
        - wire_modules: List of modules to wire for @inject
        - wire_packages: List of packages to wire
        - wire_classes: List of classes to wire

        Subclasses can override this method to pass keyword arguments directly
        to :meth:`_create_runtime`, enabling opt-in configuration overrides,
        scoped service registrations, factory/resource wiring, and container
        configuration changes without duplicating setup code in ``__init__``.

        Example:
            @classmethod
            def _runtime_bootstrap_options(cls):
                return {
                    "config_overrides": {"app_name": "MyApp"},
                    "services": {"db": my_db_service},
                }

        """
        return {}

    def _clone_runtime(
        self,
        *,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        context: p.Ctx | None = None,
        subproject: str | None = None,
        container_services: Mapping[str, t.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> m.ServiceRuntime:
        """Clone config/context and container in a single unified path."""
        config: FlextSettings = require_initialized(self._config, "Config")
        ctx: p.Ctx = require_initialized(self._context, "Context")
        container: p.DI = require_initialized(self._container, "Container")
        cloned_config = config.model_copy(
            update=config_overrides or {},
            deep=True,
        )
        # Clone context - Ctx implementations have clone() method
        runtime_context = context.clone() if context is not None else ctx.clone()
        # runtime_context implements Ctx structurally
        scoped_container = container.scoped(
            config=cloned_config,
            context=runtime_context,
            subproject=subproject,
            services=container_services,
            factories=container_factories,
        )

        # Create dispatcher with auto-discovery enabled
        runtime_dispatcher = FlextDispatcher.create(auto_discover_handlers=True)

        # Create registry with auto-discovery and deduplication enabled
        runtime_registry = FlextRegistry.create(auto_discover_handlers=True)

        return m.ServiceRuntime.model_construct(
            config=cloned_config,
            context=runtime_context,
            container=scoped_container,
            dispatcher=runtime_dispatcher,
            registry=runtime_registry,
        )

    @computed_field
    def runtime(
        self,
    ) -> m.ServiceRuntime:  # pragma: no cover - trivial access
        """View of the runtime triple for this service instance."""
        return require_initialized(self._runtime, "Runtime")

    @property
    def context(self) -> p.Ctx:
        """Service-scoped execution context."""
        return require_initialized(self._context, "Context")

    @property
    def config(self) -> FlextSettings:
        """Service-scoped configuration clone."""
        return require_initialized(self._config, "Config")

    @property
    def container(self) -> p.DI:
        """Container bound to the service context/config."""
        return require_initialized(self._container, "Container")

    @abstractmethod
    def execute(self) -> r[TDomainResult]:
        """Execute domain service logic.

        This is the core business logic method that must be implemented by all
        concrete service subclasses. It contains the actual domain operations,
        business rules, and result generation logic specific to each service.

        Business Rule: Executes the domain service business logic and returns
        a FlextResult indicating success or failure. This method is the primary
        entry point for all domain service operations in the FLEXT ecosystem.
        All business logic, domain rules, and operational workflows are executed
        through this method. The method must follow railway-oriented programming
        principles, returning ``r[TDomainResult]`` instead of raising exceptions,
        ensuring predictable error handling and composable service pipelines.

        Audit Implication: Service execution is a critical audit event that
        represents the execution of business logic and domain operations. All
        service executions should be logged with appropriate context and
        correlation IDs. The FlextResult return type ensures audit trail
        completeness by tracking both successful operations (with domain results)
        and failed operations (with error details and error codes). Audit logs
        should capture service identity, execution context, input parameters,
        execution duration, and result status.

        The method should follow railway-oriented programming principles,
        returning ``r[T]`` for consistent error handling and success
        indication.

        Returns:
            r[TDomainResult]: Success with domain result or failure
                with error details

        Note:
            Implementations should focus on business logic only. Infrastructure
            concerns like validation and error handling are handled by the base class.

        Raises:
            None: Uses FlextResult for error handling instead of exceptions.

        """
        ...

    def validate_business_rules(self) -> r[bool]:
        """Validate business rules with extensible validation pipeline.

        Base method for business rule validation that can be overridden by subclasses
        to implement custom validation logic. By default, returns success. Subclasses
        should extend this method to add domain-specific business rule validation.

        Note: Cannot be @staticmethod because subclasses override this method and
        may need to access instance state. The base implementation doesn't use self,
        but the method signature must remain an instance method for polymorphism.

        Returns:
            r[bool]: Success (True) if all business rules pass, failure with error details

        Example:
            >>> class ValidatedService(s[Data]):
            ...     def validate_business_rules(self) -> r[bool]:
            ...         if not self.has_required_data():
            ...             return r[bool].fail("Missing required data")
            ...         return r[bool].ok(True)

        """
        # Base implementation - accept all (no validation)
        # Subclasses should override for specific business rules
        return r[bool].ok(True)

    def is_valid(self) -> bool:
        """Check if service is in valid state for execution.

        Performs business rule validation and returns boolean result.
        Catches exceptions during validation to ensure safe state checking.
        Used by infrastructure components to determine if service can execute.

        Returns:
            bool: True if service is valid and ready for execution

        Example:
            >>> service = MyService()
            >>> if service.is_valid():
            ...     result = service.execute()

        """
        try:
            return self.validate_business_rules().is_success
        except Exception:
            # Validation failed due to exception - consider invalid
            return False

    def get_service_info(self) -> Mapping[str, t.FlexibleValue]:
        """Get service metadata and configuration information.

        Returns comprehensive metadata about the service instance including
        type information and execution parameters.
        Used by monitoring, logging, and debugging infrastructure.

        Returns:
            Mapping[str, t.FlexibleValue]: Service metadata dictionary containing:
                - service_type: Class name of the service
                - Additional metadata can be added by subclasses

        Example:
            >>> service = MyService()
            >>> info = service.get_service_info()
            >>> print(f"Service: {info['service_type']}")

        """
        return {
            "service_type": self.__class__.__name__,
        }

    @computed_field
    def access(self) -> _ServiceAccess:
        """Unified access to FLEXT infrastructure components.

        Provides on-demand access to CQRS models, registry, configuration,
        result helpers, context, and container utilities through a single
        gateway attached to every service instance. The access facade is
        intentionally lightweight and lazily constructed to avoid importing
        individual modules across handlers or other services.

        Returns:
            _ServiceAccess: Facade exposing commonly used FLEXT utilities

        Example:
            >>> service = MyService()
            >>> registry = service.access.registry
            >>> nested = service.access.clone_config(app_name="test")

        """
        # Direct access - _ServiceAccess accepts any FlextService instance
        # Type narrowing: self is FlextService[TDomainResult], which is compatible with
        # FlextService[t.GeneralValueType] because t.GeneralValueType covers all domain results
        # Cast is intentional: TypeVar variance issue - TDomainResult is not covariant
        service_typed: FlextService[t.GeneralValueType] = cast(
            "FlextService[t.GeneralValueType]",
            self,
        )
        # _ServiceAccess(service_typed) already returns _ServiceAccess instance
        return _ServiceAccess(service_typed)

    @classmethod
    def from_parent(
        cls,
        parent: FlextService[t.GeneralValueType],
        *,
        config_overrides: dict[str, object] | None = None,
        context_overrides: dict[str, object] | None = None,
    ) -> Self:
        """Create child service inheriting parent's infrastructure.

        Enables service composition by creating a new service instance that
        inherits the parent's config, context, and container with optional
        overrides. Useful for orchestrating multiple services in a workflow.

        Args:
            parent: Parent service to inherit from.
            config_overrides: Optional configuration overrides.
            context_overrides: Optional context data overrides.

        Returns:
            T: New service instance inheriting parent's infrastructure.

        Example:
            >>> parent = OrderService()
            >>> child = PaymentService.from_parent(
            ...     parent,
            ...     config_overrides={"timeout": 30},
            ... )
            >>> # child inherits parent's container, context, etc.

        """
        # Build config with overrides
        parent_config = parent.config
        if config_overrides:
            merged_config = parent_config.model_copy(update=config_overrides)
        else:
            merged_config = parent_config

        # Build context with overrides
        parent_context = parent.context
        if context_overrides and isinstance(parent_context, FlextContext):
            # Type narrowing: parent_context is FlextContext, which has model_copy
            # isinstance check already narrows type to FlextContext
            # FlextContext is a Pydantic model (inherits from FlextRuntime which inherits from BaseModel)
            flext_context_var = parent_context
            # Use getattr for model_copy to avoid type checker issues with protocol intersection
            model_copy_method = getattr(flext_context_var, "model_copy", None)
            if model_copy_method is not None:
                merged_context_pydantic = model_copy_method(update=context_overrides)
            else:
                # Fallback if model_copy not available
                merged_context_pydantic = flext_context_var
            # merged_context_pydantic is FlextContext which implements p.Ctx structurally
            merged_context: p.Ctx = merged_context_pydantic
        else:
            merged_context = parent_context

        # Create child with inherited infrastructure
        # Type narrowing: cls is FlextService subclass, merged_config is FlextSettings
        # Pass as **kwargs since __init__ accepts **data: t.GeneralValueType
        # Cast is intentional: Infrastructure objects (FlextSettings, FlextContext, FlextContainer)
        # are not strictly t.GeneralValueType, but __init__ accepts them via protocol dispatch
        merged_data: dict[str, t.GeneralValueType | object] = {
            "config": merged_config,
            "context": merged_context,
            "container": parent.container,
        }
        return cls(**cast("dict[str, t.GeneralValueType]", merged_data))


class _ServiceExecutionScope(FlextModelsBase.ArbitraryTypesModel):
    """Immutable view of nested execution resources for a service."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cqrs: type
    registry: FlextRegistry
    config: p.Config
    result: type
    context: p.Ctx
    runtime: m.ServiceRuntime
    service_data: t.ConfigurationMapping


class _ServiceAccess(m.ArbitraryTypesModel):
    """Gateway for service-level infrastructure access and cloning.

    Centralizes access to core FLEXT components (CQRS models, registry,
    configuration, result helpers, context, and container) and supports
    cloning of configuration plus nested execution scopes with isolated

    Note: Accesses protected attributes of FlextService (_config, _runtime, etc.)
    This is intentional as _ServiceAccess is an internal helper class.
    context.
    """

    # ConfigDict is a TypedDict - use explicit type annotation to help type checker
    # ConfigDict works correctly at runtime, type annotation helps with overload resolution
    # Match ArbitraryTypesModel pattern: direct assignment, not ClassVar
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # Suppress Pydantic warnings for runtime field during serialization
        # Runtime may be ModelPrivateAttr if not initialized, which causes warnings
        # This is expected behavior - runtime is only available after create_service_runtime()
        # Note: repr parameter removed - not supported in Pydantic v2 ConfigDict
    )

    # Use class attributes (not PrivateAttr) for consistency with FlextService
    _registry: FlextRegistry | None = None
    # s is defined at end of file, use FlextService directly to avoid forward reference issue
    _service: FlextService[t.GeneralValueType]

    def __init__(self, service: FlextService[t.GeneralValueType]) -> None:
        super().__init__()
        # Accept any FlextService instance - t.GeneralValueType covers all domain results
        # Set attributes directly (no PrivateAttr needed)
        # service is never None after __init__, so we can use non-optional type
        self._service = service
        self._registry = FlextRegistry()

    @computed_field
    def cqrs(self) -> type:
        """CQRS facade from m.

        Note: Cannot be @staticmethod because @computed_field requires instance method
        signature for Pydantic field computation, even if self is not used.
        """
        return m.Cqrs

    @computed_field
    def registry(self) -> FlextRegistry:
        """Registry singleton for service discovery."""
        return require_initialized(self._registry, "Registry")

    @computed_field
    def config(self) -> p.Config:
        """Global configuration instance for the service."""
        # Use public property instead of private attribute
        return self._service.config

    @computed_field
    def runtime(
        self,
    ) -> m.ServiceRuntime:  # pragma: no cover - trivial access
        """Protocol-backed runtime triple for the bound service."""
        # Check if runtime is initialized before accessing
        # During Pydantic serialization, _runtime may be ModelPrivateAttr
        # Access _runtime directly to check initialization status
        runtime_attr = getattr(self._service, "_runtime", None)
        if runtime_attr is None or isinstance(runtime_attr, ModelPrivateAttr):
            # Runtime not initialized - raise error to prevent Pydantic from serializing ModelPrivateAttr
            # This prevents the warning about unexpected value type
            msg = "Runtime not initialized. Call create_service_runtime() first."
            raise RuntimeError(msg)
        # Use require_initialized to get runtime (same as FlextService.runtime)
        # This ensures consistent behavior and proper error handling
        # Type narrowing: runtime_attr is m.ServiceRuntime after require_initialized
        result = require_initialized(runtime_attr, "Runtime")
        # Type narrowing: result is m.ServiceRuntime (runtime_attr type after require_initialized)
        if not isinstance(result, m.ServiceRuntime):
            msg = f"Expected m.ServiceRuntime, got {type(result).__name__}"
            raise TypeError(msg)
        return result

    @computed_field
    def result(
        self,
    ) -> type:  # pragma: no cover
        """Result factory shortcuts.

        Note: Decorator @computed_field requires instance method but value is static.
        Using minimal self reference to satisfy PLR6301.
        """
        _ = type(self)  # Required for @computed_field compliance
        return r

    @computed_field
    def context(self) -> p.Ctx:
        """Context manager for correlation and tracing."""
        # Use public property instead of private attribute
        return self._service.context

    @computed_field
    def container(self) -> p.DI:
        """Dependency injection container bound to the service."""
        # Use public property instead of private attribute
        return self._service.container

    @computed_field
    def logger(self) -> FlextLogger:
        """Scoped logger for this service.

        Returns a logger instance bound to the service's container and context,
        enabling container-scoped logging configuration and context propagation.

        Returns:
            FlextLogger: Logger instance bound to service container.

        """
        # Create logger scoped to service module
        service_name = self._service.__class__.__name__
        return FlextLogger.create_module_logger(
            f"{self._service.__class__.__module__}.{service_name}",
        )

    @property
    def metrics(self) -> FlextDispatcher:
        """Metrics collector for this service.

        Returns the MetricsTracker from the service's dispatcher runtime,
        enabling metrics collection for service operations.

        Returns:
            FlextDispatcher: Dispatcher instance which inherits from MetricsTracker mixin.

        Note:
            The dispatcher inherits from FlextMixins.CQRS.MetricsTracker, so it
            can be used directly for metrics collection.

        """
        # Access dispatcher from runtime through service
        # Type narrowing: Check runtime initialization status
        runtime_attr = getattr(self._service, "_runtime", None)
        if runtime_attr is None or isinstance(runtime_attr, ModelPrivateAttr):
            msg = "Runtime not initialized. Call create_service_runtime() first."
            raise TypeError(msg)
        # Type narrowing: runtime_attr is m.ServiceRuntime after check
        # runtime.dispatcher is p.CommandBus protocol, but we need FlextDispatcher for metrics
        # Cast needed: Protocol doesn't expose metrics API
        return cast("FlextDispatcher", runtime_attr.dispatcher)
        # Dispatcher inherits from FlextMixins.CQRS.MetricsTracker

    @property
    def rate_limiter(self) -> RateLimiterManager:
        """Rate limiter manager for this service.

        Returns the RateLimiterManager from the service's dispatcher runtime,
        enabling rate limiting for service operations.

        Returns:
            RateLimiterManager: Rate limiter manager instance from dispatcher.

        """
        # Access dispatcher from runtime through service
        # Type narrowing: Check runtime initialization status
        runtime_attr = getattr(self._service, "_runtime", None)
        if runtime_attr is None or isinstance(runtime_attr, ModelPrivateAttr):
            msg = "Runtime not initialized. Call create_service_runtime() first."
            raise TypeError(msg)
        # Type narrowing: runtime_attr is m.ServiceRuntime after check
        # runtime.dispatcher is p.CommandBus protocol, but we need FlextDispatcher for rate_limiter
        # Cast needed: Protocol doesn't expose _rate_limiter private attribute
        dispatcher = cast("FlextDispatcher", runtime_attr.dispatcher)
        # Access private attribute (dispatcher._rate_limiter is private)
        # Type assertion: dispatcher has _rate_limiter attribute
        return dispatcher._rate_limiter

    @property
    def circuit_breaker(self) -> CircuitBreakerManager:
        """Circuit breaker manager for this service.

        Returns the CircuitBreakerManager from the service's dispatcher runtime,
        enabling circuit breaking for service operations.

        Returns:
            CircuitBreakerManager: Circuit breaker manager instance from dispatcher.

        """
        # Access dispatcher from runtime through service
        # Type narrowing: Check runtime initialization status
        runtime_attr = getattr(self._service, "_runtime", None)
        if runtime_attr is None or isinstance(runtime_attr, ModelPrivateAttr):
            msg = "Runtime not initialized. Call create_service_runtime() first."
            raise TypeError(msg)
        # Type narrowing: runtime_attr is m.ServiceRuntime after check
        # runtime.dispatcher is p.CommandBus protocol, but we need FlextDispatcher for circuit_breaker
        # Cast needed: Protocol doesn't expose _circuit_breaker private attribute
        dispatcher = cast("FlextDispatcher", runtime_attr.dispatcher)
        # Access private attribute (dispatcher._circuit_breaker is private)
        # Type assertion: dispatcher has _circuit_breaker attribute
        return dispatcher._circuit_breaker

    def container_scope(
        self,
        *,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        context: p.Ctx | None = None,
        subproject: str | None = None,
        services: Mapping[str, t.FlexibleValue] | None = None,
        factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> p.DI:
        """Create a container scope with cloned config and optional overrides."""
        return self.runtime_scope(
            config_overrides=config_overrides,
            context=context,
            subproject=subproject,
            container_services=services,
            container_factories=factories,
        ).container

    def runtime_scope(
        self,
        *,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        context: p.Ctx | None = None,
        subproject: str | None = None,
        services: Mapping[str, t.FlexibleValue] | None = None,
        factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
        container_services: Mapping[str, t.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> m.ServiceRuntime:
        """Clone the service runtime triple using protocol-backed models."""
        # Cast needed: mypy sees @property/@computed_field as Callable, not the return type
        # These are workarounds for mypy's property inference limitation
        config = cast("p.Config", self.config)
        ctx = cast("p.Ctx", self.context)
        container = cast("p.DI", self.container)

        # Clone config with overrides using Pydantic's model_copy
        cloned_config = config.model_copy(
            update=config_overrides or {},
            deep=True,
        )

        # Clone context - Ctx implementations have clone() method
        runtime_context = context.clone() if context is not None else ctx.clone()

        # Create scoped container using public API
        scoped_container = container.scoped(
            config=cloned_config,
            context=runtime_context,
            subproject=subproject,
            services=container_services or services,
            factories=container_factories or factories,
        )

        # Construct ServiceRuntime using model_construct
        return m.ServiceRuntime.model_construct(
            config=cloned_config,
            context=runtime_context,
            container=scoped_container,
        )

    def clone_config(self, **overrides: t.FlexibleValue) -> p.Config:
        """Create a deep copy of the service configuration with overrides.

        Args:
            **overrides: Field overrides applied to the cloned configuration.

        Returns:
            Config: Cloned configuration instance with updates applied.

        """
        # Cast needed: mypy sees @property as Callable, not the return type
        config = cast("p.Config", self.config)
        return config.model_copy(update=overrides, deep=True)

    @contextmanager
    def nested_execution(
        self,
        *,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        service_name: str | None = None,
        version: str | None = None,
        correlation_id: str | None = None,
        container_services: Mapping[str, t.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> Iterator[_ServiceExecutionScope]:
        """Create a nested execution scope with cloned config and context.

        Yields a frozen view of the cloned configuration, registry, container,
        context, and CQRS facade. The nested context is correlation-aware and
        isolated from the parent, enabling containerized execution flows without
        mutating global state.
        """
        # Cast needed: mypy sees @computed_field as Callable, not the return type
        base_runtime = cast("m.ServiceRuntime", self.runtime)
        # Type narrowing: base_runtime.context is FlextContext for nested class access
        # Cast needed: protocol (p.Ctx) doesn't expose nested classes (.Correlation, .Service)
        base_context = cast("FlextContext", base_runtime.context)
        original_correlation = base_context.Correlation.get_correlation_id()

        # base_context implements Ctx structurally - cast to protocol type
        runtime = self.runtime_scope(
            config_overrides=config_overrides,
            context=cast("p.Ctx | None", base_context),
            container_services=container_services,
            container_factories=container_factories,
        )

        # Type narrowing: runtime.context is FlextContext
        runtime_context = cast("FlextContext", runtime.context)
        if correlation_id:
            runtime_context.Correlation.set_correlation_id(correlation_id)
        else:
            _ = runtime_context.Utilities.ensure_correlation_id()

        service_data: t.ConfigurationMapping = {
            "service_type": self._service.__class__.__name__,
            "payload": {},
        }

        scope = _ServiceExecutionScope.model_construct(
            cqrs=self.cqrs,
            registry=self.registry,
            result=self.result,
            runtime=runtime,
            service_data=service_data,
        )

        with runtime_context.Service.service_context(
            service_name or self._service.__class__.__name__,
            version or runtime.config.version,
        ):
            yield scope

        if original_correlation is None:
            base_context.Correlation.reset_correlation_id()
        else:
            base_context.Correlation.set_correlation_id(original_correlation)


s = FlextService

__all__ = ["FlextService", "s"]
