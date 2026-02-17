"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services. It relies on structural typing to satisfy
``p.Service`` and provides a clean service lifecycle.

Note: CQRS components (FlextDispatcher, FlextRegistry) should be used directly,
not through FlextService. This keeps FlextService focused on service concerns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import override

from pydantic import (
    ConfigDict,
    PrivateAttr,
    computed_field,
)

from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.mixins import FlextMixins as x
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.settings import FlextSettings
from flext_core.typings import t
from flext_core.utilities import u


class FlextService[TDomainResult](
    m.ArbitraryTypesModel,
    x,
    ABC,
):
    """Base class for domain services in FLEXT applications.

    Subclasses implement ``execute`` to run business logic and return
    ``FlextResult`` values. The base inherits :class:`FlextMixins` (which extends
    :class:`FlextRuntime`) so services can reuse runtime automation for creating
    scoped config/context/container triples via :meth:`create_service_runtime`
    while remaining protocol compliant via structural typing.

    Example:
        class UserService(FlextService[User]):
            def execute(self) -> r[User]:
                return r.ok(User(id="123", name="Alice"))

        # Usage
        service = UserService()
        result = service.execute()
        if result.is_success:
            user = result.value

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

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
        # Type narrowing: runtime.config is "p.Config", but we need FlextSettings
        # All implementations of "p.Config" in FLEXT are FlextSettings or subclasses
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
    _context: p.Context | None = PrivateAttr(default=None)
    _config: FlextSettings | None = PrivateAttr(default=None)
    _container: p.DI | None = PrivateAttr(default=None)
    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)
    _discovered_handlers: list[tuple[str, m.Handler.DecoratorConfig]] = PrivateAttr(
        default_factory=list,
    )

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
        context: p.Context | None = None,
        subproject: str | None = None,
        services: Mapping[
            str,
            t.GeneralValueType
            | p.Config
            | p.Context
            | p.DI
            | p.Service[t.GeneralValueType]
            | p.Log
            | p.Handler
            | p.Registry
            | Callable[..., t.GeneralValueType],
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
        # context parameter is p.ContextLike (minimal protocol)
        # Use TypeGuard to narrow to "p.Context" (full protocol with set method)
        runtime_context_typed: p.Context
        if context is not None and u.is_context(context):
            # TypeGuard narrowed to "p.Context" - use directly
            runtime_context_typed = context
        else:
            # Minimal ContextLike or None - create full context
            runtime_context_typed = FlextContext.create()

        # 3. Container creation with registrations
        # runtime_config is FlextSettings which implements "p.Config" structurally
        # No cast needed - FlextSettings implements "p.Config" protocol
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

        return m.ServiceRuntime.model_construct(
            config=runtime_config,
            context=runtime_context_typed,
            container=runtime_container,
        )

    @classmethod
    def _create_initial_runtime(cls) -> m.ServiceRuntime:
        """Build the initial runtime triple for a new service instance."""
        config_type = cls._get_service_config_type()
        options = cls._normalize_runtime_bootstrap_options(
            cls._runtime_bootstrap_options(),
        )

        config_type_raw = options.config_type
        config_type_val: type[FlextSettings] | None
        if config_type_raw is not None and issubclass(config_type_raw, FlextSettings):
            config_type_val = config_type_raw
        else:
            config_type_val = config_type
        context_val_raw = options.context
        context_val: p.Context | None = (
            context_val_raw if isinstance(context_val_raw, p.Context) else None
        )

        return cls._create_runtime(
            config_type=config_type_val,
            config_overrides=options.config_overrides,
            context=context_val,
            subproject=options.subproject,
            services=cls._normalize_scoped_services(options.services),
            factories=options.factories,
            resources=options.resources,
            container_overrides=options.container_overrides,
            wire_modules=options.wire_modules,
            wire_packages=options.wire_packages,
            wire_classes=options.wire_classes,
        )

    @classmethod
    def _normalize_runtime_bootstrap_options(
        cls,
        options_raw: p.RuntimeBootstrapOptions | Mapping[str, t.FlexibleValue],
    ) -> p.RuntimeBootstrapOptions:
        del cls
        if isinstance(options_raw, p.RuntimeBootstrapOptions):
            return options_raw
        if isinstance(options_raw, Mapping):
            return p.RuntimeBootstrapOptions.model_validate(
                {k: v for k, v in options_raw.items() if isinstance(k, str)},
            )
        return p.RuntimeBootstrapOptions()

    @classmethod
    def _normalize_scoped_services(
        cls,
        services: Mapping[str, t.RegisterableService] | None,
    ) -> (
        Mapping[
            str,
            t.GeneralValueType
            | p.Config
            | p.Context
            | p.DI
            | p.Service[t.GeneralValueType]
            | p.Log
            | p.Handler
            | p.Registry
            | Callable[..., t.GeneralValueType],
        ]
        | None
    ):
        del cls
        if services is None:
            return None

        normalized: dict[
            str,
            t.GeneralValueType
            | p.Config
            | p.Context
            | p.DI
            | p.Service[t.GeneralValueType]
            | p.Log
            | p.Handler
            | p.Registry
            | Callable[..., t.GeneralValueType],
        ] = {
            name: service
            for name, service in services.items()
            if FlextService._is_scoped_service_candidate(service)
        }

        return normalized or None

    @staticmethod
    def _is_scoped_service_candidate(service: t.RegisterableService) -> bool:
        if u.is_general_value_type(service):
            return True
        protocol_types = (
            p.Config,
            p.Context,
            p.DI,
            p.Service,
            p.Log,
            p.Handler,
            p.Registry,
        )
        return isinstance(service, protocol_types) or callable(service)

    @classmethod
    def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
        """Hook for subclasses to parametrize runtime automation.

        Override to customize:
        - config_overrides: Dict of config values to override
        - services: dict[str, object] to register as singletons
        - factories: dict[str, Callable] to register as factories
        - resources: dict[str, Callable] to register as resources
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
                return p.RuntimeBootstrapOptions(
                    config_overrides={"app_name": "MyApp"},
                    services={"db": my_db_service},
                )

        """
        return p.RuntimeBootstrapOptions()

    def _clone_runtime(
        self,
        *,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        context: p.Context | None = None,
        subproject: str | None = None,
        container_services: Mapping[str, t.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> m.ServiceRuntime:
        """Clone config/context and container in a single unified path."""
        config: FlextSettings = u.require_initialized(self._config, "Config")
        ctx: p.Context = u.require_initialized(self._context, "Context")
        container: p.DI = u.require_initialized(self._container, "Container")
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

        return m.ServiceRuntime.model_construct(
            config=cloned_config,
            context=runtime_context,
            container=scoped_container,
        )

    @computed_field
    def runtime(
        self,
    ) -> m.ServiceRuntime:  # pragma: no cover - trivial access
        """View of the runtime triple for this service instance."""
        return u.require_initialized(self._runtime, "Runtime")

    @property
    def context(self) -> p.Context:
        """Service-scoped execution context."""
        return u.require_initialized(self._context, "Context")

    @property
    def config(self) -> p.Config:
        """Service-scoped configuration clone."""
        return u.require_initialized(self._config, "Config")

    @property
    def container(self) -> p.DI:
        """Container bound to the service context/config."""
        return u.require_initialized(self._container, "Container")

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


s = FlextService

__all__ = ["FlextService", "s"]
