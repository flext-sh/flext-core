"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services that participate in CQRS flows. It relies
on structural typing to satisfy ``p.Domain.Service`` and aligns with the
dispatcher-centric architecture.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Self, cast

from pydantic import ConfigDict, PrivateAttr, computed_field

from flext_core.config import FlextConfig  # For instantiation only
from flext_core.constants import c
from flext_core.container import FlextContainer  # For instantiation only
from flext_core.context import FlextContext  # For instantiation only
from flext_core.exceptions import e
from flext_core.mixins import require_initialized, x
from flext_core.models import m
from flext_core.protocols import p
from flext_core.registry import FlextRegistry
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextService[TDomainResult](
    m.ArbitraryTypesModel,
    x,
    ABC,
):
    """Base class for domain services used in CQRS flows.

    Subclasses implement ``execute`` to run business logic and return
    ``FlextResult`` values. The base provides validation hooks, dependency
    injection, and context-aware logging while remaining protocol compliant via
    structural typing.
    """

    model_config = ConfigDict(
        frozen=True,
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
            result_raw = execute_method()
            if not isinstance(result_raw, r):
                msg = f"execute() must return r, got {type(result_raw).__name__}"
                raise TypeError(msg)
            result: r[TDomainResult] = cast("r[TDomainResult]", result_raw)
            # For auto_execute=True, return the result value directly (V2 Auto pattern)
            # This allows: user = AutoGetUserService(user_id="123") to get User object
            if result.is_failure:
                error_msg = result.error or "Service execution failed"
                raise e.BaseError(error_msg)
            # Return the unwrapped value directly (breaks static typing but is intended behavior)
            # Type cast is necessary because __new__ signature expects Self
            return cast("Self", result.value)
        # For auto_execute=False, return instance (normal pattern)
        type(instance).__init__(instance, **kwargs)
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
        raise e.BaseError(result.error or "Service execution failed")

    def __init__(
        self,
        **data: t.GeneralValueType,
    ) -> None:
        """Initialize service with configuration data.

        Sets up the service instance with optional configuration parameters
        passed through **data. Delegates to parent classes for proper
        initialization of mixins, models, and infrastructure components.

        Args:
            **data: Configuration parameters for service initialization

        """
        runtime = self._create_initial_runtime()
        # runtime.context is FlextContext - cast to access nested classes
        context = cast("FlextContext", runtime.context)

        with context.Service.service_context(
            self.__class__.__name__,
            runtime.config.version,
        ):
            super().__init__(**data)

        object.__setattr__(self, "_context", runtime.context)
        object.__setattr__(self, "_config", runtime.config)
        object.__setattr__(self, "_container", runtime.container)
        object.__setattr__(self, "_runtime", runtime)

    _context: p.Context.Ctx | None = PrivateAttr(default=None)
    _config: FlextConfig | None = PrivateAttr(default=None)
    _container: p.Container.DI | None = PrivateAttr(default=None)
    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)
    _auto_result: TDomainResult | None = PrivateAttr(default=None)

    @classmethod
    def _get_service_config_type(cls) -> type[FlextConfig]:
        """Get the config type for this service class.

        Services can override this method to specify their specific config type.
        Defaults to FlextConfig for generic services.

        Returns:
            type[FlextConfig]: The config class to use for this service

        """
        return FlextConfig  # Runtime return needs concrete class

    @classmethod
    def _create_initial_runtime(cls) -> m.ServiceRuntime:
        """Build the initial runtime triple for a new service instance."""
        base_context = FlextContext()
        # Get the service-specific config type
        config_type = cls._get_service_config_type()

        # If service specifies a specific config type, use it directly
        # Otherwise, clone from global FlextConfig instance
        if config_type is not FlextConfig:
            # Service has specific config type - create instance directly
            base_config = config_type()
        else:
            # Generic service - clone from global config
            global_config = FlextConfig.get_global_instance()

            class _ClonedConfig(FlextConfig):
                """Lightweight clone to bypass FlextConfig singleton semantics."""

                def __new__(
                    cls,
                    **_data: t.GeneralValueType,
                ) -> Self:  # pragma: no cover - construction hook
                    # Create raw instance using type-safe factory helper
                    return FlextRuntime.create_instance(cls)

            base_config = _ClonedConfig.model_construct(**global_config.model_dump())

        # base_context is FlextContext which implements Context.Ctx structurally
        # base_config is already FlextConfig (either config_type() or _ClonedConfig)
        base_container = FlextContainer().scoped(
            config=base_config,
            context=base_context,
        )

        return m.ServiceRuntime.model_construct(
            config=base_config,
            context=base_context,
            container=base_container,
        )

    def _clone_runtime(
        self,
        *,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        context: p.Context.Ctx | None = None,
        subproject: str | None = None,
        container_services: Mapping[str, t.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> m.ServiceRuntime:
        """Clone config/context and container in a single unified path."""
        config = require_initialized(self._config, "Config")
        ctx = require_initialized(self._context, "Context")
        container = require_initialized(self._container, "Container")
        cloned_config = config.model_copy(
            update=config_overrides or {},
            deep=True,
        )
        # Clone context - Context.Ctx implementations have clone() method
        runtime_context = context.clone() if context is not None else ctx.clone()
        # runtime_context implements Context.Ctx structurally
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
        return require_initialized(self._runtime, "Runtime")

    @property
    def context(self) -> p.Context.Ctx:
        """Service-scoped execution context."""
        return require_initialized(self._context, "Context")

    @property
    def config(self) -> FlextConfig:
        """Service-scoped configuration clone."""
        return require_initialized(self._config, "Config")

    @property
    def container(self) -> p.Container.DI:
        """Container bound to the service context/config."""
        return require_initialized(self._container, "Container")

    @abstractmethod
    def execute(self) -> r[TDomainResult]:
        """Execute domain service logic.

        This is the core business logic method that must be implemented by all
        concrete service subclasses. It contains the actual domain operations,
        business rules, and result generation logic specific to each service.

        The method should follow railway-oriented programming principles,
        returning ``r[T]`` for consistent error handling and success
        indication.

        Returns:
            r[TDomainResult]: Success with domain result or failure
                with error details

        Note:
            Implementations should focus on business logic only. Infrastructure
            concerns like validation and error handling are handled by the base class.

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
        # Direct access - GeneralValueType covers all domain results
        # Use cast to allow any FlextService[TDomainResult] to be treated as FlextService[GeneralValueType]
        return _ServiceAccess(cast("s[t.GeneralValueType]", self))


class _ServiceExecutionScope(m.ArbitraryTypesModel):
    """Immutable view of nested execution resources for a service."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    cqrs: type[m.Cqrs]
    registry: FlextRegistry
    config: p.Configuration.Config
    result: type
    context: p.Context.Ctx
    runtime: m.ServiceRuntime
    service_data: t.Types.ConfigurationMapping


class _ServiceAccess(m.ArbitraryTypesModel):
    """Gateway for service-level infrastructure access and cloning.

    Centralizes access to core FLEXT components (CQRS models, registry,
    configuration, result helpers, context, and container) and supports
    cloning of configuration plus nested execution scopes with isolated
    context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    _registry: FlextRegistry | None = PrivateAttr(default=None)
    _service: s[t.GeneralValueType] = PrivateAttr()

    def __init__(self, service: s[t.GeneralValueType]) -> None:
        super().__init__()
        # Accept any FlextService instance - GeneralValueType covers all domain results
        # Use object.__setattr__ for frozen model (Pydantic frozen=True requires this pattern)
        object.__setattr__(self, "_service", service)
        object.__setattr__(self, "_registry", FlextRegistry())

    @computed_field
    def cqrs(self) -> type[m.Cqrs]:
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
    def config(self) -> p.Configuration.Config:
        """Global configuration instance for the service."""
        return require_initialized(self._service._config, "Config")

    @computed_field
    def runtime(
        self,
    ) -> m.ServiceRuntime:  # pragma: no cover - trivial access
        """Protocol-backed runtime triple for the bound service."""
        return require_initialized(self._service._runtime, "Runtime")

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
    def context(self) -> p.Context.Ctx:
        """Context manager for correlation and tracing."""
        return require_initialized(self._service._context, "Context")

    @computed_field
    def container(self) -> p.Container.DI:
        """Dependency injection container bound to the service."""
        return require_initialized(self._service._container, "Container")

    def container_scope(
        self,
        *,
        config_overrides: Mapping[str, t.FlexibleValue] | None = None,
        context: p.Context.Ctx | None = None,
        subproject: str | None = None,
        services: Mapping[str, t.FlexibleValue] | None = None,
        factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> p.Container.DI:
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
        context: p.Context.Ctx | None = None,
        subproject: str | None = None,
        services: Mapping[str, t.FlexibleValue] | None = None,
        factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
        container_services: Mapping[str, t.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = None,
    ) -> m.ServiceRuntime:
        """Clone the service runtime triple using protocol-backed models."""
        # Access private attribute directly to avoid computed_field type issues
        # _clone_runtime accepts Context.Ctx - use directly
        context_for_clone: p.Context.Ctx | None = (
            context if context is not None else self._service._context
        )
        return self._service._clone_runtime(
            config_overrides=config_overrides,
            context=context_for_clone,
            subproject=subproject,
            container_services=container_services or services,
            container_factories=container_factories or factories,
        )

    def clone_config(self, **overrides: t.FlexibleValue) -> p.Configuration.Config:
        """Create a deep copy of the service configuration with overrides.

        Args:
            **overrides: Field overrides applied to the cloned configuration.

        Returns:
            Configuration.Config: Cloned configuration instance with updates applied.

        """
        config = require_initialized(self._service._config, "Config")
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
        base_runtime = require_initialized(self._service._runtime, "Runtime")
        # Context is guaranteed to be non-None in ServiceRuntime
        # Cast to FlextContext to access nested classes (Correlation, Service, Utilities)
        # base_runtime.context is FlextContext - cast to access nested classes
        base_context = cast("FlextContext", base_runtime.context)
        original_correlation = base_context.Correlation.get_correlation_id()

        # base_context implements Context.Ctx structurally - use directly
        runtime = self.runtime_scope(
            config_overrides=config_overrides,
            context=base_context,
            container_services=container_services,
            container_factories=container_factories,
        )

        # Cast to FlextContext to access nested classes
        # runtime.context is FlextContext - cast to access nested classes
        runtime_context = cast("FlextContext", runtime.context)
        if correlation_id:
            runtime_context.Correlation.set_correlation_id(correlation_id)
        else:
            _ = runtime_context.Utilities.ensure_correlation_id()

        service_data: t.Types.ConfigurationMapping = {
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
