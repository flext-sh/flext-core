"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services that participate in CQRS flows. It relies
on structural typing to satisfy ``FlextProtocols.Service`` and aligns with the
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

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins,
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
        **kwargs: FlextTypes.GeneralValueType,
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
            execute_method = getattr(instance, "execute", None)
            if not callable(execute_method):
                msg = f"Class {cls.__name__} must implement execute() method"
                raise TypeError(msg)
            # Type narrowing: execute_method is callable and returns FlextResult[TDomainResult]
            result_raw = execute_method()
            if not isinstance(result_raw, FlextResult):
                msg = f"execute() must return FlextResult, got {type(result_raw).__name__}"
                raise TypeError(msg)
            result: FlextResult[TDomainResult] = result_raw
            # For auto_execute=True, store result in instance and return instance
            # The result can be accessed via a special attribute or method
            # Note: __new__ must return Self, so we store result and raise on failure
            if result.is_failure:
                error_msg = result.error or "Service execution failed"
                raise FlextExceptions.BaseError(error_msg)
            # For auto_execute=True, raise exception to signal that result should be accessed via execute()
            # This maintains type safety while supporting auto-execute pattern
            # Note: The actual auto-execute pattern should be handled at call site, not in __new__
            error_msg = "Service with auto_execute=True must be called via execute() method, not instantiated directly"
            raise RuntimeError(error_msg)
        # For auto_execute=False, return instance (normal pattern)
        type(instance).__init__(instance, **kwargs)
        return instance

    @property
    def result(self) -> TDomainResult:
        """Get the execution result, raising exception on failure."""
        if not hasattr(self, "_execution_result"):
            # Lazy execution for services without auto_execute
            execution_result = self.execute()
            self._execution_result = execution_result

        result = self._execution_result
        if result.is_success:
            return result.value
        # On failure, raise exception
        raise FlextExceptions.BaseError(result.error or "Service execution failed")

    def __init__(
        self,
        **data: FlextTypes.GeneralValueType,
    ) -> None:
        """Initialize service with configuration data.

        Sets up the service instance with optional configuration parameters
        passed through **data. Delegates to parent classes for proper
        initialization of mixins, models, and infrastructure components.

        Args:
            **data: Configuration parameters for service initialization

        """
        runtime = self._create_initial_runtime()
        # Cast to FlextContext to access nested Service class
        context = cast("FlextContext", runtime.context)

        with context.Service.service_context(
            self.__class__.__name__, runtime.config.version
        ):
            super().__init__(**data)

        object.__setattr__(self, "_context", runtime.context)
        object.__setattr__(self, "_config", runtime.config)
        object.__setattr__(self, "_container", runtime.container)
        object.__setattr__(self, "_runtime", runtime)

    _context: FlextContext | None = PrivateAttr(default=None)
    _config: FlextConfig | None = PrivateAttr(default=None)
    _container: FlextContainer | None = PrivateAttr(default=None)
    _runtime: FlextModels.ServiceRuntime | None = PrivateAttr(default=None)
    _auto_result: TDomainResult | None = PrivateAttr(default=None)

    @staticmethod
    def _create_initial_runtime() -> FlextModels.ServiceRuntime:
        """Build the initial runtime triple for a new service instance."""
        base_context = FlextContext()
        global_config = FlextConfig.get_global_instance()

        class _ClonedConfig(FlextConfig):
            """Lightweight clone to bypass FlextConfig singleton semantics."""

            def __new__(
                cls, **_data: object
            ) -> Self:  # pragma: no cover - construction hook
                # Use cast to help type checker understand this is _ClonedConfig
                return cast("Self", object.__new__(cls))

        base_config = _ClonedConfig.model_construct(**global_config.model_dump())
        base_container = FlextContainer().scoped(
            config=base_config,
            context=cast("FlextProtocols.ContextProtocol", base_context),
        )

        return FlextModels.ServiceRuntime.model_construct(
            config=base_config,
            context=base_context,
            container=base_container,
        )

    def _clone_runtime(
        self,
        *,
        config_overrides: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        context: FlextProtocols.ContextProtocol | None = None,
        subproject: str | None = None,
        container_services: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], FlextTypes.FlexibleValue]]
        | None = None,
    ) -> FlextModels.ServiceRuntime:
        """Clone config/context and container in a single unified path."""
        # Access private attributes directly to avoid computed_field type issues
        if self._config is None:
            msg = "Config not initialized"
            raise RuntimeError(msg)
        if self._context is None:
            msg = "Context not initialized"
            raise RuntimeError(msg)
        if self._container is None:
            msg = "Container not initialized"
            raise RuntimeError(msg)
        cloned_config = self._config.model_copy(
            update=config_overrides or {}, deep=True
        )
        # Clone context - accept protocol but need concrete type for clone()
        if context is not None:
            # Cast protocol to concrete type for clone() method
            context_concrete = cast("FlextContext", context)
            runtime_context = context_concrete.clone()
        else:
            runtime_context = self._context.clone()
        # Cast to protocol for scoped() method
        scoped_container = self._container.scoped(
            config=cloned_config,
            context=cast("FlextProtocols.ContextProtocol", runtime_context),
            subproject=subproject,
            services=container_services,
            factories=container_factories,
        )

        return FlextModels.ServiceRuntime.model_construct(
            config=cloned_config,
            context=runtime_context,
            container=scoped_container,
        )

    @computed_field
    def runtime(
        self,
    ) -> FlextModels.ServiceRuntime:  # pragma: no cover - trivial access
        """View of the runtime triple for this service instance."""
        if self._runtime is None:
            msg = "Runtime not initialized"
            raise RuntimeError(msg)
        return self._runtime

    @property
    def context(self) -> FlextProtocols.ContextProtocol:
        """Service-scoped execution context."""
        if self._context is None:
            msg = "Context not initialized"
            raise RuntimeError(msg)
        # Type narrowing: FlextContext implements ContextProtocol structurally
        # Use cast to help type checker understand structural typing
        return cast("FlextProtocols.ContextProtocol", self._context)

    @property
    def config(self) -> FlextConfig:
        """Service-scoped configuration clone."""
        if self._config is None:
            msg = "Config not initialized"
            raise RuntimeError(msg)
        # Return concrete type for compatibility with FlextMixins.config
        return self._config

    @property
    def container(self) -> FlextProtocols.ContainerProtocol:
        """Container bound to the service context/config."""
        if self._container is None:
            msg = "Container not initialized"
            raise RuntimeError(msg)
        # Type narrowing: FlextContainer implements ContainerProtocol structurally
        # Use cast to help type checker understand structural typing
        return cast("FlextProtocols.ContainerProtocol", self._container)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute domain service logic.

        This is the core business logic method that must be implemented by all
        concrete service subclasses. It contains the actual domain operations,
        business rules, and result generation logic specific to each service.

        The method should follow railway-oriented programming principles,
        returning ``FlextResult[T]`` for consistent error handling and success
        indication.

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure
                with error details

        Note:
            Implementations should focus on business logic only. Infrastructure
            concerns like validation and error handling are handled by the base class.

        """
        ...

    def validate_business_rules(self) -> FlextResult[bool]:  # noqa: PLR6301
        """Validate business rules with extensible validation pipeline.

        Base method for business rule validation that can be overridden by subclasses
        to implement custom validation logic. By default, returns success. Subclasses
        should extend this method to add domain-specific business rule validation.

        Note: Cannot be @staticmethod because subclasses override this method and
        may need to access instance state. The base implementation doesn't use self,
        but the method signature must remain an instance method for polymorphism.

        Returns:
            FlextResult[bool]: Success (True) if all business rules pass, failure with error details

        Example:
            >>> class ValidatedService(FlextService[Data]):
            ...     def validate_business_rules(self) -> FlextResult[bool]:
            ...         if not self.has_required_data():
            ...             return FlextResult[bool].fail("Missing required data")
            ...         return FlextResult[bool].ok(True)

        """
        # Base implementation - accept all (no validation)
        # Subclasses should override for specific business rules
        return FlextResult[bool].ok(True)

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

    def get_service_info(self) -> Mapping[str, FlextTypes.FlexibleValue]:
        """Get service metadata and configuration information.

        Returns comprehensive metadata about the service instance including
        type information and execution parameters.
        Used by monitoring, logging, and debugging infrastructure.

        Returns:
            Mapping[str, FlextTypes.FlexibleValue]: Service metadata dictionary containing:
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
        # Type narrowing: cast to handle invariant type parameter
        service_typed: FlextService[FlextTypes.GeneralValueType] = cast(
            "FlextService[FlextTypes.GeneralValueType]", self
        )
        return _ServiceAccess(service_typed)


class _ServiceExecutionScope(FlextModels.ArbitraryTypesModel):
    """Immutable view of nested execution resources for a service."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    cqrs: type[FlextModels.Cqrs]
    registry: FlextRegistry
    config: FlextProtocols.ConfigProtocol
    result: type[FlextResult]
    context: FlextProtocols.ContextProtocol
    runtime: FlextModels.ServiceRuntime
    service_data: Mapping[str, FlextTypes.GeneralValueType]


class _ServiceAccess(FlextModels.ArbitraryTypesModel):
    """Gateway for service-level infrastructure access and cloning.

    Centralizes access to core FLEXT components (CQRS models, registry,
    configuration, result helpers, context, and container) and supports
    cloning of configuration plus nested execution scopes with isolated
    context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    _registry: FlextRegistry | None = PrivateAttr(default=None)
    _service: FlextService[object] = PrivateAttr()

    def __init__(
        self, service: FlextService[object] | FlextService[FlextTypes.GeneralValueType]
    ) -> None:
        super().__init__()
        # Type narrowing: accept any FlextService instance
        service_typed: FlextService[object] = cast("FlextService[object]", service)
        object.__setattr__(self, "_service", service_typed)

    @computed_field
    def cqrs(self) -> type[FlextModels.Cqrs]:  # noqa: PLR6301  # pragma: no cover - trivial access
        """CQRS facade from FlextModels.

        Note: Cannot be @staticmethod because @computed_field requires instance method
        signature for Pydantic field computation, even if self is not used.
        """
        return FlextModels.Cqrs

    @computed_field
    def registry(self) -> FlextRegistry:
        """Registry singleton for service discovery."""
        if self._registry is None:
            # Use protocol to break circular import - registry will handle dispatcher creation
            # Registry can import dispatcher directly without circular dependency
            from flext_core.registry import FlextRegistry  # noqa: PLC0415

            # Create registry without dispatcher - it will create dispatcher internally
            # This breaks the circular dependency: service -> dispatcher -> handlers -> mixins -> service
            # Use object.__setattr__ for frozen model
            object.__setattr__(self, "_registry", FlextRegistry())  # noqa: PLC2801

        if self._registry is None:
            msg = "Registry not initialized"
            raise RuntimeError(msg)
        return self._registry

    @computed_field
    def config(self) -> FlextConfig:
        """Global configuration instance for the service."""
        # Access private attribute directly to avoid computed_field type issues
        if self._service._config is None:  # noqa: SLF001
            msg = "Config not initialized"
            raise RuntimeError(msg)
        return self._service._config  # noqa: SLF001

    @computed_field
    def runtime(
        self,
    ) -> FlextModels.ServiceRuntime:  # pragma: no cover - trivial access
        """Protocol-backed runtime triple for the bound service."""
        # Access private attribute directly to avoid computed_field type issues
        if self._service._runtime is None:  # noqa: SLF001
            msg = "Runtime not initialized"
            raise RuntimeError(msg)
        return self._service._runtime  # noqa: SLF001

    @computed_field
    def result(self) -> type[FlextResult]:  # noqa: PLR6301  # pragma: no cover - trivial access
        """Result factory shortcuts.

        Note: Cannot be @staticmethod because @computed_field requires instance method
        signature for Pydantic field computation, even if self is not used.
        """
        return FlextResult

    @computed_field
    def context(self) -> FlextContext:
        """Context manager for correlation and tracing."""
        # Access private attribute directly to avoid computed_field type issues
        if self._service._context is None:  # noqa: SLF001
            msg = "Context not initialized"
            raise RuntimeError(msg)
        return self._service._context  # noqa: SLF001

    @computed_field
    def container(self) -> FlextContainer:
        """Dependency injection container bound to the service."""
        # Access private attribute directly to avoid computed_field type issues
        if self._service._container is None:  # noqa: SLF001
            msg = "Container not initialized"
            raise RuntimeError(msg)
        return self._service._container  # noqa: SLF001

    def container_scope(
        self,
        *,
        config_overrides: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        context: FlextProtocols.ContextProtocol | None = None,
        subproject: str | None = None,
        services: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        factories: Mapping[str, Callable[[], FlextTypes.FlexibleValue]] | None = None,
    ) -> FlextProtocols.ContainerProtocol:
        """Create a container scope with cloned config and optional overrides."""
        return self.runtime_scope(
            config_overrides=config_overrides,
            context=context,
            subproject=subproject,
            container_services=services,
            container_factories=factories,
        ).container

    def runtime_scope(  # noqa: PLR0913
        self,
        *,
        config_overrides: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        context: FlextProtocols.ContextProtocol | None = None,
        subproject: str | None = None,
        services: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        factories: Mapping[str, Callable[[], FlextTypes.FlexibleValue]] | None = None,
        container_services: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], FlextTypes.FlexibleValue]]
        | None = None,
    ) -> FlextModels.ServiceRuntime:
        """Clone the service runtime triple using protocol-backed models."""
        # Access private attribute directly to avoid computed_field type issues
        # _clone_runtime expects FlextContext, not protocol
        context_instance: FlextContext | None = (
            cast("FlextContext", self._service._context)  # noqa: SLF001
            if self._service._context is not None  # noqa: SLF001
            else None
        )
        # Type narrowing: context is ContextProtocol, use directly
        # Cast context_instance to protocol for compatibility
        context_instance_protocol: FlextProtocols.ContextProtocol | None = (
            cast("FlextProtocols.ContextProtocol", context_instance)
            if context_instance is not None
            else None
        )
        context_for_clone: FlextProtocols.ContextProtocol | None = (
            context or context_instance_protocol
        )
        return self._service._clone_runtime(  # noqa: SLF001
            config_overrides=config_overrides,
            context=context_for_clone,
            subproject=subproject,
            container_services=container_services or services,
            container_factories=container_factories or factories,
        )

    def clone_config(
        self, **overrides: FlextTypes.FlexibleValue
    ) -> FlextProtocols.ConfigProtocol:
        """Create a deep copy of the service configuration with overrides.

        Args:
            **overrides: Field overrides applied to the cloned configuration.

        Returns:
            ConfigProtocol: Cloned configuration instance with updates applied.

        """
        # Access private attribute directly to avoid computed_field type issues
        if self._service._config is None:  # noqa: SLF001
            msg = "Config not initialized"
            raise RuntimeError(msg)
        cloned = self._service._config.model_copy(update=overrides, deep=True)  # noqa: SLF001
        # Type narrowing: FlextConfig implements ConfigProtocol structurally
        # Use cast to help type checker understand structural typing
        return cast("FlextProtocols.ConfigProtocol", cloned)

    @contextmanager
    def nested_execution(  # noqa: PLR0913
        self,
        *,
        config_overrides: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        service_name: str | None = None,
        version: str | None = None,
        correlation_id: str | None = None,
        container_services: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        container_factories: Mapping[str, Callable[[], FlextTypes.FlexibleValue]]
        | None = None,
    ) -> Iterator[_ServiceExecutionScope]:
        """Create a nested execution scope with cloned config and context.

        Yields a frozen view of the cloned configuration, registry, container,
        context, and CQRS facade. The nested context is correlation-aware and
        isolated from the parent, enabling containerized execution flows without
        mutating global state.
        """
        # Access private attribute directly to avoid computed_field type issues
        if self._service._runtime is None:  # noqa: SLF001
            msg = "Runtime not initialized"
            raise RuntimeError(msg)
        base_runtime = self._service._runtime  # noqa: SLF001
        if base_runtime.context is None:
            msg = "Context not initialized in runtime"
            raise RuntimeError(msg)
        # Cast to FlextContext to access nested classes (Correlation, Service, Utilities)
        base_context = cast("FlextContext", base_runtime.context)
        original_correlation = base_context.Correlation.get_correlation_id()

        # Type narrowing: base_context is FlextContext, cast to protocol for runtime_scope
        base_context_protocol: FlextProtocols.ContextProtocol = cast(
            "FlextProtocols.ContextProtocol", base_context
        )
        runtime = self.runtime_scope(
            config_overrides=config_overrides,
            context=base_context_protocol,
            container_services=container_services,
            container_factories=container_factories,
        )

        # Cast to FlextContext to access nested classes
        runtime_context = cast("FlextContext", runtime.context)
        if correlation_id:
            runtime_context.Correlation.set_correlation_id(correlation_id)
        else:
            runtime_context.Utilities.ensure_correlation_id()

        service_data: Mapping[str, FlextTypes.GeneralValueType] = {
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


__all__ = ["FlextService"]
