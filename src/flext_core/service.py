"""FlextService - Domain Service Base Class Module.

This module provides FlextService[T], a base class for implementing domain
services with infrastructure support including dependency injection, validation,
type-safe result handling, and auto-execution patterns. Implements structural
typing via FlextProtocols.Service through duck typing, providing a foundation
for CQRS command and query services throughout the FLEXT ecosystem.

Scope: Domain service base class, auto-execution pattern, business rule validation,
service metadata, type-safe execution infrastructure, railway-oriented programming
with FlextResult, and dependency injection support.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Iterator

from pydantic import ConfigDict, PrivateAttr, computed_field

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.dispatcher import FlextDispatcher
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
    """Domain Service Base Class for FLEXT ecosystem.

    Provides comprehensive infrastructure support for implementing domain services
    with type-safe execution, dependency injection, business rule validation,
    and auto-execution patterns. Implements structural typing via FlextProtocols.Service
    through duck typing (no inheritance required), serving as the foundation for
    CQRS command and query services throughout the FLEXT ecosystem.

    Core Features:
    - Abstract base class with generic type parameters for type-safe results
    - Railway-oriented programming with FlextResult for error handling
    - Auto-execution pattern for immediate service execution on instantiation
    - Business rule validation with extensible validation pipeline
    - Dependency injection support through FlextMixins
    - Pydantic integration for configuration and validation
    - Service metadata and introspection capabilities

    Architecture:
    - Single class with nested service execution logic
    - DRY principle applied through centralized result handling
    - SOLID principles: Single Responsibility for domain service execution
    - Railway pattern for consistent error handling without exceptions
    - Structural typing for protocol compliance without inheritance

    Type Parameters:
    - TDomainResult: The type of result returned by service execution

    Usage Examples:
        >>> # Standard service usage
        >>> class UserService(FlextService[User]):
        ...     def execute(self) -> FlextResult[User]:
        ...         # Domain logic here
        ...         user = User(id=1, name="John")
        ...         return self.ok(user)
        >>>
        >>> service = UserService()
        >>> result = service.execute()
        >>> if result.is_success:
        ...     user = result.value
    """

    def __init__(
        self,
        **data: object,
    ) -> None:
        """Initialize service with configuration data.

        Sets up the service instance with optional configuration parameters
        passed through **data. Delegates to parent classes for proper
        initialization of mixins, models, and infrastructure components.

        Args:
            **data: Configuration parameters for service initialization

        """
        runtime = self._create_initial_runtime()

        with runtime.context.Service.service_context(
            self.__class__.__name__, runtime.config.version
        ):
            super().__init__(**data)

        object.__setattr__(self, "_context", runtime.context)
        object.__setattr__(self, "_config", runtime.config)
        object.__setattr__(self, "_container", runtime.container)
        object.__setattr__(self, "_runtime", runtime)

    _context: FlextProtocols.ContextProtocol = PrivateAttr(default=None)
    _config: FlextProtocols.ConfigProtocol = PrivateAttr(default=None)
    _container: FlextProtocols.ContainerProtocol = PrivateAttr(default=None)
    _runtime: FlextModels.ServiceRuntime = PrivateAttr(default=None)

    @staticmethod
    def _create_initial_runtime() -> FlextModels.ServiceRuntime:
        """Build the initial runtime triple for a new service instance."""

        base_context = FlextContext()
        global_config = FlextConfig.get_global_instance()

        class _ClonedConfig(FlextConfig):
            """Lightweight clone to bypass FlextConfig singleton semantics."""

            def __new__(cls, **_data: object):  # pragma: no cover - construction hook
                return object.__new__(cls)

        base_config = _ClonedConfig.model_construct(**global_config.model_dump())
        base_container = FlextContainer().scoped(
            config=base_config, context=base_context
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
        container_services: Mapping[str, object] | None = None,
        container_factories: Mapping[str, Callable[[], object]] | None = None,
    ) -> FlextModels.ServiceRuntime:
        """Clone config/context and container in a single unified path."""

        cloned_config = self.config.model_copy(
            update=config_overrides or {}, deep=True
        )
        runtime_context = (
            context.clone() if context is not None else self.context.clone()
        )
        scoped_container = self.container.scoped(
            config=cloned_config,
            context=runtime_context,
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
    def runtime(self) -> FlextModels.ServiceRuntime:  # pragma: no cover - trivial access
        """View of the runtime triple for this service instance."""

        return self._runtime

    @computed_field
    def context(self) -> FlextProtocols.ContextProtocol:  # type: ignore[override]
        """Service-scoped execution context."""

        return self._context

    @computed_field
    def config(self) -> FlextProtocols.ConfigProtocol:  # type: ignore[override]
        """Service-scoped configuration clone."""

        return self._config

    @computed_field
    def container(self) -> FlextProtocols.ContainerProtocol:  # type: ignore[override]
        """Container bound to the service context/config."""

        return self._container

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute domain service logic - abstract method to be implemented by subclasses.

        This is the core business logic method that must be implemented by all
        concrete service subclasses. It contains the actual domain operations,
        business rules, and result generation logic specific to each service.

        The method should follow railway-oriented programming principles,
        returning FlextResult[T] for consistent error handling and success indication.

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure with error details

        Note:
            Implementations should focus on business logic only. Infrastructure
            concerns like validation and error handling are handled by the base class.

        """
        ...

    @computed_field
    def result(self) -> TDomainResult:
        """Get execution result with lazy evaluation.

        Computed property that executes the service and returns the result value.
        Uses Pydantic's computed_field for caching and lazy evaluation. Raises
        exception on failure to maintain synchronous error handling.

        Returns:
            TDomainResult: The successful execution result

        Raises:
            FlextExceptions.BaseError: When execution fails

        Example:
            >>> service = MyService()
            >>> result_value = service.result  # Executes and returns value

        """
        result = self.execute()
        if result.is_success:
            return result.value
        raise FlextExceptions.BaseError(result.error or "Service execution failed")

    def validate_business_rules(self) -> FlextResult[bool]:
        """Validate business rules with extensible validation pipeline.

        Base method for business rule validation that can be overridden by subclasses
        to implement custom validation logic. By default, returns success. Subclasses
        should extend this method to add domain-specific business rule validation.

        The validation follows railway-oriented programming principles, allowing
        for complex validation pipelines that can fail early or accumulate errors.

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
    def access(self) -> "_ServiceAccess":
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

        return _ServiceAccess(self)


class _ServiceExecutionScope(FlextModels.ArbitraryTypesModel):
    """Immutable view of nested execution resources for a service."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    cqrs: type[FlextModels.Cqrs]
    registry: FlextRegistry
    config: FlextProtocols.ConfigProtocol
    result: type[FlextResult]
    context: FlextProtocols.ContextProtocol
    runtime: FlextModels.ServiceRuntime
    service_data: Mapping[str, object]



class _ServiceAccess(FlextModels.ArbitraryTypesModel):
    """Gateway for service-level infrastructure access and cloning.

    Centralizes access to core FLEXT components (CQRS models, registry,
    configuration, result helpers, context, and container) and supports
    cloning of configuration plus nested execution scopes with isolated
    context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    _registry: FlextRegistry | None = PrivateAttr(default=None)

    def __init__(self, service: FlextService[object]):
        super().__init__()
        object.__setattr__(self, "_service", service)

    @computed_field
    def cqrs(self) -> type[FlextModels.Cqrs]:  # pragma: no cover - trivial access
        """CQRS facade from FlextModels."""

        return FlextModels.Cqrs

    @computed_field
    def registry(self) -> FlextRegistry:
        """Registry singleton for service discovery."""

        if self._registry is None:
            try:
                dispatcher = FlextDispatcher()
            except Exception:
                class _RegistryDispatcher(FlextDispatcher):
                    def __init__(self) -> None:  # pragma: no cover - safety fallback
                        # Bypass base initialization to avoid optional dependencies
                        pass

                dispatcher = _RegistryDispatcher()

            self._registry = FlextRegistry(dispatcher)

        return self._registry

    @computed_field
    def config(self) -> FlextConfig:
        """Global configuration instance for the service."""

        return self._service.config

    @computed_field
    def runtime(self) -> FlextModels.ServiceRuntime:  # pragma: no cover - trivial access
        """Protocol-backed runtime triple for the bound service."""

        return self._service.runtime

    @computed_field
    def result(self) -> type[FlextResult]:  # pragma: no cover - trivial access
        """Result factory shortcuts."""

        return FlextResult

    @computed_field
    def context(self) -> FlextContext:
        """Context manager for correlation and tracing."""

        return self._service.context

    @computed_field
    def container(self) -> FlextContainer:
        """Dependency injection container bound to the service."""

        return self._service.container

    def container_scope(
        self,
        *,
        config_overrides: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        context: FlextProtocols.ContextProtocol | None = None,
        subproject: str | None = None,
        services: Mapping[str, object] | None = None,
        factories: Mapping[str, Callable[[], object]] | None = None,
    ) -> FlextProtocols.ContainerProtocol:
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
        config_overrides: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        context: FlextProtocols.ContextProtocol | None = None,
        subproject: str | None = None,
        services: Mapping[str, object] | None = None,
        factories: Mapping[str, Callable[[], object]] | None = None,
        container_services: Mapping[str, object] | None = None,
        container_factories: Mapping[str, Callable[[], object]] | None = None,
    ) -> FlextModels.ServiceRuntime:
        """Clone the service runtime triple using protocol-backed models."""

        return self._service._clone_runtime(
            config_overrides=config_overrides,
            context=context or self.context,
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
            FlextConfig: Cloned configuration instance with updates applied.
        """

        return self.config.model_copy(update=overrides, deep=True)

    @contextmanager
    def nested_execution(
        self,
        *,
        config_overrides: Mapping[str, FlextTypes.FlexibleValue] | None = None,
        service_name: str | None = None,
        version: str | None = None,
        correlation_id: str | None = None,
        container_services: Mapping[str, object] | None = None,
        container_factories: Mapping[str, Callable[[], object]] | None = None,
    ) -> Iterator[_ServiceExecutionScope]:
        """Create a nested execution scope with cloned config and context.

        Yields a frozen view of the cloned configuration, registry, container,
        context, and CQRS facade. The nested context is correlation-aware and
        isolated from the parent, enabling containerized execution flows without
        mutating global state.
        """

        base_runtime = self.runtime
        base_context = base_runtime.context
        original_correlation = base_context.Correlation.get_correlation_id()

        runtime = self.runtime_scope(
            config_overrides=config_overrides,
            context=base_context,
            container_services=container_services,
            container_factories=container_factories,
        )

        if correlation_id:
            runtime.context.Correlation.set_correlation_id(correlation_id)
        else:
            runtime.context.Utilities.ensure_correlation_id()

        service_data = {
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

        with runtime.context.Service.service_context(
            service_name or self._service.__class__.__name__,
            version or runtime.config.version,
        ):
            yield scope

        if original_correlation is None:
            base_context.Correlation.reset_correlation_id()
        else:
            base_context.Correlation.set_correlation_id(original_correlation)


__all__ = ["FlextService"]
