"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services. It relies on structural typing to satisfy
``p.Service`` and provides a clean service lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from types import ModuleType
from typing import ClassVar, override

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    computed_field,
    field_validator,
)

from flext_core import (
    FlextSettings,
    c,
    e,
    h,
    m,
    p,
    r,
    t,
    u,
    x,
)


class FlextService[
    TDomainResult: t.ValueOrModel | Sequence[t.ValueOrModel] = t.ValueOrModel
    | Sequence[t.ValueOrModel],
](x):
    """Base class for domain services in FLEXT applications.

    Subclasses implement ``execute`` to run business logic and return
    ``r`` (r) values. The base inherits :class:`x` (which extends
    :class:`u`) so services can reuse runtime automation for creating
    scoped config/context/container triples via :meth:`create_service_runtime`
    while remaining protocol compliant via structural typing.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )
    # --- Service Bootstrap Configuration ---
    config_type: type | None = Field(
        default=None,
        exclude=True,
        description="Settings class used to load runtime configuration.",
    )
    config_overrides: t.ContainerMapping | None = Field(
        default=None,
        exclude=True,
        description="Key-value overrides applied on top of the loaded configuration.",
    )
    initial_context: p.Context | None = Field(
        default=None,
        exclude=True,
        description="Initial execution context to inject into the runtime.",
    )
    subproject: str | None = Field(
        default=None,
        exclude=True,
        description="Subproject name used to scope configuration and wiring.",
    )
    services: t.ServiceMap | None = Field(
        default=None,
        exclude=True,
        description="Named services to register in the dependency container.",
    )
    factories: t.FactoryMap | None = Field(
        default=None,
        exclude=True,
        description="Named factory callables to register in the dependency container.",
    )
    resources: t.ResourceMap | None = Field(
        default=None,
        exclude=True,
        description="Named lifecycle resources to register in the dependency container.",
    )
    container_overrides: t.ScalarMapping | None = Field(
        default=None,
        exclude=True,
        description="Provider overrides applied to the dependency container.",
    )
    wire_modules: Sequence[ModuleType] | None = Field(
        default=None,
        exclude=True,
        description="Modules to wire for dependency-injector resolution.",
    )
    wire_packages: t.StrSequence | None = Field(
        default=None,
        exclude=True,
        description="Package names to consider for dependency wiring.",
    )
    wire_classes: Sequence[type] | None = Field(
        default=None,
        exclude=True,
        description="Classes whose modules are wired for dependency resolution.",
    )

    _context: p.Context | None = PrivateAttr(default=None)
    _config: p.Settings | None = PrivateAttr(default=None)
    _container: p.Container | None = PrivateAttr(default=None)
    _runtime: m.ServiceRuntime | None = PrivateAttr(default=None)
    _discovered_handlers: Sequence[tuple[str, m.DecoratorConfig]] = PrivateAttr(
        default_factory=tuple,
    )

    # --- Internal State ---
    _execution_result: p.Result[TDomainResult] | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        config: p.Settings | None = None,
        config_type: type | None = None,
        config_overrides: t.ContainerMapping | None = None,
        initial_context: p.Context | None = None,
        subproject: str | None = None,
        services: t.ServiceMap | None = None,
        factories: t.FactoryMap | None = None,
        resources: t.ResourceMap | None = None,
        container_overrides: t.ScalarMapping | None = None,
        wire_modules: Sequence[ModuleType] | None = None,
        wire_packages: t.StrSequence | None = None,
        wire_classes: Sequence[type] | None = None,
        **data: object,
    ) -> None:
        """Canonical public bootstrap signature for all FLEXT services."""
        super().__init__(
            config=config,
            config_type=config_type,
            config_overrides=config_overrides,
            initial_context=initial_context,
            subproject=subproject,
            services=services,
            factories=factories,
            resources=resources,
            container_overrides=container_overrides,
            wire_modules=wire_modules,
            wire_packages=wire_packages,
            wire_classes=wire_classes,
            **data,
        )

    @override
    def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
        """Post-initialization hook.

        Sets up the service instance with runtime configuration after Pydantic
        has populated the fields.

        Auto-discovery of handler-decorated methods enables zero-config handler
        registration: developers can mark methods with @h.handler() and they are
        automatically discovered.
        """
        runtime = self._create_initial_runtime()
        with u.service_context_scope(
            self.__class__.__name__,
            runtime.config.version,
        ):
            self._context = runtime.context
            self._config = runtime.config
            self._container = runtime.container
            self._runtime = runtime
            self._discovered_handlers = (
                h.Discovery.scan_class(self.__class__)
                if h.Discovery.has_handlers(self.__class__)
                else []
            )

    @computed_field(repr=False)
    @property
    def result(self) -> TDomainResult:
        """Get the execution result, raising exception on failure."""
        return self._unwrap_execution_result(self._get_execution_result())

    @staticmethod
    def _unwrap_execution_result[
        TResult: t.ValueOrModel | Sequence[t.ValueOrModel],
    ](
        execution_result: p.Result[TResult],
    ) -> TResult:
        """Unwrap one successful execution result with the original generic type."""
        if execution_result.failure:
            operation = "service execution"
            reason = execution_result.error or c.ERR_SERVICE_EXECUTION_FAILED
            params = m.OperationErrorParams(operation=operation, reason=reason)
            raise e.BaseError(
                e.render_error_template(
                    c.ERR_TEMPLATE_FAILED_WITH_ERROR,
                    operation=operation,
                    error=reason,
                    params=params,
                ),
            )
        return execution_result.unwrap()

    def _get_execution_result(self) -> p.Result[TDomainResult]:
        """Return cached execution result or execute the service once."""
        execution_result = self._execution_result
        if execution_result is None:
            execution_result = self.execute()
            self._execution_result = execution_result
        return execution_result

    @property
    @override
    def config(self) -> p.Settings:
        """Service-scoped configuration clone."""
        return u.require_initialized(self._config, "Config")

    @property
    @override
    def container(self) -> p.Container:
        """Container bound to the service context/config."""
        return u.require_initialized(self._container, "Container")

    @property
    @override
    def context(self) -> p.Context:
        """Service-scoped execution context."""
        return u.require_initialized(self._context, "Context")

    @computed_field
    def runtime(self) -> m.ServiceRuntime:
        """View of the runtime triple for this service instance."""
        return u.require_initialized(self._runtime, "Runtime")

    @property
    def settings(self) -> p.Settings:
        """Return service config narrowed to FlextSettings."""
        return self.config

    @classmethod
    def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions | None:
        return None

    def _create_initial_runtime(self) -> m.ServiceRuntime:
        """Build the initial runtime triple for a new service instance."""
        return u.build_service_runtime(self)

    @classmethod
    def _create_runtime(
        cls,
        runtime_options: m.RuntimeBootstrapOptions | None = None,
        **runtime_kwargs: t.RuntimeData,
    ) -> m.ServiceRuntime:
        """Materialize config, context, and container with DI wiring in one call.

        This method provides the same parameterized automation previously found in
        ``u.create_service_runtime`` but uses the factory methods of each
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
        return u.build_service_runtime(runtime_options, **runtime_kwargs)

    @classmethod
    def _get_service_config_type(cls) -> type[FlextSettings]:
        """Get the config type for this service class.

        Services can override this method to specify their specific config type.
        Defaults to FlextSettings for generic services.

        Returns:
            type[FlextSettings]: The config class to use for this service

        """
        return FlextSettings

    @field_validator("services", mode="before")
    @classmethod
    def _normalize_scoped_services(
        cls,
        services: t.ServiceMap | None,
    ) -> t.ServiceMap | None:
        """Normalize and validate scoped services using Pydantic model."""
        if services is None:
            return None
        normalized: MutableMapping[str, t.RegisterableService] = {}
        for name, service in services.items():
            try:
                m.ServiceRegistration(name=str(name), service=service)
                normalized[str(name)] = service
            except ValidationError:
                continue
        return normalized or None

    def execute(self) -> r[TDomainResult]:
        """Execute domain service logic.

        This is the core business logic method that must be implemented by all
        concrete service subclasses. It contains the actual domain operations,
        business rules, and result generation logic specific to each service.

        Business Rule: Executes the domain service business logic and returns
        a r indicating success or failure. This method is the primary
        entry point for all domain service operations in the FLEXT ecosystem.
        All business logic, domain rules, and operational workflows are executed
        through this method. The method must follow railway-oriented programming
        principles, returning ``r[TDomainResult]`` instead of raising exceptions,
        ensuring predictable error handling and composable service pipelines.

        Audit Implication: Service execution is a critical audit event that
        represents the execution of business logic and domain operations. All
        service executions should be logged with appropriate context and
        correlation IDs. The r return type ensures audit trail
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
            NotImplementedError: Subclasses must implement this method.

        """
        raise NotImplementedError

    def valid(self) -> bool:
        """Check if service is in valid state for execution."""
        validation_result = (
            r[bool].ok(True).map(lambda _: self.validate_business_rules().success)
        )
        if validation_result.failure:
            exc = getattr(validation_result, "_exception", None)
            self.logger.debug(
                "Service business rule validation failed",
                exc_info=bool(exc),
            )
            return False
        return validation_result.value

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
        return r[bool].ok(True)


s = FlextService
__all__ = ["FlextService", "s"]
