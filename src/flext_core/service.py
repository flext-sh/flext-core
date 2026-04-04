"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services. It relies on structural typing to satisfy
``p.Service`` and provides a clean service lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from types import ModuleType
from typing import ClassVar, cast, override

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextSettings,
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
    services: Mapping[str, t.RegisterableService] | None = Field(
        default=None,
        exclude=True,
        description="Named services to register in the dependency container.",
    )
    factories: Mapping[str, t.FactoryCallable] | None = Field(
        default=None,
        exclude=True,
        description="Named factory callables to register in the dependency container.",
    )
    resources: Mapping[str, t.ResourceCallable] | None = Field(
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
    _execution_result: r[TDomainResult] | None = PrivateAttr(default=None)

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
        with FlextContext.create().Service.service_context(
            self.__class__.__name__,
            runtime.config.version,
        ):
            pass

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
        if self._execution_result is None:
            self._execution_result = self.execute()
        execution_result: r[TDomainResult] = self._execution_result
        if execution_result.is_success:
            if execution_result.value is not None:
                return execution_result.value
            error_msg = "Service execution returned None value"
            raise e.BaseError(error_msg)
        raise e.BaseError(execution_result.error or "Service execution failed")

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
        bootstrap_opts = self._runtime_bootstrap_options()
        config_type = self._get_service_config_type()
        config_type_raw = self.config_type or (
            bootstrap_opts.config_type if bootstrap_opts is not None else None
        )
        config_type_val: type[FlextSettings | p.Settings | BaseSettings]
        try:
            is_settings = config_type_raw is not None and issubclass(
                config_type_raw,
                FlextSettings,
            )
        except TypeError:
            is_settings = False

        if is_settings and config_type_raw is not None:
            config_type_val = cast("type[FlextSettings]", config_type_raw)
        else:
            config_type_val = config_type
        ctx_raw = self.initial_context or (
            bootstrap_opts.context if bootstrap_opts is not None else None
        )
        context_val: p.Context | None = ctx_raw if u.is_context(ctx_raw) else None
        config_overrides = self.config_overrides or (
            bootstrap_opts.config_overrides if bootstrap_opts is not None else None
        )
        subproject = self.subproject or (
            bootstrap_opts.subproject if bootstrap_opts is not None else None
        )
        services = self.services or (
            bootstrap_opts.services if bootstrap_opts is not None else None
        )
        factories = self.factories or (
            bootstrap_opts.factories if bootstrap_opts is not None else None
        )
        resources = self.resources or (
            bootstrap_opts.resources if bootstrap_opts is not None else None
        )
        container_overrides = self.container_overrides or (
            bootstrap_opts.container_overrides if bootstrap_opts is not None else None
        )
        raw_wire_modules = self.wire_modules or (
            bootstrap_opts.wire_modules if bootstrap_opts is not None else None
        )
        wire_modules: Sequence[ModuleType] | None = (
            [mod for mod in raw_wire_modules if isinstance(mod, ModuleType)]
            if raw_wire_modules is not None
            else None
        )
        wire_packages = self.wire_packages or (
            bootstrap_opts.wire_packages if bootstrap_opts is not None else None
        )
        wire_classes = self.wire_classes or (
            bootstrap_opts.wire_classes if bootstrap_opts is not None else None
        )
        try:
            is_flext_settings = issubclass(config_type_val, FlextSettings)
        except TypeError:
            is_flext_settings = False
        config_type_for_options: (
            type[FlextSettings | p.Settings | BaseSettings] | None
        ) = config_type_val if is_flext_settings else None
        config_overrides_scalar: t.ScalarMapping | None = None
        if config_overrides is not None:
            normalized_overrides: t.ScalarMapping = {
                key: value
                for key, value in config_overrides.items()
                if u.is_scalar(value)
            }
            config_overrides_scalar = normalized_overrides or None
        runtime_options = m.RuntimeBootstrapOptions.model_validate({
            "config_type": config_type_for_options,
            "config_overrides": config_overrides_scalar,
            "context": context_val,
            "subproject": subproject,
            "services": services,
            "factories": factories,
            "resources": resources,
            "container_overrides": container_overrides,
            "wire_modules": wire_modules,
            "wire_packages": wire_packages,
            "wire_classes": wire_classes,
        })
        return self._create_runtime(
            runtime_options=runtime_options,
        )

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
        base_options = (
            runtime_options
            if runtime_options is not None
            else m.RuntimeBootstrapOptions()
        )
        if runtime_kwargs:
            override_options = m.RuntimeBootstrapOptions.model_validate(runtime_kwargs)
            overrides: Mapping[str, t.ValueOrModel] = {
                field: getattr(override_options, field)
                for field in m.RuntimeBootstrapOptions.model_fields
                if getattr(override_options, field) is not None
            }
            runtime_options = base_options.model_copy(update=overrides)
        else:
            runtime_options = base_options
        config_type = runtime_options.config_type
        config_overrides = runtime_options.config_overrides
        context = runtime_options.context
        subproject = runtime_options.subproject
        services = runtime_options.services
        factories = runtime_options.factories
        resources = runtime_options.resources
        container_overrides = runtime_options.container_overrides
        raw_wire_modules = runtime_options.wire_modules
        wire_modules: Sequence[ModuleType] | None = (
            [module for module in raw_wire_modules if isinstance(module, ModuleType)]
            if raw_wire_modules is not None
            else None
        )
        wire_packages = runtime_options.wire_packages
        wire_classes = runtime_options.wire_classes
        try:
            cfg_is_settings = isinstance(config_type, type) and issubclass(
                config_type,
                FlextSettings,
            )
        except TypeError:
            cfg_is_settings = False
        config_cls: type[FlextSettings] = FlextSettings
        if cfg_is_settings and isinstance(config_type, type):
            try:
                if issubclass(config_type, FlextSettings):
                    config_cls = config_type
            except TypeError:
                pass
        runtime_config = config_cls.model_validate(config_overrides or {})
        runtime_context_input = (
            context if context is not None else FlextContext.create()
        )
        runtime_config_typed: p.Settings = runtime_config
        runtime_container = FlextContainer.create().scoped(
            config=runtime_config_typed,
            context=runtime_context_input,
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
        return m.ServiceRuntime(
            config=runtime_config,
            context=runtime_container.context,
            container=runtime_container,
        )

    @classmethod
    def _get_service_config_type(cls) -> type[p.Settings]:
        """Get the config type for this service class.

        Services can override this method to specify their specific config type.
        Defaults to FlextSettings for generic services.

        Returns:
            type[p.Settings]: The config class to use for this service

        """
        return FlextSettings

    @field_validator("services", mode="before")
    @classmethod
    def _normalize_scoped_services(
        cls,
        services: Mapping[str, t.RegisterableService] | None,
    ) -> Mapping[str, t.RegisterableService] | None:
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

    def is_valid(self) -> bool:
        """Check if service is in valid state for execution."""
        validation_result = (
            r[bool].ok(True).map(lambda _: self.validate_business_rules().is_success)
        )
        if validation_result.is_failure:
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
