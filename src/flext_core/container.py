"""Truthful dependency container for the dispatcher-first CQRS stack.

The container is the registry of the core runtime. It keeps one bookkeeping of
named services, factories and resources, and every registration passes through
one private write path: an empty, duplicate or reserved name raises
``e.ValidationError`` instead of being ignored. Resolution surfaces ``r``
(Result) so lookup failures stay explicit.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Self, TypeGuard, cast, overload, override

import flext_core
from flext_core._container import FlextContainerTestingOps

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType, ModuleType


class FlextContainer(FlextContainerTestingOps, flext_core.p.Container):
    """Process-wide registry of services, factories and resources.

    One mapping holds every registration; the core runtime names
    (``c.CONTAINER_RESERVED_NAMES``) are written only by the container itself
    and stay out of ``has``/``names``/``drop``. Scoped containers built with
    ``scope`` inherit the public registrations and bind their own core
    services to their own settings and context.
    """

    _settings_type: ClassVar[flext_core.p.SettingsType] = flext_core.FlextSettings

    _context_type: ClassVar[flext_core.p.ContextType] = flext_core.FlextContext

    _context: flext_core.p.Context

    _config: flext_core.p.Settings

    _user_overrides: flext_core.p.ConfigMap

    _initialized: bool = False

    _registrations: MutableMapping[
        str,
        flext_core.m.ServiceRegistration
        | flext_core.m.FactoryRegistration
        | flext_core.m.ResourceRegistration,
    ]

    _global_config: flext_core.p.ContainerConfig

    def __new__(
        cls,
        *,
        registration: flext_core.p.ServiceRegistrationSpec | None = None,
    ) -> Self:
        """Create or return the global singleton instance.

        ``registration`` mirrors the ``__init__`` signature the constructor
        protocol forwards here; ``__init__`` alone applies it.
        """
        _ = registration
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        return cls._global_instance

    def __init__(
        self,
        *,
        registration: flext_core.p.ServiceRegistrationSpec | None = None,
    ) -> None:
        """Initialize the singleton once; later calls apply the explicit spec."""
        if not self._initialized:
            self.initialize_registrations(registration=registration)
        elif registration is not None:
            spec = flext_core.m.ServiceRegistrationSpec.model_validate(
                registration, from_attributes=True
            )
            self._apply_explicit_bootstrap(spec)
            self._write_spec(spec)

    @property
    @override
    def settings(self) -> flext_core.p.Settings:
        """Configuration bound to this container."""
        return self._config

    @property
    @override
    def context(self) -> flext_core.p.Context:
        """Execution context bound to this container."""
        return self._context

    @staticmethod
    def _matches_service_type[T: flext_core.p.RegisterableService](
        value: flext_core.p.RegisterableService,
        expected: type[T],
    ) -> TypeGuard[T]:
        """Narrow a resolved service through its structural runtime type.

        Returns:
            The resulting ``TypeGuard[T]``.

        """
        return isinstance(value, expected)

    def _write(
        self,
        name: str,
        build: Callable[
            [],
            flext_core.m.ServiceRegistration
            | flext_core.m.FactoryRegistration
            | flext_core.m.ResourceRegistration,
        ],
        *,
        internal: bool = False,
    ) -> Self:
        """Apply the registration rules and store one validated record.

        This is the only path that mutates the registrations. Public writes
        reject empty, reserved and duplicate names; the container's own core
        writes (``internal``) may only target reserved names.

        Returns:
            The resulting ``Self``.

        Raises:
            ValidationError: If ``not name``; or if ``reserved != internal``; or if
                ``not internal and name in self._registrations``; or if a ``ValueError``
                is caught.

        """
        if not name:
            raise flext_core.e.ValidationError(flext_core.c.ERR_CONTAINER_NAME_EMPTY)
        reserved = name in flext_core.c.CONTAINER_RESERVED_NAMES
        if reserved != internal:
            raise flext_core.e.ValidationError(
                flext_core.c.ERR_CONTAINER_NAME_RESERVED.format(name=name)
            )
        if not internal and name in self._registrations:
            raise flext_core.e.ValidationError(
                flext_core.c.ERR_CONTAINER_NAME_DUPLICATE.format(name=name)
            )
        try:
            record = build()
        except ValueError as exc:
            raise flext_core.e.ValidationError(
                flext_core.c.ERR_CONTAINER_REGISTRATION_FAILED.format(
                    name=name, reason=exc
                ),
            ) from exc
        self._registrations[name] = record
        return self

    def _write_spec(self, spec: flext_core.p.ServiceRegistrationSpec) -> None:
        """Write every service, factory and resource declared by a spec."""
        for name, service in (spec.services or {}).items():
            _ = self.bind(name, service)
        for name, factory in (spec.factories or {}).items():
            _ = self.factory(name, factory)
        for name, resource in (spec.resources or {}).items():
            _ = self.resource(name, resource)

    @override
    def bind(self, name: str, impl: flext_core.p.RegisterableService) -> Self:
        """Bind a concrete service instance or value.

        Returns:
            The resulting ``Self``.

        """
        return self._write(
            name,
            partial(flext_core.m.ServiceRegistration, name=name, service=impl),
        )

    @override
    def factory(self, name: str, impl: flext_core.p.FactoryCallable) -> Self:
        """Bind a factory callable invoked on every resolve.

        Returns:
            The resulting ``Self``.

        """
        return self._write(
            name,
            partial(flext_core.m.FactoryRegistration, name=name, factory=impl),
        )

    @override
    def resource(self, name: str, impl: flext_core.p.ResourceCallable) -> Self:
        """Bind a resource factory invoked on every resolve.

        Returns:
            The resulting ``Self``.

        """
        return self._write(
            name,
            partial(flext_core.m.ResourceRegistration, name=name, factory=impl),
        )

    @staticmethod
    def _resolve_callable(
        callable_obj: flext_core.p.FactoryCallable,
        kind: str,
        name: str,
    ) -> flext_core.p.Result[flext_core.p.RegisterableService]:
        """Validate factory/resource admission and return the original value.

        Returns:
            The resulting ``p.Result[p.RegisterableService]``.

        """
        try:
            resolved = callable_obj()
            flext_core.m.ServiceRegistration(name=name, service=resolved)
        except flext_core.c.EXC_BROAD_RUNTIME as exc:
            return flext_core.r[flext_core.p.RegisterableService].from_result(
                flext_core.e.fail_operation(
                    f"resolve {kind}",
                    exc,
                    result_type=flext_core.r[flext_core.p.RegisterableService],
                ),
            )
        return flext_core.r[flext_core.p.RegisterableService].ok(resolved)

    @overload
    def resolve[T: flext_core.p.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T],
    ) -> flext_core.p.Result[T]: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        type_cls: None = None,
    ) -> flext_core.p.Result[flext_core.p.RegisterableService]: ...

    @override
    def resolve[T: flext_core.p.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T] | None = None,
    ) -> flext_core.p.Result[T] | flext_core.p.Result[flext_core.p.RegisterableService]:
        """Resolve a registered service, factory or resource by name.

        Args:
            name: The registered name to resolve.
            type_cls: When given, the resolved value is type-checked against
                this class and the failure result carries a type-mismatch
                error on mismatch.

        Returns:
            ``r.Ok`` wrapping the resolved instance, or ``r.Fail`` with a
            not-found or type-mismatch error; lookup failures never raise.

        """
        match self._registrations.get(name):
            case None:
                return flext_core.r[flext_core.p.RegisterableService].from_result(
                    flext_core.e.fail_not_found("service", name),
                )
            case flext_core.m.ServiceRegistration() as record:
                result = flext_core.r[flext_core.p.RegisterableService].ok(
                    record.service
                )
            case flext_core.m.FactoryRegistration() as record:
                result = self._resolve_callable(record.factory, "factory", name)
            case record:
                result = self._resolve_callable(record.factory, "resource", name)
        if type_cls is None or result.failure:
            return result
        if self._matches_service_type(result.value, type_cls):
            return flext_core.r[T].ok(result.value)
        return flext_core.r[T].from_result(
            flext_core.e.fail_type_mismatch(
                type_cls.__name__,
                type(result.value).__name__,
                result_type=flext_core.r[T],
            ),
        )

    @override
    def snapshot(self) -> flext_core.p.ConfigMap:
        """Return the merged settings exposed by this container.

        Returns:
            The merged settings exposed by this container.

        """
        return flext_core.m.ConfigMap.model_validate(self._global_config.model_dump())

    @override
    def has(self, name: str) -> bool:
        """Return whether a public service, factory, or resource is registered.

        Returns:
            Whether a public service, factory, or resource is registered.

        """
        return (
            name in self._registrations
            and name not in flext_core.c.CONTAINER_RESERVED_NAMES
        )

    @override
    def names(self) -> Sequence[str]:
        """List the public services, factories, and resources.

        Returns:
            The resulting ``Sequence[str]``.

        """
        return [name for name in self._registrations if self.has(name)]

    def initialize_registrations(
        self,
        *,
        registration: flext_core.p.ServiceRegistrationSpec | None = None,
    ) -> None:
        """Reset the registrations from a spec and register the core services.

        Args:
            registration: Optional spec providing container config, user
                overrides, settings and context; defaults apply when omitted.

        """
        spec = (
            flext_core.m.ServiceRegistrationSpec.model_validate(
                registration, from_attributes=True
            )
            if registration is not None
            else flext_core.m.ServiceRegistrationSpec()
        )
        self._registrations = {}
        self._global_config = (
            cast("flext_core.m.ContainerConfig", spec.container_config)
            if spec.container_config is not None
            else flext_core.m.ContainerConfig()
        )
        overrides = spec.user_overrides
        if isinstance(overrides, flext_core.m.ConfigMap):
            self._user_overrides = overrides
        elif isinstance(overrides, Mapping):
            self._user_overrides = flext_core.m.ConfigMap.model_validate(overrides)
        else:
            self._user_overrides = flext_core.m.ConfigMap(root={})
        self._config = (
            spec.settings
            if spec.settings is not None
            else self._settings_type.fetch_global()
        )
        self._context = (
            spec.context if spec.context is not None else self._context_type.create()
        )
        self._write_spec(spec)
        self.register_core_services(dispatcher=spec.dispatcher)
        self._initialized = True

    @override
    def register_core_services(
        self, *, dispatcher: flext_core.p.Dispatcher | None = None
    ) -> None:
        """Register core services, retaining an explicitly injected command bus."""
        core = (
            (
                flext_core.c.Directory.CONFIG,
                partial(
                    flext_core.m.ServiceRegistration,
                    name=flext_core.c.Directory.CONFIG,
                    service=self._config,
                ),
            ),
            (
                flext_core.c.ServiceName.LOGGER,
                partial(
                    flext_core.m.FactoryRegistration,
                    name=flext_core.c.ServiceName.LOGGER,
                    factory=partial(
                        flext_core.u.fetch_logger, flext_core.c.LOGGER_NAME_FLEXT_CORE
                    ),
                ),
            ),
            (
                flext_core.c.FIELD_CONTEXT,
                partial(
                    flext_core.m.ServiceRegistration,
                    name=flext_core.c.FIELD_CONTEXT,
                    service=self._context,
                ),
            ),
            (
                flext_core.c.ServiceName.COMMAND_BUS,
                lambda: flext_core.m.ServiceRegistration(
                    name=flext_core.c.ServiceName.COMMAND_BUS,
                    service=dispatcher
                    if dispatcher is not None
                    else flext_core.FlextDispatcher(),
                ),
            ),
        )
        for name, build in core:
            if name not in self._registrations or (
                name == flext_core.c.ServiceName.COMMAND_BUS and dispatcher is not None
            ):
                _ = self._write(name, build, internal=True)

    @override
    def scope(
        self,
        *,
        subproject: str | None = None,
        registration: flext_core.p.ServiceRegistrationSpec | None = None,
    ) -> Self:
        """Create an isolated scope inheriting the public registrations.

        Registrations declared by ``registration`` override inherited names;
        the scope binds its own core services to its own settings and context.

        Returns:
            The resulting ``Self``.

        """
        spec = (
            flext_core.m.ServiceRegistrationSpec.model_validate(
                registration, from_attributes=True
            )
            if registration is not None
            else flext_core.m.ServiceRegistrationSpec()
        )
        settings_source = (
            spec.settings if spec.settings is not None else self._config.clone()
        )
        scoped_context = self.context.clone() if spec.context is None else spec.context
        if subproject:
            _ = scoped_context.set("subproject", subproject)
        inherited = {name: self._registrations[name] for name in self.names()}
        scoped = flext_core.u.create_instance(self.__class__)
        scoped.initialize_registrations(
            registration=flext_core.m.ServiceRegistrationSpec(
                settings=settings_source,
                context=scoped_context,
                dispatcher=spec.dispatcher,
                services={
                    **{
                        name: record.service
                        for name, record in inherited.items()
                        if isinstance(record, flext_core.m.ServiceRegistration)
                    },
                    **(spec.services or {}),
                },
                factories={
                    **{
                        name: record.factory
                        for name, record in inherited.items()
                        if isinstance(record, flext_core.m.FactoryRegistration)
                    },
                    **(spec.factories or {}),
                },
                resources={
                    **{
                        name: record.factory
                        for name, record in inherited.items()
                        if isinstance(record, flext_core.m.ResourceRegistration)
                    },
                    **(spec.resources or {}),
                },
                user_overrides=dict(self._user_overrides.root),
                container_config=self._global_config.model_copy(deep=True),
            ),
        )
        return scoped

    @override
    def drop(self, name: str) -> flext_core.p.Result[bool]:
        """Remove a public service, factory, or resource registration by name.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        if not self.has(name):
            return flext_core.r[bool].from_result(
                flext_core.e.fail_not_found(
                    "service", name, result_type=flext_core.r[bool]
                ),
            )
        del self._registrations[name]
        return flext_core.r[bool].ok(value=True)

    @override
    def dispatcher(self) -> flext_core.p.Result[flext_core.p.Dispatcher]:
        """Resolve the canonical dispatcher / command bus.

        Returns:
            The resulting ``p.Result[p.Dispatcher]``.

        """
        result = self.resolve(flext_core.c.ServiceName.COMMAND_BUS)
        if result.failure:
            return flext_core.r[flext_core.p.Dispatcher].from_result(
                flext_core.e.fail_not_found(
                    "dispatcher",
                    flext_core.c.ServiceName.COMMAND_BUS,
                    result_type=flext_core.r[flext_core.p.Dispatcher],
                ),
            )
        if isinstance(result.value, flext_core.p.Dispatcher):
            return flext_core.r[flext_core.p.Dispatcher].ok(result.value)
        return flext_core.r[flext_core.p.Dispatcher].from_result(
            flext_core.e.fail_type_mismatch(
                "dispatcher",
                flext_core.u.type_name(result.value),
                result_type=flext_core.r[flext_core.p.Dispatcher],
            ),
        )

    @classmethod
    def build_service_runtime(
        cls,
        source: flext_core.p.RuntimeBootstrapProvider
        | flext_core.p.RuntimeBootstrapOptions
        | flext_core.p.MixinsInfrastructure
        | None = None,
    ) -> flext_core.p.ServiceRuntime:
        """Materialize settings, context, container and dispatcher for one component.

        The container scopes the injected container, or the shared container,
        preserving public DI registrations. Its settings are the runtime's
        actual elected provider. A dispatcher the options do not inject is the
        container's command bus; failing to resolve it raises with its cause.

        Returns:
            The component's service runtime.

        Raises:
            ValidationError: If neither logging policy nor an explicit logger is given.

        """
        options = flext_core.u.resolve_runtime_options(source)
        settings = flext_core.u.resolve_runtime_settings(options)
        logger = options.logger
        if logger is None:
            if not isinstance(settings, flext_core.p.LoggingSettings):
                msg = "Runtime settings must declare logging policy or inject a logger"
                raise flext_core.e.ValidationError(msg)
            logger = flext_core.u.create_module_logger(
                f"{type(source).__module__}.{type(source).__qualname__}",
                runtime_settings=settings,
                options=options.logging_options
                if options.logging_options is not None
                else flext_core.u.logging_options(),
            )
        context = (
            options.context
            if options.context is not None
            else flext_core.FlextContext.create()
        )
        source_container = options.container or cls.shared()
        container = source_container.scope(
            registration=flext_core.m.ServiceRegistrationSpec(
                settings=settings,
                context=context,
                dispatcher=options.dispatcher,
            ),
        )
        return flext_core.m.ServiceRuntime(
            settings=container.settings,
            context=container.context,
            container=container,
            dispatcher=container.dispatcher().unwrap(),
            logger=logger,
            logger_injected=options.logger is not None,
        )

    @classmethod
    def shared(
        cls,
        *,
        settings: flext_core.p.Settings | None = None,
        context: flext_core.p.Context | None = None,
        auto_register_factories: bool = False,
    ) -> Self:
        """Return the canonical shared container instance.

        ``auto_register_factories`` registers every ``@d.factory()`` function
        of the calling module; a caller that cannot be resolved to an imported
        module raises ``e.ValidationError``.

        Returns:
            The canonical shared container instance.

        """
        instance = cls()
        if settings is not None or context is not None:
            instance._apply_explicit_bootstrap(
                flext_core.m.ServiceRegistrationSpec(
                    settings=settings, context=context
                ),
            )
        if auto_register_factories:
            caller_module = cls._resolve_caller_module(inspect.currentframe())
            cls._auto_register_module_factories(instance, caller_module)
        return instance

    @staticmethod
    def _resolve_caller_module(frame: FrameType | None) -> ModuleType:
        """Resolve the imported module that called ``shared`` or raise.

        Returns:
            The resulting ``ModuleType``.

        Raises:
            ValidationError: If ``module is None``.

        """
        caller = frame.f_back if frame is not None else None
        module_name = caller.f_globals.get("__name__") if caller is not None else None
        module = sys.modules.get(module_name) if isinstance(module_name, str) else None
        if module is None:
            raise flext_core.e.ValidationError(
                flext_core.c.ERR_CONTAINER_CALLER_UNRESOLVED
            )
        return module

    @staticmethod
    def _auto_register_module_factories(
        instance: flext_core.p.Container,
        caller_module: ModuleType,
    ) -> None:
        """Register every ``@d.factory()`` function of a module.

        Each discovered function passes the validated factory record, so a
        non-callable, duplicate or reserved factory raises ``e.ValidationError``.
        """
        module_symbols = vars(caller_module)
        factory_adapter = flext_core.u.type_adapter(
            flext_core.t.TypeForm(flext_core.p.FactoryCallable)
        )
        for factory_name, factory_config in flext_core.u.scan_module(caller_module):
            impl = factory_adapter.validate_python(module_symbols[factory_name])
            _ = instance.factory(factory_config.name, impl)

    def _apply_explicit_bootstrap(
        self,
        registration: flext_core.p.ServiceRegistrationSpec,
    ) -> None:
        """Rebind explicitly selected core collaborators of an existing container."""
        if registration.settings is not None:
            self._config = registration.settings
            _ = self._write(
                flext_core.c.Directory.CONFIG,
                partial(
                    flext_core.m.ServiceRegistration,
                    name=flext_core.c.Directory.CONFIG,
                    service=self._config,
                ),
                internal=True,
            )

        if registration.dispatcher is not None:
            self.register_core_services(dispatcher=registration.dispatcher)
        if registration.context is not None:
            self._context = registration.context
            _ = self._write(
                flext_core.c.FIELD_CONTEXT,
                partial(
                    flext_core.m.ServiceRegistration,
                    name=flext_core.c.FIELD_CONTEXT,
                    service=self._context,
                ),
                internal=True,
            )

    @override
    def clear(self) -> None:
        """Clear every registration and re-register the core services."""
        self._registrations.clear()
        self._config = self._settings_type.fetch_global()
        self.register_core_services()

    @override
    def apply(self, settings: flext_core.t.UserOverridesMapping | None = None) -> Self:
        """Apply user-provided overrides to container configuration.

        Returns:
            The resulting ``Self``.

        """
        if settings is None:
            return self
        self._global_config = flext_core.m.ContainerConfig.model_validate(
            {
                **self._global_config.model_dump(),
                **self._user_overrides.root,
                **settings,
            },
        )
        self._user_overrides.update(settings)
        return self


__all__: MutableSequence[str] = ["FlextContainer"]
