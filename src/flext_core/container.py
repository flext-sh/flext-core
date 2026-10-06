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
import threading
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Self, TypeGuard, cast, overload, override

from flext_core import FlextSettings, FlextUtilitiesLogging, c, e, m, p, r, t, u
from flext_core._models.flext_context import FlextContext

# NOTE (multi-agent): mro-i6nq.12 — the concrete public facade remains the
# runtime implementation; p.ContainerType is only its structural contract.
if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from types import FrameType, ModuleType


class FlextContainer(p.Container):
    """Process-wide registry of services, factories and resources.

    One mapping holds every registration; the core runtime names
    (``c.CONTAINER_RESERVED_NAMES``) are written only by the container itself
    and stay out of ``has``/``names``/``drop``. Scoped containers built with
    ``scope`` inherit the public registrations and bind their own core
    services to their own settings and context.
    """

    _global_instance: Self | None = None

    _global_lock: threading.RLock = threading.RLock()

    _settings_type: ClassVar[p.SettingsType] = FlextSettings

    _context_type: ClassVar[p.ContextType] = FlextContext

    _context: p.Context

    _config: p.Settings

    _user_overrides: m.ConfigMap

    _initialized: bool = False

    _registrations: MutableMapping[
        str,
        m.ServiceRegistration | m.FactoryRegistration | m.ResourceRegistration,
    ]

    _global_config: m.ContainerConfig

    def __new__(cls, *, registration: m.ServiceRegistrationSpec | None = None) -> Self:
        """Create or return the global singleton instance."""
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
        registration: m.ServiceRegistrationSpec | None = None,
    ) -> None:
        """Initialize the singleton once; later calls apply the explicit spec."""
        if not self._initialized:
            self.initialize_registrations(registration=registration)
        elif registration is not None:
            self._apply_explicit_bootstrap(registration)
            self._write_spec(registration)

    @property
    @override
    def settings(self) -> p.Settings:
        """Configuration bound to this container."""
        return self._config

    @property
    @override
    def context(self) -> p.Context:
        """Execution context bound to this container."""
        return self._context

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._global_lock:
            cls._global_instance = None

    @override
    def logger(
        self,
        module_name: str,
        *,
        service_name: str | None = None,
        service_version: str | None = None,
        correlation_id: str | None = None,
    ) -> p.Logger:
        """Create a module logger for the specified runtime scope.

        Returns:
            The resulting ``p.Logger``.

        """
        _ = service_name, service_version, correlation_id
        logger: p.Logger = FlextUtilitiesLogging.fetch_logger(module_name)
        return logger

    @staticmethod
    def _matches_service_type[T: t.RegisterableService](
        value: t.RegisterableService,
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
            m.ServiceRegistration | m.FactoryRegistration | m.ResourceRegistration,
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
            raise e.ValidationError(c.ERR_CONTAINER_NAME_EMPTY)
        reserved = name in c.CONTAINER_RESERVED_NAMES
        if reserved != internal:
            raise e.ValidationError(c.ERR_CONTAINER_NAME_RESERVED.format(name=name))
        if not internal and name in self._registrations:
            raise e.ValidationError(c.ERR_CONTAINER_NAME_DUPLICATE.format(name=name))
        try:
            record = build()
        except ValueError as exc:
            raise e.ValidationError(
                c.ERR_CONTAINER_REGISTRATION_FAILED.format(name=name, reason=exc),
            ) from exc
        self._registrations[name] = record
        return self

    def _write_spec(self, spec: m.ServiceRegistrationSpec) -> None:
        """Write every service, factory and resource declared by a spec."""
        for name, service in (spec.services or {}).items():
            _ = self.bind(name, service)
        for name, factory in (spec.factories or {}).items():
            _ = self.factory(name, factory)
        for name, resource in (spec.resources or {}).items():
            _ = self.resource(name, resource)

    @override
    def bind(self, name: str, impl: t.RegisterableService) -> Self:
        """Bind a concrete service instance or value.

        Returns:
            The resulting ``Self``.

        """
        return self._write(
            name,
            partial(m.ServiceRegistration, name=name, service=impl),
        )

    @override
    def factory(self, name: str, impl: t.FactoryCallable) -> Self:
        """Bind a factory callable invoked on every resolve.

        Returns:
            The resulting ``Self``.

        """
        return self._write(
            name,
            partial(m.FactoryRegistration, name=name, factory=impl),
        )

    @override
    def resource(self, name: str, impl: t.ResourceCallable) -> Self:
        """Bind a resource factory invoked on every resolve.

        Returns:
            The resulting ``Self``.

        """
        return self._write(
            name,
            partial(m.ResourceRegistration, name=name, factory=impl),
        )

    @staticmethod
    def _resolve_callable(
        callable_obj: t.FactoryCallable,
        kind: str,
    ) -> p.Result[t.RegisterableService]:
        """Invoke a factory/resource callable and validate what it produced.

        Returns:
            The resulting ``p.Result[t.RegisterableService]``.

        """
        try:
            resolved = callable_obj()
            _ = u.normalize_registerable_service(resolved)
        except c.EXC_BROAD_RUNTIME as exc:
            return r[t.RegisterableService].from_result(
                e.fail_operation(
                    f"resolve {kind}",
                    exc,
                    result_type=r[t.RegisterableService],
                ),
            )
        return r[t.RegisterableService].ok(resolved)

    @overload
    def resolve[T: t.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T],
    ) -> p.Result[T]: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        type_cls: None = None,
    ) -> p.Result[t.RegisterableService]: ...

    @override
    def resolve[T: t.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T] | None = None,
    ) -> p.Result[T] | p.Result[t.RegisterableService]:
        """Resolve a registered service, factory or resource by name.

        Returns:
            The resulting ``p.Result[T] | p.Result[t.RegisterableService]``.

        """
        match self._registrations.get(name):
            case None:
                return r[t.RegisterableService].from_result(
                    e.fail_not_found("service", name),
                )
            case m.ServiceRegistration() as record:
                result = r[t.RegisterableService].ok(record.service)
            case m.FactoryRegistration() as record:
                result = self._resolve_callable(record.factory, "factory")
            case record:
                result = self._resolve_callable(record.factory, "resource")
        if type_cls is None or result.failure:
            return result
        if self._matches_service_type(result.value, type_cls):
            return r[T].ok(result.value)
        return r[T].from_result(
            e.fail_type_mismatch(
                type_cls.__name__,
                type(result.value).__name__,
                result_type=r[T],
            ),
        )

    @override
    def snapshot(self) -> m.ConfigMap:
        """Return the merged settings exposed by this container.

        Returns:
            The merged settings exposed by this container.

        """
        config_dict = self._global_config.model_dump()
        return m.ConfigMap(
            root={k: u.normalize_to_container(v) for k, v in config_dict.items()},
        )

    @override
    def has(self, name: str) -> bool:
        """Return whether a public service, factory, or resource is registered.

        Returns:
            Whether a public service, factory, or resource is registered.

        """
        return name in self._registrations and name not in c.CONTAINER_RESERVED_NAMES

    @override
    def names(self) -> t.StrSequence:
        """List the public services, factories, and resources.

        Returns:
            The resulting ``t.StrSequence``.

        """
        return [name for name in self._registrations if self.has(name)]

    def initialize_registrations(
        self,
        *,
        registration: m.ServiceRegistrationSpec | None = None,
    ) -> None:
        """Reset the registrations from a spec and register the core services."""
        spec = registration or m.ServiceRegistrationSpec()
        self._registrations = {}
        self._global_config = spec.container_config or m.ContainerConfig()
        overrides = spec.user_overrides
        self._user_overrides = (
            overrides
            if isinstance(overrides, m.ConfigMap)
            else m.ConfigMap(
                root={
                    k: list(v)
                    if isinstance(v, Sequence) and not isinstance(v, str | bytes)
                    else v
                    for k, v in (overrides or {}).items()
                },
            )
        )
        self._config = (
            spec.settings.clone()
            if spec.settings is not None
            else self._settings_type.fetch_global()
        )
        self._context = (
            spec.context if spec.context is not None else self._context_type.create()
        )
        self._write_spec(spec)
        self.register_core_services()
        self._initialized = True

    @override
    def register_core_services(self) -> None:
        """Register the reserved core services that are not registered yet."""
        core: tuple[
            tuple[str, Callable[[], m.ServiceRegistration | m.FactoryRegistration]],
            ...,
        ] = (
            (
                c.Directory.CONFIG,
                partial(
                    m.ServiceRegistration,
                    name=c.Directory.CONFIG,
                    service=self._config,
                ),
            ),
            (
                c.ServiceName.LOGGER,
                partial(
                    m.FactoryRegistration,
                    name=c.ServiceName.LOGGER,
                    factory=partial(u.fetch_logger, c.LOGGER_NAME_FLEXT_CORE),
                ),
            ),
            (
                c.FIELD_CONTEXT,
                partial(
                    m.ServiceRegistration,
                    name=c.FIELD_CONTEXT,
                    service=self._context,
                ),
            ),
            (
                c.ServiceName.COMMAND_BUS,
                lambda: m.ServiceRegistration(
                    name=c.ServiceName.COMMAND_BUS,
                    service=u.build_dispatcher(),
                ),
            ),
        )
        for name, build in core:
            if name not in self._registrations:
                _ = self._write(name, build, internal=True)

    @override
    def scope(
        self,
        *,
        subproject: str | None = None,
        registration: m.ServiceRegistrationSpec | None = None,
    ) -> Self:
        """Create an isolated scope inheriting the public registrations.

        Registrations declared by ``registration`` override inherited names;
        the scope binds its own core services to its own settings and context.

        Returns:
            The resulting ``Self``.

        """
        spec = registration or m.ServiceRegistrationSpec()
        settings_source = spec.settings if spec.settings is not None else self._config
        scoped_context = self.context.clone() if spec.context is None else spec.context
        if subproject:
            _ = scoped_context.set("subproject", subproject)
        inherited = {name: self._registrations[name] for name in self.names()}
        scoped = u.create_instance(self.__class__)
        scoped.initialize_registrations(
            registration=m.ServiceRegistrationSpec(
                settings=settings_source.clone(),
                context=scoped_context,
                services={
                    **{
                        name: record.service
                        for name, record in inherited.items()
                        if isinstance(record, m.ServiceRegistration)
                    },
                    **(spec.services or {}),
                },
                factories={
                    **{
                        name: record.factory
                        for name, record in inherited.items()
                        if isinstance(record, m.FactoryRegistration)
                    },
                    **(spec.factories or {}),
                },
                resources={
                    **{
                        name: record.factory
                        for name, record in inherited.items()
                        if isinstance(record, m.ResourceRegistration)
                    },
                    **(spec.resources or {}),
                },
                user_overrides=self._user_overrides.model_copy(),
                container_config=self._global_config.model_copy(deep=True),
            ),
        )
        return scoped

    @override
    def drop(self, name: str) -> p.Result[bool]:
        """Remove a public service, factory, or resource registration by name.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        if not self.has(name):
            return r[bool].from_result(
                e.fail_not_found("service", name, result_type=r[bool]),
            )
        del self._registrations[name]
        return r[bool].ok(value=True)

    @override
    def dispatcher(self) -> p.Result[p.Dispatcher]:
        """Resolve the canonical dispatcher / command bus.

        Returns:
            The resulting ``p.Result[p.Dispatcher]``.

        """
        result = self.resolve(c.ServiceName.COMMAND_BUS)
        if result.failure:
            return r[p.Dispatcher].from_result(
                e.fail_not_found(
                    "dispatcher",
                    c.ServiceName.COMMAND_BUS,
                    result_type=r[p.Dispatcher],
                ),
            )
        if isinstance(result.value, p.Dispatcher):
            return r[p.Dispatcher].ok(result.value)
        return r[p.Dispatcher].from_result(
            e.fail_type_mismatch(
                "dispatcher",
                u.type_name(result.value),
                result_type=r[p.Dispatcher],
            ),
        )

    @classmethod
    def shared(
        cls,
        *,
        settings: p.Settings | None = None,
        context: p.Context | None = None,
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
                m.ServiceRegistrationSpec(settings=settings, context=context),
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
            raise e.ValidationError(c.ERR_CONTAINER_CALLER_UNRESOLVED)
        return module

    @staticmethod
    def _auto_register_module_factories(
        instance: p.Container,
        caller_module: ModuleType,
    ) -> None:
        """Register every ``@d.factory()`` function of a module.

        Each discovered function passes the validated factory record, so a
        non-callable, duplicate or reserved factory raises ``e.ValidationError``.
        """
        module_symbols = vars(caller_module)
        for factory_name, factory_config in u.scan_module(caller_module):
            impl = cast("t.FactoryCallable", module_symbols[factory_name])
            _ = instance.factory(factory_config.name, impl)

    def _apply_explicit_bootstrap(
        self,
        registration: m.ServiceRegistrationSpec,
    ) -> None:
        """Rebind the core settings and context of an existing container."""
        if registration.settings is not None:
            self._config = registration.settings
            _ = self._write(
                c.Directory.CONFIG,
                partial(
                    m.ServiceRegistration,
                    name=c.Directory.CONFIG,
                    service=self._config,
                ),
                internal=True,
            )
        if registration.context is not None:
            self._context = registration.context
            _ = self._write(
                c.FIELD_CONTEXT,
                partial(
                    m.ServiceRegistration,
                    name=c.FIELD_CONTEXT,
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
    def apply(self, settings: t.UserOverridesMapping | None = None) -> Self:
        """Apply user-provided overrides to container configuration.

        Returns:
            The resulting ``Self``.

        """
        if settings is None:
            return self
        merged = self._user_overrides.model_copy()
        merged.update({k: u.normalize_to_container(v) for k, v in settings.items()})
        self._user_overrides = merged
        self._global_config = m.ContainerConfig.model_validate(
            self._global_config.model_dump() | dict(merged),
        )
        return self


__all__: t.MutableSequenceOf[str] = ["FlextContainer"]
