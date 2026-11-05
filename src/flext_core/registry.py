"""Canonical handler registration and identity-preserving plugin lifecycle.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Self, Unpack, override

import flext_core

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextRegistry(flext_core.s[bool], flext_core.m.RegistryOptions):
    """Coordinate the injected runtime without another dispatch or storage engine."""

    def __init_subclass__(cls, **kwargs: Unpack[flext_core.t.ConfigDict]) -> None:
        """Keep explicitly selected class-scope plugins local to each subclass."""
        super().__init_subclass__(**kwargs)
        cls._class_plugin_storage = {}

    @classmethod
    def create(
        cls,
        dispatcher: flext_core.p.Dispatcher | None = None,
        *,
        discovery_module: flext_core.t.RegistryModule | None = None,
    ) -> Self:
        """Build a registry on a fresh service runtime around ``dispatcher``.

        Handlers discovered in ``discovery_module`` are registered on the same
        dispatcher the registry dispatches through.

        Returns:
            The registry sharing one runtime for registration and dispatch.

        """
        elected = dispatcher if dispatcher is not None else flext_core.FlextDispatcher()
        registry = cls(
            registry_runtime=flext_core.FlextContainer.build_service_runtime(
                flext_core.m.RuntimeBootstrapOptions(dispatcher=elected),
            ),
        )
        if discovery_module is not None:
            elected.register_handlers(
                tuple(
                    flext_core.m.HandlerRegistration(
                        handler=handler, message_type=metadata.command
                    )
                    for _, handler, metadata in flext_core.h.Discovery.scan_module(
                        discovery_module
                    )
                )
            ).unwrap()
        return registry

    @override
    def _get_runtime(self) -> flext_core.p.ServiceRuntime:
        return self.registry_runtime

    @property
    def dispatcher(self) -> flext_core.p.Dispatcher:
        """Dispatcher used by the same runtime as plugin registration."""
        return self.registry_runtime.dispatcher

    @override
    def execute(self) -> flext_core.p.Result[bool]:
        """Confirm the runtime already validated at construction.

        Returns:
            Result[bool]: The result of the validation.

        """
        return flext_core.r[bool].ok(value=True)

    def register(
        self, name: str, service: flext_core.p.RegisterableService
    ) -> flext_core.p.Result[bool]:
        """Bind a collaborator unchanged, rejecting conflicting registrations.

        Args:
            name: The name under which to register the service.
            service: The service instance to register.

        Returns:
            A ``p.Result[bool]`` indicating the success of the registration.

        """
        if self.container.has(name):
            existing = self.container.resolve(name)
            if existing.failure:
                return flext_core.r[bool].from_failure(existing)
            previous = existing.value
            if flext_core.u.registry_same_value(previous, service):
                return flext_core.r[bool].ok(value=True)
            return flext_core.r[bool].fail_op(
                "register service in registry",
                flext_core.c.ERR_CONTAINER_NAME_DUPLICATE.format(name=name),
            )
        try:
            self.container.bind(name, service)
        except flext_core.e.ValidationError as exc:
            return flext_core.e.fail_operation("register service in registry", exc)
        return flext_core.r[bool].ok(value=True)

    def register_handler(
        self,
        handler: flext_core.p.DispatchableHandler,
    ) -> flext_core.p.Result[flext_core.p.RegistrationDetails]:
        """Register one handler through the canonical atomic batch owner.

        Args:
            handler: The dispatchable handler to register.

        Returns:
            A ``p.Result[p.RegistrationDetails]`` containing the registration details or
                an error if registration failed.

        """
        outcome = self.dispatcher.register_handlers((
            flext_core.m.HandlerRegistration(handler=handler),
        ))
        if outcome.failure:
            return flext_core.r[flext_core.p.RegistrationDetails].from_failure(outcome)
        report = outcome.value
        binding = (report.registered or report.skipped)[0]
        return flext_core.r[flext_core.p.RegistrationDetails].ok(
            flext_core.u.registry_registration_details(binding)
        )

    def register_handlers(
        self,
        handlers: Sequence[flext_core.p.DispatchableHandler],
    ) -> flext_core.p.Result[flext_core.p.RegistrySummary]:
        """Register every handler atomically without discarding failure metadata.

        Args:
            handlers: A sequence of dispatchable handlers to register.

        Returns:
            A ``p.Result[p.RegistrySummary]`` containing the summary of the registration
                process or an error if registration failed.

        """
        return self._register_batch(
            tuple(
                flext_core.m.HandlerRegistration(handler=handler)
                for handler in handlers
            )
        )

    def register_bindings(
        self,
        bindings: Mapping[
            flext_core.t.RegistryBindingKey, flext_core.p.DispatchableHandler
        ],
    ) -> flext_core.p.Result[flext_core.p.RegistrySummary]:
        """Bind each declared message route through the dispatcher itself.

        Args:
            bindings: A mapping of message types to dispatchable handlers.

        Returns:
            A ``p.Result[p.RegistrySummary]`` containing the summary of the registration
                process or an error if registration failed.

        """
        return self._register_batch(
            tuple(
                flext_core.m.HandlerRegistration(
                    handler=handler, message_type=message_type
                )
                for message_type, handler in bindings.items()
            )
        )

    def _register_batch(
        self,
        bindings: Sequence[flext_core.p.HandlerRegistration],
    ) -> flext_core.p.Result[flext_core.p.RegistrySummary]:
        outcome = self.dispatcher.register_handlers(bindings)
        if outcome.failure:
            return flext_core.r[flext_core.p.RegistrySummary].from_failure(outcome)
        report = outcome.value
        return flext_core.r[flext_core.p.RegistrySummary].ok(
            flext_core.m.RegistrySummary.model_validate({
                "registered": [
                    flext_core.u.registry_registration_details(binding)
                    for binding in report.registered
                ],
                "skipped": tuple(
                    flext_core.u.registry_registration_name(binding)
                    for binding in report.skipped
                ),
            })
        )

    def register_plugin(
        self,
        category: str,
        name: str,
        plugin: flext_core.p.RegisterableService,
        *,
        validate: Callable[
            [flext_core.p.RegisterableService], flext_core.p.Result[bool]
        ]
        | None = None,
        scope: flext_core.c.RegistrationScope = flext_core.c.RegistrationScope.INSTANCE,
    ) -> flext_core.p.Result[bool]:
        """Store the original plugin, never its string or serialized projection.

        Args:
            category: The category of the plugin to register.
            name: The name of the plugin to register.
            plugin: The plugin instance to register.
            validate: An optional callable to validate the plugin before registration.
            scope: The registration scope, either INSTANCE or CLASS.

        Returns:
            A ``p.Result[bool]`` indicating the success of the registration.

        """
        key = flext_core.u.registry_plugin_key(category, name)
        if key.failure:
            return flext_core.r[bool].from_failure(key)
        if validate is not None:
            validated = flext_core.u.guard_result(
                partial(validate, plugin),
                catch=flext_core.c.EXC_RUNTIME_TYPE,
                op_name="validate plugin registration",
            )
            if validated.failure or not validated.value:
                return (
                    validated
                    if validated.failure
                    else flext_core.r[bool].fail_op(
                        "validate plugin registration", name
                    )
                )
        with self._lock:
            if scope == flext_core.c.RegistrationScope.INSTANCE:
                outcome = self.register(key.value, plugin)
                if outcome.success:
                    self._registered_keys.add(key.value)
                return outcome
            storage = self._class_plugin_storage
            if key.value in storage:
                previous = storage[key.value]
                if flext_core.u.registry_same_value(previous, plugin):
                    return flext_core.r[bool].ok(value=True)
                return flext_core.r[bool].fail_op(
                    "register class plugin",
                    flext_core.c.ERR_CONTAINER_NAME_DUPLICATE.format(name=key.value),
                )
            storage[key.value] = plugin
        return flext_core.r[bool].ok(value=True)

    def fetch_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: flext_core.c.RegistrationScope = flext_core.c.RegistrationScope.INSTANCE,
    ) -> flext_core.p.Result[flext_core.p.RegisterableService]:
        """Fetch the unchanged collaborator from the selected registry scope.

        Args:
            category: The category of the plugin to fetch.
            name: The name of the plugin to fetch.
            scope: The registration scope, either INSTANCE or CLASS.

        Returns:
            A ``p.Result[p.RegisterableService]`` containing the requested plugin or an
                error if not found.

        """
        key = flext_core.u.registry_plugin_key(category, name)
        if key.failure:
            return flext_core.r[flext_core.p.RegisterableService].from_failure(key)
        with self._lock:
            if scope == flext_core.c.RegistrationScope.INSTANCE:
                if key.value not in self._registered_keys:
                    return flext_core.e.fail_not_found(category, name)
                return self.container.resolve(key.value)
            if key.value not in self._class_plugin_storage:
                return flext_core.e.fail_not_found(category, name)
            return flext_core.r[flext_core.p.RegisterableService].ok(
                self._class_plugin_storage[key.value]
            )

    def list_plugins(
        self,
        category: str,
        *,
        scope: flext_core.c.RegistrationScope = flext_core.c.RegistrationScope.INSTANCE,
    ) -> flext_core.p.Result[Sequence[str]]:
        """List registered names deterministically in one category.

        Args:
            category: The category of the plugins to list.
            scope: The registration scope, either INSTANCE or CLASS.

        Returns:
            A ``p.Result[Sequence[str]]`` containing the list of registered plugin
                names.

        """
        key_prefix = flext_core.u.registry_plugin_key(category)
        if key_prefix.failure:
            return flext_core.r[Sequence[str]].from_failure(key_prefix)
        prefix = key_prefix.value
        with self._lock:
            keys = (
                self._registered_keys
                if scope == flext_core.c.RegistrationScope.INSTANCE
                else self._class_plugin_storage.keys()
            )
            return flext_core.r[Sequence[str]].ok(
                tuple(
                    sorted(
                        key.removeprefix(prefix)
                        for key in keys
                        if key.startswith(prefix)
                    )
                )
            )

    def unregister_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: flext_core.c.RegistrationScope = flext_core.c.RegistrationScope.INSTANCE,
    ) -> flext_core.p.Result[bool]:
        """Remove storage before ownership, allowing a new registration afterward.

        Args:
            category: The category of the plugin to unregister.
            name: The name of the plugin to unregister.
            scope: The registration scope, either INSTANCE or CLASS.

        Returns:
            A ``p.Result[bool]`` indicating the success of the unregistration.

        """
        key = flext_core.u.registry_plugin_key(category, name)
        if key.failure:
            return flext_core.r[bool].from_failure(key)
        with self._lock:
            if scope == flext_core.c.RegistrationScope.INSTANCE:
                if key.value not in self._registered_keys:
                    return flext_core.e.fail_not_found(category, name)
                outcome = self.container.drop(key.value)
                if outcome.failure:
                    return outcome
                self._registered_keys.remove(key.value)
                return outcome
            if key.value not in self._class_plugin_storage:
                return flext_core.e.fail_not_found(category, name)
            del self._class_plugin_storage[key.value]
        return flext_core.r[bool].ok(value=True)


__all__: list[str] = ["FlextRegistry"]
