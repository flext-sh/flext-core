"""Canonical handler registration and identity-preserving plugin lifecycle.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Unpack, override

from flext_core import c, e, m, p, r, s, t, u

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextRegistry(s[bool], m.RegistryOptions):
    """Coordinate the injected runtime without another dispatch or storage engine."""

    def __init_subclass__(cls, **kwargs: Unpack[m.ConfigDict]) -> None:
        """Keep explicitly selected class-scope plugins local to each subclass."""
        super().__init_subclass__(**kwargs)
        cls._class_plugin_storage = {}

    @override
    def _get_runtime(self) -> p.ServiceRuntime:
        return self.registry_runtime

    @property
    def dispatcher(self) -> p.Dispatcher:
        """Dispatcher used by the same runtime as plugin registration."""
        return self.registry_runtime.dispatcher

    @override
    def execute(self) -> p.Result[bool]:
        """Confirm the runtime already validated at construction.

        Returns:
            Result[bool]: The result of the validation.

        """
        return r[bool].ok(value=True)

    def register(self, name: str, service: p.RegisterableService) -> p.Result[bool]:
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
                return r[bool].from_failure(existing)
            previous = existing.value
            if u.registry_same_value(previous, service):
                return r[bool].ok(value=True)
            return r[bool].fail_op(
                "register service in registry",
                c.ERR_CONTAINER_NAME_DUPLICATE.format(name=name),
            )
        try:
            self.container.bind(name, service)
        except e.ValidationError as exc:
            return e.fail_operation("register service in registry", exc)
        return r[bool].ok(value=True)

    def register_handler(
        self,
        handler: p.DispatchableHandler,
    ) -> p.Result[p.RegistrationDetails]:
        """Register one handler through the canonical atomic batch owner.

        Args:
            handler: The dispatchable handler to register.

        Returns:
            A ``p.Result[p.RegistrationDetails]`` containing the registration details or
                an error if registration failed.

        """
        outcome = self.dispatcher.register_handlers((
            m.HandlerRegistration(handler=handler),
        ))
        if outcome.failure:
            return r[p.RegistrationDetails].from_failure(outcome)
        report = outcome.value
        binding = (report.registered or report.skipped)[0]
        return r[p.RegistrationDetails].ok(u.registry_registration_details(binding))

    def register_handlers(
        self,
        handlers: t.SequenceOf[p.DispatchableHandler],
    ) -> p.Result[p.RegistrySummary]:
        """Register every handler atomically without discarding failure metadata.

        Args:
            handlers: A sequence of dispatchable handlers to register.

        Returns:
            A ``p.Result[p.RegistrySummary]`` containing the summary of the registration
                process or an error if registration failed.

        """
        return self._register_batch(
            tuple(m.HandlerRegistration(handler=handler) for handler in handlers)
        )

    def register_bindings(
        self,
        bindings: t.MappingKV[t.RegistryBindingKey, p.DispatchableHandler],
    ) -> p.Result[p.RegistrySummary]:
        """Bind each declared message route through the dispatcher itself.

        Args:
            bindings: A mapping of message types to dispatchable handlers.

        Returns:
            A ``p.Result[p.RegistrySummary]`` containing the summary of the registration
                process or an error if registration failed.

        """
        return self._register_batch(
            tuple(
                m.HandlerRegistration(handler=handler, message_type=message_type)
                for message_type, handler in bindings.items()
            )
        )

    def _register_batch(
        self,
        bindings: t.SequenceOf[p.HandlerRegistration],
    ) -> p.Result[p.RegistrySummary]:
        outcome = self.dispatcher.register_handlers(bindings)
        if outcome.failure:
            return r[p.RegistrySummary].from_failure(outcome)
        report = outcome.value
        return r[p.RegistrySummary].ok(
            m.RegistrySummary.model_validate({
                "registered": [
                    u.registry_registration_details(binding)
                    for binding in report.registered
                ],
                "skipped": tuple(
                    u.registry_registration_name(binding) for binding in report.skipped
                ),
            })
        )

    def register_plugin(
        self,
        category: str,
        name: str,
        plugin: p.RegisterableService,
        *,
        validate: Callable[[p.RegisterableService], p.Result[bool]] | None = None,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[bool]:
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
        key = u.registry_plugin_key(category, name)
        if key.failure:
            return r[bool].from_failure(key)
        if validate is not None:
            validated = u.guard_result(
                partial(validate, plugin),
                catch=c.EXC_RUNTIME_TYPE,
                op_name="validate plugin registration",
            )
            if validated.failure or not validated.value:
                return (
                    validated
                    if validated.failure
                    else r[bool].fail_op("validate plugin registration", name)
                )
        with self._lock:
            if scope == c.RegistrationScope.INSTANCE:
                outcome = self.register(key.value, plugin)
                if outcome.success:
                    self._registered_keys.add(key.value)
                return outcome
            storage = self._class_plugin_storage
            if key.value in storage:
                previous = storage[key.value]
                if u.registry_same_value(previous, plugin):
                    return r[bool].ok(value=True)
                return r[bool].fail_op(
                    "register class plugin",
                    c.ERR_CONTAINER_NAME_DUPLICATE.format(name=key.value),
                )
            storage[key.value] = plugin
        return r[bool].ok(value=True)

    def fetch_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[p.RegisterableService]:
        """Fetch the unchanged collaborator from the selected registry scope.

        Args:
            category: The category of the plugin to fetch.
            name: The name of the plugin to fetch.
            scope: The registration scope, either INSTANCE or CLASS.

        Returns:
            A ``p.Result[p.RegisterableService]`` containing the requested plugin or an
                error if not found.

        """
        key = u.registry_plugin_key(category, name)
        if key.failure:
            return r[p.RegisterableService].from_failure(key)
        with self._lock:
            if scope == c.RegistrationScope.INSTANCE:
                if key.value not in self._registered_keys:
                    return e.fail_not_found(category, name)
                return self.container.resolve(key.value)
            if key.value not in self._class_plugin_storage:
                return e.fail_not_found(category, name)
            return r[p.RegisterableService].ok(self._class_plugin_storage[key.value])

    def list_plugins(
        self,
        category: str,
        *,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[t.StrSequence]:
        """List registered names deterministically in one category.

        Args:
            category: The category of the plugins to list.
            scope: The registration scope, either INSTANCE or CLASS.

        Returns:
            A ``p.Result[t.StrSequence]`` containing the list of registered plugin
                names.

        """
        key_prefix = u.registry_plugin_key(category)
        if key_prefix.failure:
            return r[t.StrSequence].from_failure(key_prefix)
        prefix = key_prefix.value
        with self._lock:
            keys = (
                self._registered_keys
                if scope == c.RegistrationScope.INSTANCE
                else self._class_plugin_storage.keys()
            )
            return r[t.StrSequence].ok(
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
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[bool]:
        """Remove storage before ownership, allowing a new registration afterward.

        Args:
            category: The category of the plugin to unregister.
            name: The name of the plugin to unregister.
            scope: The registration scope, either INSTANCE or CLASS.

        Returns:
            A ``p.Result[bool]`` indicating the success of the unregistration.

        """
        key = u.registry_plugin_key(category, name)
        if key.failure:
            return r[bool].from_failure(key)
        with self._lock:
            if scope == c.RegistrationScope.INSTANCE:
                if key.value not in self._registered_keys:
                    return e.fail_not_found(category, name)
                outcome = self.container.drop(key.value)
                if outcome.failure:
                    return outcome
                self._registered_keys.remove(key.value)
                return outcome
            if key.value not in self._class_plugin_storage:
                return e.fail_not_found(category, name)
            del self._class_plugin_storage[key.value]
        return r[bool].ok(value=True)


__all__: list[str] = ["FlextRegistry"]
