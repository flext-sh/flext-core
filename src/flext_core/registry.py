"""Handler registration and discovery utilities.

Provides handler registration with binding, tracking, and batch operations
for the FLEXT dispatcher system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import (
    Callable,
    MutableMapping,
)
from typing import Annotated, ClassVar, Literal, Self, override

from pydantic import PrivateAttr

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextHandlers as h,
    FlextModels as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextService as s,
    FlextTypes as t,
    FlextUtilities as u,
)


class FlextRegistry(s[bool]):
    """Application-layer registry for CQRS handlers.

    Extends s for automatic infrastructure (settings, context,
    container, logging) while providing handler registration and management
    capabilities. The registry pairs message types with handlers, enforces
    idempotent registration, and exposes batch operations that return ``r``
    summaries.

    It delegates to ``FlextDispatcher`` (which implements ``p.Dispatcher``)
    for actual handler registration and execution.
    """

    _state: m.RegistryState = PrivateAttr(default_factory=lambda: m.RegistryState())
    _class_plugin_storage: ClassVar[MutableMapping[str, t.RegistrablePlugin]] = {}
    _class_registered_keys: ClassVar[set[str]] = set()

    dispatcher: Annotated[
        p.Dispatcher | None,
        m.Field(
            exclude=True,
            description="The dispatcher instance for executing handlers.",
        ),
    ] = None

    @override
    def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
        """Post-initialization hook for registry.

        Initializes dispatcher state without triggering recursive runtime
        build (registry IS part of the runtime triple — building it here
        would recurse via ``build_service_runtime → build_registry``).
        """
        super().model_post_init(__context)
        resolved_dispatcher = (
            self.dispatcher if isinstance(self.dispatcher, p.Dispatcher) else None
        )
        self._state = m.RegistryState(dispatcher=resolved_dispatcher)

    def __init_subclass__(
        cls,
        **kwargs: t.Scalar | m.ConfigMap | t.ScalarList,
    ) -> None:
        """Auto-create per-subclass class-level storage.

        Each subclass gets its OWN storage (not shared with parent or siblings).
        This enables auto-discovery patterns where plugins registered via
        register_plugin(..., scope="class") are visible across all instances of that
        subclass.
        """
        super().__init_subclass__()
        cls._class_plugin_storage = {}  # MutableMapping[str, t.RegistrablePlugin]
        cls._class_registered_keys = set()  # set[str]

    @classmethod
    def create(
        cls,
        dispatcher: p.Dispatcher | None = None,
        *,
        runtime: m.ServiceRuntime | None = None,
        auto_discover_handlers: bool = False,
    ) -> Self:
        """Factory method to create a new FlextRegistry instance.

        This is the preferred way to instantiate FlextRegistry. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Auto-discovery of handlers discovers all functions marked with
        @h.handler() decorator in the calling module and auto-registers them
        with built-in deduplication. This enables zero-settings handler
        registration for services with idempotent tracking.

        Args:
            dispatcher: Optional CommandBus instance (defaults to DSL dispatcher)
            runtime: Optional runtime snapshot whose container/context are reused
            auto_discover_handlers: If True, scan calling module for @handler()
                decorated functions and auto-register them with deduplication.
                Default: False.

        Returns:
            FlextRegistry instance with auto-discovered handlers if enabled.

        """
        if runtime is None:
            instance = cls(dispatcher=dispatcher or u.build_dispatcher())
        else:
            resolved = (
                dispatcher
                if isinstance(dispatcher, p.Dispatcher)
                else runtime.dispatcher
            )
            instance = cls(
                initial_context=runtime.context, dispatcher=resolved
            ).configure_runtime(runtime, dispatcher=resolved)
        if auto_discover_handlers:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                module_name = caller_globals.get("__name__", "__main__")
                caller_module = sys.modules.get(module_name)
                if caller_module:
                    handlers = h.Discovery.scan_module(caller_module)
                    for _handler_name, handler_func, _handler_config in handlers:
                        _ = handler_func
        return instance

    def configure_runtime(
        self,
        runtime: m.ServiceRuntime,
        *,
        dispatcher: p.Dispatcher | None = None,
    ) -> Self:
        """Bind this registry to a pre-built runtime snapshot."""
        resolved_dispatcher = (
            dispatcher if isinstance(dispatcher, p.Dispatcher) else runtime.dispatcher
        )
        self.dispatcher = resolved_dispatcher
        self._state = self._state.model_copy(update={"dispatcher": resolved_dispatcher})
        self._runtime = runtime.model_copy(
            update={
                "dispatcher": resolved_dispatcher,
                "registry": self,
            },
        )
        return self

    def _create_initial_runtime(self) -> m.ServiceRuntime:
        """Build the registry runtime without recursively materializing another registry."""
        return u.build_service_runtime(self, registry=self)

    @staticmethod
    def _narrow_value(
        value: (
            t.JsonValue
            | t.RegisterableService
            | t.RegistrablePlugin
            | m.BaseModel
            | None
        ),
    ) -> t.JsonPayload | None:
        """Safe conversion using centralized utilities."""
        narrowed: t.JsonPayload | None = None
        if value is None:
            narrowed = None
        elif isinstance(value, m.BaseModel):
            narrowed = value
        elif isinstance(
            value, (p.Logger, p.Settings, p.Context, p.Dispatcher)
        ) or callable(value):
            narrowed = str(value)
        else:
            normalized = u.normalize_to_metadata(value)
            narrowed = t.json_value_adapter().validate_python(normalized)
        return narrowed

    @staticmethod
    def _normalize_registration_impl(
        value: t.RegistrablePlugin,
    ) -> t.RegisterableService:
        """Normalize registry payloads to the container bind contract."""
        if callable(value):

            def normalized_callable(
                *args: object,
                **kwargs: object,
            ) -> t.JsonPayload | m.BaseModel | None:
                result = value(*args, **kwargs)
                return FlextRegistry._narrow_value(result)

            return normalized_callable
        return FlextRegistry._narrow_value(value)

    def _get_handler_mode(self, value: t.JsonPayload) -> c.HandlerType:
        """Safe conversion to HandlerType (falls back to COMMAND)."""
        text = str(value)
        if text in c.HandlerType.__members__:
            return c.HandlerType[text]
        try:
            return c.HandlerType(text)
        except ValueError:
            return c.HandlerType.COMMAND

    def _get_status(self, value: t.JsonPayload) -> c.Status:
        """Safe conversion to CommonStatus (falls back to ACTIVE)."""
        text = str(value)
        if text in c.Status.__members__:
            return c.Status[text]
        try:
            return c.Status(text)
        except ValueError:
            return c.Status.ACTIVE

    @override
    def execute(self) -> p.Result[bool]:
        """Validate registry is properly initialized.

        Returns:
            r[bool]: Success if dispatcher is configured, failure otherwise.

        """
        dispatcher = self._state.dispatcher
        if dispatcher is None or (not dispatcher):
            return e.fail_operation("execute registry", c.ERR_DISPATCHER_NOT_CONFIGURED)
        return r[bool].ok(True)

    def _remember_registered_key(self, key: str) -> None:
        """Persist one instance-scoped registry key via immutable model state."""
        self._state = self._state.model_copy(
            update={
                "registered_keys": self._state.registered_keys | frozenset({key}),
            },
        )

    def _forget_registered_key(self, key: str) -> None:
        """Remove one instance-scoped registry key via immutable model state."""
        self._state = self._state.model_copy(
            update={
                "registered_keys": frozenset(
                    existing_key
                    for existing_key in self._state.registered_keys
                    if existing_key != key
                ),
            },
        )

    def fetch_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[t.JsonPayload | None]:
        """Get a registered plugin by category and name.

        Returns:
            Success with plugin (RegisterableService) or failure if not found.

        """
        key = f"{category}::{name}"
        if scope == c.RegistrationScope.INSTANCE:
            if key not in self._state.registered_keys:
                return e.fail_not_found(category, name)
            raw_result = self.container.resolve(key)
            if raw_result.failure:
                return e.fail_operation(
                    f"retrieve {category} '{name}'",
                    raw_result.error or c.CQRS_OPERATION_FAILED,
                )
            return r[t.JsonPayload | None].ok(self._narrow_value(raw_result.value))
        cls = type(self)
        if key not in cls._class_registered_keys:
            return e.fail_not_found(category, name)
        return r[t.JsonPayload | None].ok(
            self._narrow_value(cls._class_plugin_storage[key]),
        )

    def list_plugins(
        self,
        category: str,
        *,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[t.StrSequence]:
        """List all plugins in a category.

        Args:
            category: Plugin category to list

        Returns:
            r[t.StrSequence]: Success with list of plugin names.

        """
        keys = self._state.registered_keys
        if scope == c.RegistrationScope.CLASS:
            keys = self._class_registered_keys
        plugins = [k.split("::")[1] for k in keys if k.startswith(f"{category}::")]
        return r[t.StrSequence].ok(plugins)

    def register(
        self,
        name: str,
        service: t.RegistrablePlugin,
    ) -> p.Result[bool]:
        """Register a service component in the runtime container.

        Args:
            name: Service name for later retrieval
            service: Service instance to register

        Returns:
            r[bool]: Success (True) if registered or failure with error details.

        """
        was_registered = self.container.has(name)
        normalized_service = self._normalize_registration_impl(service)
        _ = self.container.bind(name, normalized_service)
        if was_registered or self.container.has(name):
            return r[bool].ok(True)
        return r[bool].fail_op(
            "register service in registry",
            f"Service '{name}' was not registered",
        )

    def register_bindings(
        self,
        bindings: t.MappingKV[t.RegistryBindingKey, t.DispatchableHandler],
    ) -> p.Result[m.RegistrySummary]:
        """Register message-to-handler bindings.

        Args:
            bindings: Map of MessageType -> HandlerInstance

        Returns:
            r[m.RegistrySummary]: Batch registration summary

        """
        summary = m.RegistrySummary()
        for message_type, handler in bindings.items():
            message_type_name = getattr(message_type, "__name__", str(message_type))
            handler_name = getattr(handler, "__name__", handler.__class__.__name__)
            key = f"binding::{message_type_name}::{handler_name}"

            reg_result = self.register_handler(handler)
            if reg_result.success:
                self._add_successful_registration(key, reg_result.value, summary)
            else:
                summary.errors.append(
                    reg_result.error
                    or f"Failed to register binding for {message_type_name}",
                )
        return self._finalize_summary(summary)

    def register_handler(
        self,
        handler: t.DispatchableHandler,
    ) -> p.Result[m.RegistrationDetails]:
        """Register a handler instance or callable.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            r[m.RegistrationDetails]: Success result with registration details.

        """
        handler_id = str(getattr(handler, "handler_id", id(handler)))
        status_raw: t.JsonPayload = getattr(
            handler,
            c.FIELD_STATUS,
            c.Status.ACTIVE,
        )
        status = self._get_status(status_raw)
        handler_mode_raw: t.JsonPayload = getattr(
            handler,
            c.FIELD_HANDLER_MODE,
            getattr(handler, "mode", c.HandlerType.COMMAND),
        )
        handler_mode = self._get_handler_mode(handler_mode_raw)

        # Standard Dispatcher registration avoids passing name/metadata
        # as it discovers routes from the handler itself.
        registration_handler: t.DispatchableHandler = handler
        dispatcher = self._state.dispatcher
        if dispatcher is None:
            return e.fail_operation(
                "register handler in registry",
                c.ERR_DISPATCHER_NOT_CONFIGURED,
            )
        registration_result = dispatcher.register_handler(
            registration_handler,
            is_event=(handler_mode == c.HandlerType.EVENT),
        )

        if registration_result.failure:
            return e.fail_operation(
                "register handler in dispatcher",
                registration_result.error or c.ERR_HANDLER_FAILED,
            )

        self._remember_registered_key(handler_id)
        return r[m.RegistrationDetails].ok(
            m.RegistrationDetails(
                registration_id=handler_id,
                handler_mode=handler_mode,
                status=status,
            ),
        )

    def register_handlers(
        self,
        handlers: t.SequenceOf[t.DispatchableHandler],
    ) -> p.Result[m.RegistrySummary]:
        """Register multiple handlers in batch.

        Args:
            handlers: Sequence of handler instances or callables to register

        Returns:
            r[m.RegistrySummary]: Batch registration summary

        """
        summary = m.RegistrySummary()
        for handler in handlers:
            result = self.register_handler(handler)
            key = getattr(handler, "__name__", handler.__class__.__name__)
            if result.success:
                self._add_successful_registration(key, result.value, summary)
            else:
                summary.errors.append(
                    result.error or f"Failed to register handler '{key}'",
                )
        return self._finalize_summary(summary)

    def register_plugin(
        self,
        category: str,
        name: str,
        plugin: t.RegistrablePlugin,
        *,
        validate: Callable[[t.RegistrablePlugin], r[bool]] | None = None,
        scope: Literal[
            c.RegistrationScope.INSTANCE,
            c.RegistrationScope.CLASS,
        ] = c.RegistrationScope.INSTANCE,
    ) -> p.Result[bool]:
        """Register a plugin with optional validation.

        Args:
            category: Plugin category (e.g., "protocols", "validators")
            name: Plugin name within the category
            plugin: Plugin instance to register
            validate: Optional validation callable returning r[bool]
            scope: Registration scope ("instance" or "class")

        Returns:
            r[bool]: Success if registered, failure with error details.

        """
        result: p.Result[bool] = r[bool].ok(True)
        if not name:
            params = m.RegistryPluginParams(
                category=category,
                name=name,
                scope=scope,
            )
            result = e.fail_validation(
                m.ValidationErrorParams(field="name", value=name),
                error=e.render_template(
                    c.ERR_REGISTRY_CATEGORY_NAME_CANNOT_BE_EMPTY,
                    category=category,
                    params=params,
                ),
            )
        if result.success and validate:
            try:
                validation_result = validate(plugin)
                if validation_result.failure:
                    result = e.fail_operation(
                        "validate plugin registration",
                        validation_result.error or c.ERR_VALIDATION_FAILED,
                    )
            except (TypeError, ValueError, RuntimeError) as exc:
                result = e.fail_operation("validate plugin registration", exc)
        if result.success:
            key = f"{category}::{name}"
            if scope == c.RegistrationScope.INSTANCE:
                if key not in self._state.registered_keys:
                    normalized_plugin = self._normalize_registration_impl(plugin)
                    self.container.bind(key, normalized_plugin)
                    self._remember_registered_key(key)
            else:
                cls = type(self)
                if key not in cls._class_registered_keys:
                    cls._class_plugin_storage[key] = plugin
                    cls._class_registered_keys.add(key)
        return result

    def unregister_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: Literal[
            c.RegistrationScope.INSTANCE,
            c.RegistrationScope.CLASS,
        ] = c.RegistrationScope.INSTANCE,
    ) -> p.Result[bool]:
        """Unregister a plugin.

        Args:
            category: Plugin category
            name: Plugin name

        Returns:
            r[bool]: Success if unregistered, failure if not found.

        """
        key = f"{category}::{name}"
        if scope == c.RegistrationScope.INSTANCE:
            if key not in self._state.registered_keys:
                return e.fail_not_found(category, name)
            self._forget_registered_key(key)
            return r[bool].ok(True)
        cls = type(self)
        if key not in cls._class_registered_keys:
            return e.fail_not_found(category, name)
        del cls._class_plugin_storage[key]
        cls._class_registered_keys.discard(key)
        return r[bool].ok(True)

    def _add_successful_registration(
        self,
        key: str,
        registration: m.RegistrationDetails,
        summary: m.RegistrySummary,
    ) -> None:
        """Add successful registration to summary."""
        self._remember_registered_key(key)
        summary.registered.append(registration)

    def _finalize_summary(
        self,
        summary: m.RegistrySummary,
    ) -> p.Result[m.RegistrySummary]:
        """Finalize summary based on error state.

        Returns:
            r[m.RegistrySummary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return e.fail_operation(
                "finalize registry summary", "; ".join(summary.errors)
            )
        return r[m.RegistrySummary].ok(summary)


__all__: list[str] = ["FlextRegistry"]
