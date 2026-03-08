"""Handler registration and discovery utilities.

Provides handler registration with binding, tracking, and batch operations
for the FLEXT dispatcher system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Annotated, ClassVar, Literal, Self, TypeGuard, override

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, computed_field

from flext_core import (
    FlextContainer,
    FlextDispatcher,
    FlextHandlers,
    c,
    m,
    p,
    r,
    s,
    t,
    u,
)

type RegistrablePlugin = t.RegistrablePlugin
type RegistryBindingKey = str | type[object]


class FlextRegistry(s[bool]):
    """Application-layer registry for CQRS handlers.

    Extends s for automatic infrastructure (config, context,
    container, logging) while providing handler registration and management
    capabilities. The registry pairs message types with handlers, enforces
    idempotent registration, and exposes batch operations that return ``r``
    summaries.

    It delegates to ``FlextDispatcher`` (which implements ``p.CommandBus``)
    for actual handler registration and execution.
    """

    class Summary(m.Value):
        """Aggregated outcome for batch handler registration tracking.

        Tracks successful, skipped, and failed registrations with computed
        success indicators for batch handler operations.
        """

        registered: list[m.HandlerRegistrationDetails] = Field(
            default_factory=lambda: list[m.HandlerRegistrationDetails](),
            description="Successfully registered handlers with registration details.",
        )
        skipped: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Handler identifiers that were skipped (already registered)",
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = Field(
            default_factory=list,
            description="Handler identifiers skipped because they were already registered.",
        )
        errors: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = Field(
            default_factory=list,
            description="Error messages captured for failed handler registrations.",
        )

        @computed_field
        def is_failure(self) -> bool:
            """Indicate whether the batch registration had errors.

            Returns:
                True if any errors occurred, False otherwise

            """
            return bool(self.errors)

        @computed_field
        def is_success(self) -> bool:
            """Indicate whether the batch registration fully succeeded.

            Returns:
                True if no errors occurred, False otherwise

            """
            return not self.errors

    _dispatcher: p.CommandBus | FlextDispatcher = PrivateAttr()
    _registered_keys: set[str] = PrivateAttr(default_factory=lambda: set[str]())
    _class_plugin_storage: ClassVar[MutableMapping[str, t.RegistrablePlugin]] = {}
    _class_registered_keys: ClassVar[set[str]] = set()

    def __init__(
        self,
        dispatcher: p.CommandBus | None = None,
        **data: t.Scalar | m.ConfigMap | Sequence[t.Scalar],
    ) -> None:
        """Initialize the registry with a CommandBus protocol instance.

        Args:
            dispatcher: CommandBus protocol instance (defaults to container DI resolution)
            **data: Additional configuration passed to s

        """
        super().__init__(**data)
        if dispatcher is not None:
            self._dispatcher = dispatcher
        else:
            container_value = FlextContainer.get_global().get("command_bus").unwrap()
            if isinstance(container_value, FlextDispatcher):
                self._dispatcher = container_value
            else:
                msg = f"Expected CommandBus, got {type(container_value).__name__}"
                raise TypeError(msg)

    def __init_subclass__(
        cls, **kwargs: t.Scalar | m.ConfigMap | Sequence[t.Scalar]
    ) -> None:
        """Auto-create per-subclass class-level storage.

        Each subclass gets its OWN storage (not shared with parent or siblings).
        This enables auto-discovery patterns where plugins registered via
        register_plugin(..., scope="class") are visible across all instances of that
        subclass.
        """
        super().__init_subclass__()
        cls._class_plugin_storage = {}
        cls._class_registered_keys = set()

    @classmethod
    def create(
        cls,
        dispatcher: p.CommandBus | None = None,
        *,
        auto_discover_handlers: bool = False,
    ) -> Self:
        """Factory method to create a new FlextRegistry instance.

        This is the preferred way to instantiate FlextRegistry. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Auto-discovery of handlers discovers all functions marked with
        @FlextHandlers.handler() decorator in the calling module and auto-registers them
        with built-in deduplication. This enables zero-config handler
        registration for services with idempotent tracking.

        Args:
            dispatcher: Optional CommandBus instance (defaults to FlextDispatcher)
            auto_discover_handlers: If True, scan calling module for @handler()
                decorated functions and auto-register them with deduplication.
                Default: False.

        Returns:
            FlextRegistry instance with auto-discovered handlers if enabled.

        """
        instance = cls(dispatcher)
        if auto_discover_handlers:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                module_name = caller_globals.get("__name__", "__main__")
                caller_module = sys.modules.get(module_name)
                if caller_module:
                    handlers = FlextHandlers.Discovery.scan_module(caller_module)
                    for _handler_name, handler_func, _handler_config in handlers:
                        if handler_func is not None and callable(handler_func):
                            handler_typed = handler_func
                            if FlextRegistry._is_protocol_handler(handler_typed):
                                _ = instance.register_handler(handler_typed)
        return instance

    @staticmethod
    def _add_registration_error(
        key: str, error: str, summary: FlextRegistry.Summary
    ) -> str:
        """Add registration error to summary.

        Returns:
            str: The error message that was added.

        """
        summary.errors.append(error or f"Failed to register handler '{key}'")
        return error

    @staticmethod
    def _is_protocol_handler(
        value: object,
    ) -> TypeGuard[p.Handler[t.ContainerValue, t.ContainerValue]]:
        return bool(
            hasattr(value, "handle")
            and hasattr(value, "can_handle")
            and hasattr(value, "_protocol_name")
        )

    @staticmethod
    def _resolve_handler_key(
        handler: p.Handler[t.ContainerValue, t.ContainerValue],
    ) -> str:
        """Resolve registration key from handler."""
        handler_id = getattr(handler, "handler_id", None)
        return str(handler_id) if handler_id else handler.__class__.__name__

    @staticmethod
    def _safe_get_handler_mode(value: t.Scalar | BaseModel) -> c.Cqrs.HandlerType:
        """Safely extract and validate handler mode from value."""
        parse_result = u.parse(
            value,
            c.Cqrs.HandlerType,
            default=c.Cqrs.HandlerType.COMMAND,
            case_insensitive=True,
        )
        return (
            parse_result.value
            if parse_result.is_success
            else c.Cqrs.HandlerType.COMMAND
        )

    @staticmethod
    def _safe_get_status(value: c.Cqrs.RegistrationStatus | str) -> c.Cqrs.CommonStatus:
        """Safely extract and validate status from c.Cqrs.RegistrationStatus value."""
        if value == c.Cqrs.RegistrationStatus.ACTIVE:
            return c.Cqrs.CommonStatus.RUNNING
        if value == c.Cqrs.RegistrationStatus.INACTIVE:
            return c.Cqrs.CommonStatus.FAILED
        parse_result = u.parse(
            value,
            c.Cqrs.CommonStatus,
            default=c.Cqrs.CommonStatus.RUNNING,
            case_insensitive=True,
        )
        return (
            parse_result.value
            if parse_result.is_success
            else c.Cqrs.CommonStatus.RUNNING
        )

    @staticmethod
    def _to_dispatcher_handler(
        handler_for_dispatch: p.Handler[t.ContainerValue, t.ContainerValue],
    ) -> t.HandlerLike:
        """Convert handler to dispatcher-compatible callable."""
        handler_ref = handler_for_dispatch

        def _dispatch_wrapper(*args: t.ContainerValue) -> t.ContainerValue | None:
            if args:
                result = handler_ref.handle(args[0])
                return result.value if result.is_success else None
            return None

        message_type_attr = getattr(handler_for_dispatch, "message_type", None)
        if message_type_attr is not None:
            setattr(_dispatch_wrapper, "message_type", message_type_attr)
        event_type_attr = getattr(handler_for_dispatch, "event_type", None)
        if event_type_attr is not None:
            setattr(_dispatch_wrapper, "event_type", event_type_attr)
        can_handle_attr = getattr(handler_for_dispatch, "can_handle", None)
        if can_handle_attr is not None:
            setattr(_dispatch_wrapper, "can_handle", can_handle_attr)
        return _dispatch_wrapper

    @override
    def execute(self) -> r[bool]:
        """Validate registry is properly initialized.

        Returns:
            r[bool]: Success if dispatcher is configured, failure otherwise.

        """
        if not self._dispatcher:
            return r[bool].fail("Dispatcher not configured")
        return r[bool].ok(value=True)

    def get_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: Literal["instance", "class"] = "instance",
    ) -> r[t.RegisterableService | t.RegistrablePlugin]:
        """Get a registered plugin by category and name.

        Returns:
            Success with plugin (RegisterableService) or failure if not found.

        """
        key = f"{category}::{name}"
        if scope == "instance":
            if key not in self._registered_keys:
                available = [
                    k.split("::")[1]
                    for k in self._registered_keys
                    if k.startswith(f"{category}::")
                ]
                return r[t.RegisterableService | t.RegistrablePlugin].fail(
                    f"{category} '{name}' not found. Available: {available}"
                )
            raw_result = self.container.get(key)
            if raw_result.is_failure:
                return r[t.RegisterableService | t.RegistrablePlugin].fail(
                    f"Failed to retrieve {category} '{name}': {raw_result.error}"
                )
            return r[t.RegisterableService | t.RegistrablePlugin].ok(raw_result.value)
        cls = type(self)
        if key not in cls._class_registered_keys:
            available = [
                k.split("::")[1]
                for k in cls._class_registered_keys
                if k.startswith(f"{category}::")
            ]
            return r[t.RegisterableService | t.RegistrablePlugin].fail(
                f"{category} '{name}' not found. Available: {available}"
            )
        return r[t.RegisterableService | t.RegistrablePlugin].ok(
            cls._class_plugin_storage[key]
        )

    def list_plugins(
        self, category: str, *, scope: Literal["instance", "class"] = "instance"
    ) -> r[list[str]]:
        """List all plugins in a category.

        Args:
            category: Plugin category to list

        Returns:
            r[list[str]]: Success with list of plugin names.

        """
        keys = self._registered_keys
        if scope == "class":
            keys = self._class_registered_keys
        plugins = [k.split("::")[1] for k in keys if k.startswith(f"{category}::")]
        return r[list[str]].ok(plugins)

    def register(
        self,
        name: str,
        service: RegistrablePlugin,
        metadata: m.ConfigMap | m.Metadata | None = None,
    ) -> r[bool]:
        """Register a service component with optional metadata.

        Delegates to the container's register method for dependency injection.
        Metadata is currently stored for future use but not actively used.

        Args:
            name: Service name for later retrieval
            service: Service instance to register
            metadata: Optional metadata (dict or m.Metadata)

        Returns:
            r[bool]: Success (True) if registered or failure with error details.

        """
        validated_metadata: m.ConfigMap | None = None
        if metadata is not None:
            raw_metadata: object
            if isinstance(metadata, m.Metadata):
                raw_metadata = metadata.attributes
            else:
                raw_metadata = metadata
            validated_metadata = m.ConfigMap.model_validate(raw_metadata)
        if validated_metadata is not None:
            metadata_dict = m.ConfigMap.model_validate(validated_metadata)
            metadata_keys_str: str = ",".join(metadata_dict.keys())
            self.logger.debug(
                "Registering service with metadata",
                operation="with_service",
                service_name=name,
                has_metadata=True,
                metadata_keys=metadata_keys_str,
            )
        try:
            _ = self.container.register(name, service)
            return r[bool].ok(value=True)
        except ValueError as e:
            error_str = str(e)
            return r[bool].fail(error_str)

    def register_bindings(
        self,
        bindings: Mapping[
            RegistryBindingKey, p.Handler[t.ContainerValue, t.ContainerValue]
        ],
    ) -> r[FlextRegistry.Summary]:
        """Register message-to-handler bindings.

        Args:
            bindings: Map of MessageType -> HandlerInstance

        Returns:
            r[FlextRegistry.Summary]: Batch registration summary

        """
        summary = FlextRegistry.Summary()
        for message_type, handler in bindings.items():
            message_type_name = getattr(message_type, "__name__", str(message_type))
            key = f"binding::{message_type_name}::{handler.__class__.__name__}"
            try:
                try:
                    m.HandlerRegistrationDetails.model_validate({
                        "registration_id": key,
                        "handler_mode": c.Cqrs.HandlerType.COMMAND,
                        "timestamp": "",
                        "status": c.Cqrs.CommonStatus.RUNNING,
                    })
                except ValidationError:
                    _ = self._add_registration_error(
                        key, "Handler validation failed", summary
                    )
                    continue
                handler_for_dispatch: p.Handler[t.ContainerValue, t.ContainerValue] = (
                    handler
                )
                reg_result: r[m.HandlerRegistrationResult]
                if isinstance(self._dispatcher, FlextDispatcher):
                    dispatcher_handler = FlextRegistry._to_dispatcher_handler(
                        handler_for_dispatch
                    )
                    if (
                        not getattr(dispatcher_handler, "message_type", None)
                        and (not getattr(dispatcher_handler, "event_type", None))
                        and (not getattr(dispatcher_handler, "query_type", None))
                        and (not getattr(dispatcher_handler, "command_type", None))
                    ):
                        handler_name = getattr(
                            dispatcher_handler,
                            "__name__",
                            dispatcher_handler.__class__.__name__,
                        )
                        msg = f"Handler {handler_name} must implement p.Handler with self-describing message_type, event_type, query_type, or command_type attribute"
                        raise TypeError(msg)
                    raw_result = self._dispatcher.register_handler(dispatcher_handler)
                    if raw_result.is_failure:
                        reg_result = r[m.HandlerRegistrationResult].fail(
                            raw_result.error or "Unknown error"
                        )
                    else:
                        reg_result = r[m.HandlerRegistrationResult].ok(
                            m.HandlerRegistrationResult(
                                handler_name=key,
                                status=c.Cqrs.CommonStatus.RUNNING,
                                mode="explicit",
                            )
                        )
                else:
                    if not getattr(handler_for_dispatch, "message_type", None):
                        handler_name = getattr(
                            handler_for_dispatch,
                            "__name__",
                            handler_for_dispatch.__class__.__name__,
                        )
                        msg = f"Handler {handler_name} must implement p.Handler with self-describing message_type attribute for protocol-based registration"
                        raise TypeError(msg)
                    handler_callable = FlextRegistry._to_dispatcher_handler(
                        handler_for_dispatch
                    )
                    protocol_result = self._dispatcher.register_handler(
                        handler_callable
                    )
                    if protocol_result.is_failure:
                        reg_result = r[m.HandlerRegistrationResult].fail(
                            protocol_result.error or "Unknown error"
                        )
                    else:
                        reg_result = r[m.HandlerRegistrationResult].ok(
                            m.HandlerRegistrationResult(
                                handler_name=key,
                                status=c.Cqrs.CommonStatus.RUNNING,
                                mode="explicit",
                            )
                        )
                if reg_result.is_success:
                    val = reg_result.value
                    details = self._create_registration_details(
                        m.HandlerRegistrationResult.model_validate(val), key
                    )
                    self._add_successful_registration(key, details, summary)
                else:
                    self._add_registration_error(
                        key, reg_result.error or "Unknown error", summary
                    )
            except Exception as e:
                self._add_registration_error(key, str(e), summary)
        return self._finalize_summary(summary)

    def register_handler(
        self, handler: p.Handler[t.ContainerValue, t.ContainerValue]
    ) -> r[m.HandlerRegistrationDetails]:
        """Register an already-constructed handler instance.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            r[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        try:
            m.HandlerRegistrationDetails.model_validate({
                "registration_id": handler.__class__.__name__,
                "handler_mode": c.Cqrs.HandlerType.COMMAND,
                "timestamp": "",
                "status": c.Cqrs.CommonStatus.RUNNING,
            })
        except ValidationError:
            return r[m.HandlerRegistrationDetails].fail("Handler validation failed")
        handler_name = handler.__class__.__name__
        self._propagate_context(f"register_handler_{handler_name}")
        self.logger.debug(
            "Starting handler registration",
            operation="register_handler",
            handler_name=handler_name,
            handler_type=handler.__class__.__name__,
        )
        key = FlextRegistry._resolve_handler_key(handler)
        if key in self._registered_keys:
            self.logger.debug(
                "Handler already registered, returning existing registration",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
            )
            return r[m.HandlerRegistrationDetails].ok(
                m.HandlerRegistrationDetails(
                    registration_id=key,
                    handler_mode=c.Cqrs.HandlerType.COMMAND,
                    timestamp="",
                    status=c.Cqrs.CommonStatus.RUNNING,
                )
            )
        self.logger.debug(
            "Registering handler with dispatcher",
            operation="register_handler",
            handler_name=handler_name,
            handler_key=key,
        )
        handler_for_dispatch: p.Handler[t.ContainerValue, t.ContainerValue] = handler
        registration_result: r[m.HandlerRegistrationResult]
        if isinstance(self._dispatcher, FlextDispatcher):
            dispatcher_handler = FlextRegistry._to_dispatcher_handler(
                handler_for_dispatch
            )
            raw_result = self._dispatcher.register_handler(dispatcher_handler)
            if raw_result.is_failure:
                registration_result = r[m.HandlerRegistrationResult].fail(
                    raw_result.error or "Unknown error"
                )
            else:
                registration_result = r[m.HandlerRegistrationResult].ok(
                    m.HandlerRegistrationResult(
                        handler_name=key,
                        status=c.Cqrs.CommonStatus.RUNNING,
                        mode="explicit",
                    )
                )
        else:
            handler_callable = FlextRegistry._to_dispatcher_handler(
                handler_for_dispatch
            )
            protocol_result = self._dispatcher.register_handler(handler_callable)
            if protocol_result.is_failure:
                registration_result = r[m.HandlerRegistrationResult].fail(
                    protocol_result.error or "Unknown error"
                )
            else:
                protocol_value = protocol_result.value
                if isinstance(protocol_value, m.HandlerRegistrationResult):
                    normalized_result = protocol_value
                elif isinstance(protocol_value, BaseModel):
                    normalized_result = m.HandlerRegistrationResult.model_validate(
                        protocol_value.model_dump()
                    )
                else:
                    normalized_result = m.HandlerRegistrationResult.model_validate(
                        protocol_value
                    )
                registration_result = r[m.HandlerRegistrationResult].ok(
                    normalized_result
                )
        if registration_result.is_success:
            reg_result = registration_result.value
            reg_details = self._create_registration_details(
                m.HandlerRegistrationResult.model_validate(reg_result), key
            )
            if not reg_details:
                reg_details = m.HandlerRegistrationDetails(
                    registration_id=key,
                    handler_mode=c.Cqrs.HandlerType.COMMAND,
                    timestamp="",
                    status=c.Cqrs.CommonStatus.RUNNING,
                )
            self._registered_keys.add(key)
            self.logger.debug(
                "Handler registered successfully",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
            )
            return r[m.HandlerRegistrationDetails].ok(reg_details)
        error_msg = registration_result.error
        self.logger.warning(
            "Handler registration failed",
            operation="register_handler",
            handler_name=handler_name,
            handler_key=key,
            error=error_msg or "",
            consequence="Handler will not be available for dispatch",
        )
        error_str = registration_result.error or "Unknown error"
        return r[m.HandlerRegistrationDetails].fail(error_str)

    def register_handlers(
        self, handlers: Sequence[p.Handler[t.ContainerValue, t.ContainerValue]]
    ) -> r[FlextRegistry.Summary]:
        """Register multiple handlers in batch.

        Args:
            handlers: Sequence of handler instances to register

        Returns:
            r[FlextRegistry.Summary]: Batch registration summary

        """
        summary = FlextRegistry.Summary()
        for handler in handlers:
            result = self.register_handler(handler)
            key = FlextRegistry._resolve_handler_key(handler)
            if result.is_success:
                registration_details = result.value
                self._add_successful_registration(key, registration_details, summary)
            else:
                self._add_registration_error(
                    key, result.error or "Unknown error", summary
                )
        return self._finalize_summary(summary)

    def register_plugin(
        self,
        category: str,
        name: str,
        plugin: RegistrablePlugin,
        *,
        validate: Callable[[t.RegistrablePlugin], r[bool]] | None = None,
        scope: Literal["instance", "class"] = "instance",
    ) -> r[bool]:
        """Register a plugin with optional validation.

        Generic plugin registration that can be used by subclasses for
        specialized registries (protocols, schemas, transports, auth, etc.)

        Args:
            category: Plugin category (e.g., "protocols", "validators")
            name: Plugin name within the category
            plugin: Plugin instance to register
            validate: Optional validation callable returning r[bool]

        Returns:
            r[bool]: Success if registered, failure with error details.

        """
        if not name:
            return r[bool].fail(f"{category} name cannot be empty")
        if validate:
            try:
                validation_result = validate(plugin)
                if (
                    hasattr(validation_result, "is_failure")
                    and validation_result.is_failure
                ):
                    return r[bool].fail(f"Validation failed: {validation_result.error}")
            except (TypeError, ValueError, RuntimeError) as exc:
                return r[bool].fail(f"Validation error: {exc}")
        key = f"{category}::{name}"
        if scope == "instance":
            if key in self._registered_keys:
                self.logger.debug(
                    "Plugin already registered (idempotent)",
                    category=category,
                    name=name,
                )
                return r[bool].ok(value=True)
            self.container.register(key, plugin)
            self._registered_keys.add(key)
            self.logger.info("Registered %s: %s", category, name)
            return r[bool].ok(value=True)
        cls = type(self)
        if key in cls._class_registered_keys:
            self.logger.debug(
                "Class plugin already registered (idempotent)",
                category=category,
                name=name,
            )
            return r[bool].ok(value=True)
        cls._class_plugin_storage[key] = plugin
        cls._class_registered_keys.add(key)
        self.logger.info("Registered class plugin %s: %s", category, name)
        return r[bool].ok(value=True)

    def unregister_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: Literal["instance", "class"] = "instance",
    ) -> r[bool]:
        """Unregister a plugin.

        Args:
            category: Plugin category
            name: Plugin name

        Returns:
            r[bool]: Success if unregistered, failure if not found.

        """
        key = f"{category}::{name}"
        if scope == "instance":
            if key not in self._registered_keys:
                return r[bool].fail(f"{category} '{name}' not registered")
            self._registered_keys.discard(key)
            self.logger.info("Unregistered %s: %s", category, name)
            return r[bool].ok(value=True)
        cls = type(self)
        if key not in cls._class_registered_keys:
            return r[bool].fail(f"{category} '{name}' not registered")
        del cls._class_plugin_storage[key]
        cls._class_registered_keys.discard(key)
        self.logger.info("Unregistered class plugin %s: %s", category, name)
        return r[bool].ok(value=True)

    def _add_successful_registration(
        self,
        key: str,
        registration: m.HandlerRegistrationDetails,
        summary: FlextRegistry.Summary,
    ) -> None:
        """Add successful registration to summary."""
        self._registered_keys.add(key)
        summary.registered.append(registration)

    def _create_registration_details(
        self, reg_result: m.HandlerRegistrationResult, key: str
    ) -> m.HandlerRegistrationDetails:
        """Create RegistrationDetails from registration result (DRY helper).

        Args:
            reg_result: Registration result model from dispatcher
            key: Handler key for registration_id

        Returns:
            RegistrationDetails: Validated registration details model

        """
        handler_mode_val = reg_result.mode
        handler_mode = c.Cqrs.HandlerType.COMMAND
        if "query" in handler_mode_val.lower():
            handler_mode = c.Cqrs.HandlerType.QUERY
        elif "event" in handler_mode_val.lower():
            handler_mode = c.Cqrs.HandlerType.EVENT
        timestamp = getattr(reg_result, "timestamp", "")
        status = reg_result.status
        return m.HandlerRegistrationDetails(
            registration_id=key,
            handler_mode=FlextRegistry._safe_get_handler_mode(handler_mode),
            timestamp=timestamp,
            status=self._safe_get_status(status),
        )

    def _finalize_summary(
        self, summary: FlextRegistry.Summary
    ) -> r[FlextRegistry.Summary]:
        """Finalize summary based on error state.

        Returns:
            r[FlextRegistry.Summary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return r[FlextRegistry.Summary].fail("; ".join(summary.errors))
        return r[FlextRegistry.Summary].ok(summary)


__all__ = ["FlextRegistry"]
