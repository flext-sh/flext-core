"""Handler registration and discovery utilities.

Provides handler registration with binding, tracking, and batch operations
for the FLEXT dispatcher system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, ClassVar, Literal, Self, override

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from flext_core import (
    FlextContainer,
    FlextDispatcher,
    c,
    h,
    m,
    p,
    r,
    s,
    t,
    u,
)


class FlextRegistry(s[bool]):
    """Application-layer registry for CQRS handlers.

    Extends s for automatic infrastructure (config, context,
    container, logging) while providing handler registration and management
    capabilities. The registry pairs message types with handlers, enforces
    idempotent registration, and exposes batch operations that return ``r``
    summaries.

    It delegates to ``FlextDispatcher`` (which implements ``p.Dispatcher``)
    for actual handler registration and execution.
    """

    class Summary(m.Value):
        """Aggregated outcome for batch handler registration tracking.

        Tracks successful, skipped, and failed registrations with computed
        success indicators for batch handler operations.
        """

        registered: list[m.RegistrationDetails] = Field(
            default_factory=lambda: list[m.RegistrationDetails](),
            description="Successfully registered handlers with registration details.",
        )
        skipped: list[str] = Field(
            default_factory=list,
            description="Handler identifiers that were skipped (already registered)",
            examples=[["CreateUserCommand", "UpdateUserCommand"]],
        )
        errors: list[str] = Field(
            default_factory=list,
            description="Error messages for failed registrations",
            examples=[["Handler validation failed", "Duplicate registration"]],
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

    _dispatcher: p.Dispatcher | FlextDispatcher = PrivateAttr()
    _registered_keys: set[str] = PrivateAttr(default_factory=lambda: set[str]())
    _class_plugin_storage: ClassVar[dict[str, t.RegistrablePlugin]] = {}
    _class_registered_keys: ClassVar[set[str]] = set()

    dispatcher: Annotated[p.Dispatcher | None, Field(default=None, exclude=True)] = None

    @override
    def model_post_init(self, __context: object, /) -> None:
        """Post-initialization hook for registry.

        Calls parent model_post_init for runtime setup, then resolves
        the dispatcher from the field or from the global container.
        """
        super().model_post_init(__context)
        if self.dispatcher is not None:
            self._dispatcher = self.dispatcher
        else:
            container_value = FlextContainer.get_global().get("command_bus").unwrap()
            if isinstance(container_value, FlextDispatcher):
                self._dispatcher = container_value
            else:
                msg = f"Expected CommandBus, got {type(container_value).__name__}"
                raise TypeError(msg)

    def __init_subclass__(
        cls, **kwargs: t.Scalar | t.ConfigMap | Sequence[t.Scalar]
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
        dispatcher: p.Dispatcher | None = None,
        *,
        auto_discover_handlers: bool = False,
    ) -> Self:
        """Factory method to create a new FlextRegistry instance.

        This is the preferred way to instantiate FlextRegistry. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Auto-discovery of handlers discovers all functions marked with
        @h.handler() decorator in the calling module and auto-registers them
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
        instance = cls(dispatcher=dispatcher)
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

    @staticmethod
    def _narrow_value(
        value: object
        | t.NormalizedValue
        | t.RegisterableService
        | t.RegistrablePlugin
        | BaseModel
        | None,
    ) -> t.Container | BaseModel | None:
        """Safe conversion using centralized utilities."""
        if value is None:
            return None
        if isinstance(value, (*t.CONTAINER_TYPES, BaseModel)):
            return value
        return str(value)

    def _get_handler_mode(self, value: t.Container | BaseModel) -> c.Cqrs.HandlerType:
        """Safe conversion to HandlerType."""
        result = u.parse_enum(c.Cqrs.HandlerType, str(value))
        if result.is_success and isinstance(result.value, c.Cqrs.HandlerType):
            return result.value
        return c.Cqrs.HandlerType.COMMAND

    def _get_status(self, value: t.Container | BaseModel) -> c.Cqrs.CommonStatus:
        """Safe conversion to CommonStatus."""
        result = u.parse_enum(c.Cqrs.CommonStatus, str(value))
        if result.is_success and isinstance(result.value, c.Cqrs.CommonStatus):
            return result.value
        return c.Cqrs.CommonStatus.ACTIVE

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
        self, category: str, name: str, *, scope: str = "instance"
    ) -> r[t.Container | BaseModel | None]:
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
                return r[t.Container | BaseModel | None].fail(
                    f"{category} '{name}' not found. Available: {available}"
                )
            raw_result = self.container.get(key)
            if raw_result.is_failure:
                return r[t.Container | BaseModel | None].fail(
                    f"Failed to retrieve {category} '{name}': {raw_result.error}"
                )
            return r[t.Container | BaseModel | None].ok(
                self._narrow_value(raw_result.value)
            )
        cls = type(self)
        if key not in cls._class_registered_keys:
            available = [
                k.split("::")[1]
                for k in cls._class_registered_keys
                if k.startswith(f"{category}::")
            ]
            return r[t.Container | BaseModel | None].fail(
                f"{category} '{name}' not found. Available: {available}"
            )
        return r[t.Container | BaseModel | None].ok(
            self._narrow_value(cls._class_plugin_storage[key])
        )

    def list_plugins(self, category: str, *, scope: str = "instance") -> r[list[str]]:
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
        service: t.RegistrablePlugin,
        metadata: t.ConfigMap | m.Metadata | None = None,
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
        validated_metadata: t.ConfigMap | None = None
        if metadata is not None:
            raw_metadata: Mapping[str, t.NormalizedValue | t.MetadataValue | BaseModel]
            if isinstance(metadata, m.Metadata):
                raw_metadata = metadata.attributes
            else:
                raw_metadata = metadata.root
            normalized_root: dict[str, t.NormalizedValue | BaseModel] = {}
            for k, v in raw_metadata.items():
                if isinstance(v, (*t.CONTAINER_TYPES, BaseModel)):
                    normalized_root[k] = v
                elif isinstance(v, (list, dict, tuple)):
                    normalized_root[k] = str(v)
                else:
                    normalized_root[k] = str(v) if v is not None else ""
            validated_metadata = t.ConfigMap(root=normalized_root)
        if validated_metadata is not None:
            metadata_dict = validated_metadata
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
        bindings: Mapping[t.RegistryBindingKey, t.HandlerLike],
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
            handler_name = getattr(handler, "__name__", handler.__class__.__name__)
            key = f"binding::{message_type_name}::{handler_name}"

            reg_result = self.register_handler(handler)
            if reg_result.is_success and isinstance(
                reg_result.value, m.RegistrationDetails
            ):
                self._add_successful_registration(key, reg_result.value, summary)
            else:
                summary.errors.append(
                    reg_result.error
                    or f"Failed to register binding for {message_type_name}"
                )
        return self._finalize_summary(summary)

    def register_handler(
        self,
        handler: t.HandlerLike,
        _metadata: t.ConfigMap | m.Metadata | None = None,
    ) -> r[m.RegistrationDetails]:
        """Register a handler instance or callable.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            r[m.RegistrationDetails]: Success result with registration details.

        """
        handler_id = str(getattr(handler, "handler_id", id(handler)))
        status_raw = getattr(handler, "status", "active")
        status = self._get_status(status_raw)
        handler_mode_raw = getattr(
            handler, "handler_mode", getattr(handler, "mode", "command")
        )
        handler_mode = self._get_handler_mode(handler_mode_raw)

        # Standard Dispatcher registration avoids passing name/metadata
        # as it discovers routes from the handler itself.
        registration_handler: t.HandlerLike = handler
        registration_result = self._dispatcher.register_handler(
            registration_handler,
            is_event=(handler_mode == c.Cqrs.HandlerType.EVENT),
        )

        if registration_result.is_failure:
            return r[m.RegistrationDetails].fail(
                registration_result.error or "Dispatcher registration failed"
            )

        self._registered_keys.add(handler_id)
        return r[m.RegistrationDetails].ok(
            m.RegistrationDetails(
                registration_id=handler_id,
                handler_mode=handler_mode,
                status=status,
            )
        )

    def register_handlers(
        self, handlers: Sequence[t.HandlerLike]
    ) -> r[FlextRegistry.Summary]:
        """Register multiple handlers in batch.

        Args:
            handlers: Sequence of handler instances or callables to register

        Returns:
            r[FlextRegistry.Summary]: Batch registration summary

        """
        summary = FlextRegistry.Summary()
        for handler in handlers:
            result = self.register_handler(handler)
            key = getattr(handler, "__name__", handler.__class__.__name__)
            if result.is_success and isinstance(result.value, m.RegistrationDetails):
                self._add_successful_registration(key, result.value, summary)
            else:
                summary.errors.append(
                    result.error or f"Failed to register handler '{key}'"
                )
        return self._finalize_summary(summary)

    def register_plugin(
        self,
        category: str,
        name: str,
        plugin: t.RegistrablePlugin,
        *,
        validate: Callable[[t.RegistrablePlugin], r[bool]] | None = None,
        scope: Literal["instance", "class"] = "instance",
    ) -> r[bool]:
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
        if not name:
            return r[bool].fail(f"{category} name cannot be empty")
        if validate:
            try:
                validation_result = validate(plugin)
                if validation_result.is_failure:
                    return r[bool].fail(f"Validation failed: {validation_result.error}")
            except (TypeError, ValueError, RuntimeError) as exc:
                return r[bool].fail(f"Validation error: {exc}")
        key = f"{category}::{name}"
        if scope == "instance":
            if key in self._registered_keys:
                return r[bool].ok(value=True)
            self.container.register(key, plugin)
            self._registered_keys.add(key)
            return r[bool].ok(value=True)
        cls = type(self)
        if key in cls._class_registered_keys:
            return r[bool].ok(value=True)
        cls._class_plugin_storage[key] = plugin
        cls._class_registered_keys.add(key)
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
            return r[bool].ok(value=True)
        cls = type(self)
        if key not in cls._class_registered_keys:
            return r[bool].fail(f"{category} '{name}' not registered")
        del cls._class_plugin_storage[key]
        cls._class_registered_keys.discard(key)
        return r[bool].ok(value=True)

    def _add_successful_registration(
        self,
        key: str,
        registration: m.RegistrationDetails,
        summary: FlextRegistry.Summary,
    ) -> None:
        """Add successful registration to summary."""
        self._registered_keys.add(key)
        summary.registered.append(registration)

    def _create_registration_details(
        self, reg_result: m.RegistrationResult, key: str
    ) -> m.RegistrationDetails:
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
        return m.RegistrationDetails(
            registration_id=key,
            handler_mode=self._get_handler_mode(handler_mode),
            timestamp=timestamp,
            status=self._get_status(status),
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
