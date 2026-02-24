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
from pathlib import Path
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from flext_core._models.entity import FlextModelsEntity
from flext_core.constants import c
from flext_core.dispatcher import FlextDispatcher
from flext_core.handlers import FlextHandlers
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import t
from flext_core.utilities import u

# Use centralized version from utilities
_to_general_value_type = u.Conversion.to_general_value_type

type RegistryHandler = Callable[..., object] | BaseModel
type RegistryBindingKey = str | type[object]


class FlextRegistry(FlextService[bool]):
    """Application-layer registry for CQRS handlers.

    Extends FlextService for automatic infrastructure (config, context,
    container, logging) while providing handler registration and management
    capabilities. The registry pairs message types with handlers, enforces
    idempotent registration, and exposes batch operations that return ``r``
    summaries.

    It delegates to ``FlextDispatcher`` (which implements ``p.CommandBus``)
    for actual handler registration and execution.
    """

    class Summary(FlextModelsEntity.Value):
        """Aggregated outcome for batch handler registration tracking.

        Tracks successful, skipped, and failed registrations with computed
        success indicators for batch handler operations.
        """

        registered: Annotated[
            list[m.HandlerRegistrationDetails],
            Field(
                default_factory=list,
                description="Successfully registered handlers with registration details",
            ),
        ] = Field(default_factory=list)
        skipped: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Handler identifiers that were skipped (already registered)",
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = Field(default_factory=list)
        errors: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = Field(default_factory=list)

        @computed_field
        def is_success(self) -> bool:
            """Indicate whether the batch registration fully succeeded.

            Returns:
                True if no errors occurred, False otherwise

            """
            return not self.errors

        @computed_field
        def is_failure(self) -> bool:
            """Indicate whether the batch registration had errors.

            Returns:
                True if any errors occurred, False otherwise

            """
            return bool(self.errors)

        @computed_field
        def successful_registrations(self) -> int:
            """Number of successful registrations.

            Returns:
                Count of successfully registered handlers

            """
            return len(self.registered)

        @computed_field
        def failed_registrations(self) -> int:
            """Number of failed registrations.

            Returns:
                Count of failed registration attempts

            """
            return len(self.errors)

        def __bool__(self) -> bool:
            """Boolean representation - False when there are errors, True otherwise.

            Returns:
                False if any errors occurred (is_failure), True if successful

            """
            return not self.errors

    # Private attributes using Pydantic v2 PrivateAttr pattern
    _dispatcher: p.CommandBus | FlextDispatcher = PrivateAttr()
    _registered_keys: set[str] = PrivateAttr(default_factory=set)

    # Class-level storage declarations (created per-subclass via __init_subclass__)
    # These ClassVars are automatically created for each subclass to ensure
    # per-subclass isolation (FlextApiRegistry, FlextAuthRegistry, FlextLdifServer
    # each get their own storage, not inherited from parent)
    # Uses t.RegistrablePlugin for type-safe plugin storage (includes callables)
    _class_plugin_storage: ClassVar[MutableMapping[str, t.RegistrablePlugin]] = {}
    _class_registered_keys: ClassVar[set[str]] = set()

    def __init_subclass__(cls, **kwargs: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue]) -> None:
        """Auto-create per-subclass class-level storage.

        Each subclass gets its OWN storage (not shared with parent or siblings).
        This enables auto-discovery patterns where plugins registered via
        register_class_plugin() are visible across all instances of that subclass.
        """
        # Filter kwargs - BaseModel.__init_subclass__ expects specific types
        # but we accept broader types for registry customization
        super().__init_subclass__()
        # Each subclass gets its OWN storage (not inherited from parent)
        cls._class_plugin_storage = {}
        cls._class_registered_keys = set()

    def __init__(
        self,
        dispatcher: p.CommandBus | FlextDispatcher | None = None,
        **data: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue],
    ) -> None:
        """Initialize the registry with a CommandBus protocol instance.

        Args:
            dispatcher: CommandBus or FlextDispatcher instance (defaults to creating FlextDispatcher)
            **data: Additional configuration passed to FlextService

        """
        super().__init__(**data)

        # Create dispatcher instance if not provided
        self._dispatcher = dispatcher if dispatcher is not None else FlextDispatcher()

    def execute(self) -> r[bool]:
        """Validate registry is properly initialized.

        Returns:
            r[bool]: Success if dispatcher is configured, failure otherwise.

        """
        if not self._dispatcher:
            return r[bool].fail("Dispatcher not configured")
        return r[bool].ok(value=True)

    # ------------------------------------------------------------------
    # Factory Method with Auto-Discovery
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        dispatcher: p.CommandBus | FlextDispatcher | None = None,
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
            # Get the caller's frame to discover handlers in calling module
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                # Get module name from globals
                module_name = caller_globals.get("__name__", "__main__")
                # Get module object from globals
                caller_module = sys.modules.get(module_name)
                if caller_module:
                    # Scan module for handler-decorated functions
                    handlers = FlextHandlers.Discovery.scan_module(caller_module)
                    for _handler_name, handler_func, _handler_config in handlers:
                        # Get actual handler from module
                        # Check if handler_func is not None before checking callable
                        if handler_func is not None and callable(handler_func):
                            # Register handler with deduplication built-in
                            # Deduplication happens in register_handler() via _registered_keys
                            # Type narrowing: handler_func is callable and not None here
                            handler_typed = handler_func
                            _ = instance.register_handler(handler_typed)

        return instance

    @staticmethod
    def _safe_get_handler_mode(
        value: t.ScalarValue | BaseModel,
    ) -> c.Cqrs.HandlerType:
        """Safely extract and validate handler mode from value."""
        # Use u.parse() for cleaner enum parsing
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
    def _safe_get_status(
        value: c.Cqrs.RegistrationStatus | str,
    ) -> c.Cqrs.CommonStatus:
        """Safely extract and validate status from c.Cqrs.RegistrationStatus value."""
        # Handle special case: RegistrationStatus.ACTIVE -> CommonStatus.RUNNING
        if value == c.Cqrs.RegistrationStatus.ACTIVE:
            return c.Cqrs.CommonStatus.RUNNING
        if value == c.Cqrs.RegistrationStatus.INACTIVE:
            return c.Cqrs.CommonStatus.FAILED
        # Use u.parse() for cleaner enum parsing
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

    def _create_registration_details(
        self,
        reg_result: m.HandlerRegistrationResult,
        key: str,
    ) -> m.HandlerRegistrationDetails:
        """Create RegistrationDetails from registration result (DRY helper).

        Args:
            reg_result: Registration result model from dispatcher
            key: Handler key for registration_id

        Returns:
            RegistrationDetails: Validated registration details model

        """
        # Map fields from result to details
        handler_mode_val = reg_result.mode
        # Map internal mode strings to Cqrs.HandlerType if possible
        handler_mode = c.Cqrs.HandlerType.COMMAND
        # If mode matches known types, use it (simplified mapping)
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_handler(
        self,
        handler: RegistryHandler,
    ) -> r[m.HandlerRegistrationDetails]:
        """Register an already-constructed handler instance.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            r[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        # Propagate context for distributed tracing
        handler_name = handler.__class__.__name__ if handler else "unknown"
        self._propagate_context(f"register_handler_{handler_name}")

        self.logger.debug(
            "Starting handler registration",
            operation="register_handler",
            handler_name=handler_name,
            handler_type=type(handler).__name__ if handler else "None",
        )

        key = FlextRegistry._resolve_handler_key(handler)
        if key in self._registered_keys:
            self.logger.debug(
                "Handler already registered, returning existing registration",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
            )
            # Return successful registration details for idempotent registration
            return r[m.HandlerRegistrationDetails].ok(
                m.HandlerRegistrationDetails(
                    registration_id=key,
                    handler_mode=c.Cqrs.HandlerType.COMMAND,
                    timestamp="",  # Will be set by model if needed
                    status=c.Cqrs.CommonStatus.RUNNING,
                ),
            )

        # Handler is already the correct type
        self.logger.debug(
            "Registering handler with dispatcher",
            operation="register_handler",
            handler_name=handler_name,
            handler_key=key,
        )
        # register_handler returns r[m.HandlerRegistrationResult]
        # register_handler accepts t.ConfigMapValue | BaseModel, but h works via runtime check
        # Type narrowing: handler is FlextHandlers which is compatible with t.ConfigMapValue
        registration_result = self._dispatcher.register_handler(
            handler,
        )
        if registration_result.is_success:
            # Convert model result to RegistrationDetails
            reg_result = registration_result.value
            if m.HandlerRegistrationResult not in type(reg_result).__mro__:
                return r[m.HandlerRegistrationDetails].fail(
                    "Dispatcher returned invalid registration payload",
                )
            reg_details = self._create_registration_details(reg_result, key)
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
            error=error_msg,
            consequence="Handler will not be available for dispatch",
        )
        # Use error property for type-safe str
        error_str = registration_result.error or "Unknown error"
        return r[m.HandlerRegistrationDetails].fail(error_str)

    def _add_successful_registration(
        self,
        key: str,
        registration: m.HandlerRegistrationDetails,
        summary: FlextRegistry.Summary,
    ) -> None:
        """Add successful registration to summary."""
        self._registered_keys.add(key)
        summary.registered.append(
            registration,
        )

    @staticmethod
    def _add_registration_error(
        key: str,
        error: str,
        summary: FlextRegistry.Summary,
    ) -> str:
        """Add registration error to summary.

        Returns:
            str: The error message that was added.

        """
        # When is_failure is True, error is never None (fail() converts None to "")
        # Use error or fallback message
        summary.errors.append(error or f"Failed to register handler '{key}'")
        return error

    def _finalize_summary(
        self,
        summary: FlextRegistry.Summary,
    ) -> r[FlextRegistry.Summary]:
        """Finalize summary based on error state.

        Returns:
            r[FlextRegistry.Summary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return r[FlextRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return r[FlextRegistry.Summary].ok(summary)

    def register_handlers(
        self,
        handlers: Sequence[RegistryHandler],
    ) -> r[FlextRegistry.Summary]:
        """Register multiple handlers in batch.

        Args:
            handlers: Sequence of handler instances to register

        Returns:
            r[FlextRegistry.Summary]: Batch registration summary

        """
        summary = FlextRegistry.Summary()
        for handler in handlers:
            # We must determine a key BEFORE registration to track it properly
            # The register_handler method recalculates this, but we need it here for consistent logging
            # However, simpler to let register_handler do the work and check result
            result = self.register_handler(handler)

            # Extract key/ID from handler for tracking
            key = FlextRegistry._resolve_handler_key(handler)

            if result.is_success:
                # Use value directly (it's HandlerRegistrationDetails)
                self._add_successful_registration(key, result.value, summary)
            else:
                self._add_registration_error(
                    key, result.error or "Unknown error", summary
                )

        return self._finalize_summary(summary)

    def register_bindings(
        self,
        bindings: Mapping[RegistryBindingKey, RegistryHandler],
    ) -> r[FlextRegistry.Summary]:
        """Register message-to-handler bindings.

        Args:
            bindings: Map of MessageType -> HandlerInstance

        Returns:
            r[FlextRegistry.Summary]: Batch registration summary

        """
        summary = FlextRegistry.Summary()

        # This implementation delegates to dispatcher's register_handler for each binding
        # But wait - register_handler takes (request, handler) or just (handler) if it's decorated
        # FlextRegistry.register_handler only takes (handler).
        # It seems FlextRegistry assumes handlers are self-describing (decorated) or we need to use dispatcher directly.

        # If we look at dispatcher.register_handler signature:
        # def register_handler(self, request: ..., handler: ... = None)

        for message_type, handler in bindings.items():
            message_type_name = (
                message_type if type(message_type) is str else message_type.__name__
            )
            key = f"binding::{message_type_name}::{handler.__class__.__name__}"

            # We use the dispatcher directly because registry's register_handler
            # currently only supports single-argument (self-describing) handlers
            # Use strict type checking for message_type (it's a type)
            try:
                # Dispatcher return type is r[m.HandlerRegistrationResult]
                # We need to adapt it to Registry logic
                reg_result = self._dispatcher.register_handler(
                    message_type,
                    handler,
                )

                if reg_result.is_success:
                    # Convert dispatcher result to Registry details
                    val = reg_result.value
                    if m.HandlerRegistrationResult not in type(val).__mro__:
                        self._add_registration_error(
                            key,
                            "Dispatcher returned invalid registration payload",
                            summary,
                        )
                        continue
                    details = self._create_registration_details(val, key)
                    self._add_successful_registration(key, details, summary)
                else:
                    self._add_registration_error(
                        key, reg_result.error or "Unknown error", summary
                    )
            except Exception as e:
                self._add_registration_error(key, str(e), summary)

        return self._finalize_summary(summary)

    # ------------------------------------------------------------------
    # Generic Plugin Registry API
    # ------------------------------------------------------------------
    def register_plugin(
        self,
        category: str,
        name: str,
        plugin: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue] | BaseModel,
        *,
        validate: Callable[[t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue] | BaseModel], r[bool]] | None = None,
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
        if key in self._registered_keys:
            self.logger.debug(
                "Plugin already registered (idempotent)",
                category=category,
                name=name,
            )
            return r[bool].ok(value=True)

        # Store plugin in container for retrieval
        # plugin is from method signature
        self.container.register(key, plugin)
        self._registered_keys.add(key)
        self.logger.info("Registered %s: %s", category, name)
        return r[bool].ok(value=True)

    def get_plugin(self, category: str, name: str) -> r[t.RegisterableService]:
        """Get a registered plugin by category and name.

        Returns:
            Success with plugin (RegisterableService) or failure if not found.
        """
        key = f"{category}::{name}"
        if key not in self._registered_keys:
            available = [
                k.split("::")[1]
                for k in self._registered_keys
                if k.startswith(f"{category}::")
            ]
            return r[t.RegisterableService].fail(
                f"{category} '{name}' not found. Available: {available}",
            )

        raw_result = self.container.get(key)
        if raw_result.is_failure:
            return r[t.RegisterableService].fail(
                f"Failed to retrieve {category} '{name}': {raw_result.error}",
            )
        plugin_value = raw_result.value
        _mro = getattr(type(plugin_value), "__mro__", ())
        if type(plugin_value) in (str, int, float, bool, type(None)) or BaseModel in _mro or type(plugin_value) is Path:
            return r[t.RegisterableService].ok(plugin_value)
        if type(plugin_value) is dict:
            return r[t.RegisterableService].ok(
                FlextRuntime.normalize_to_general_value(plugin_value),
            )
        _items = getattr(plugin_value, "items", None)
        if callable(_items):
            view = _items()
            built: dict[str, t.ConfigMapValue] = {str(k): v for k, v in view}
            return r[t.RegisterableService].ok(
                FlextRuntime.normalize_to_general_value(built),
            )
        if (type(plugin_value) in (list, tuple) or (hasattr(plugin_value, "__getitem__") and hasattr(plugin_value, "__len__"))) and type(plugin_value) not in (str, bytes, bytearray):
            return r[t.RegisterableService].ok(
                FlextRuntime.normalize_to_general_value(list(plugin_value)),
            )
        return r[t.RegisterableService].fail(
            f"{category} '{name}' is not a registerable value",
        )

    def list_plugins(self, category: str) -> r[list[str]]:
        """List all plugins in a category.

        Args:
            category: Plugin category to list

        Returns:
            r[list[str]]: Success with list of plugin names.

        """
        plugins = [
            k.split("::")[1]
            for k in self._registered_keys
            if k.startswith(f"{category}::")
        ]
        return r[list[str]].ok(plugins)

    def unregister_plugin(self, category: str, name: str) -> r[bool]:
        """Unregister a plugin.

        Args:
            category: Plugin category
            name: Plugin name

        Returns:
            r[bool]: Success if unregistered, failure if not found.

        """
        key = f"{category}::{name}"
        if key not in self._registered_keys:
            return r[bool].fail(f"{category} '{name}' not registered")

        self._registered_keys.discard(key)
        self.logger.info("Unregistered %s: %s", category, name)
        return r[bool].ok(value=True)

    # ------------------------------------------------------------------
    # Class-Level Plugin Registry API (for auto-discovery patterns)
    # ------------------------------------------------------------------
    def register_class_plugin(
        self,
        category: str,
        name: str,
        plugin: t.RegistrablePlugin,
    ) -> r[bool]:
        """Register plugin to class-level storage (shared across all instances).

        Use this for auto-discovery patterns where plugins discovered once
        should be visible to all instances of this registry class.

        Args:
            category: Plugin category (e.g., "ldif_servers")
            name: Plugin name within the category
            plugin: Plugin instance to register

        Returns:
            r[bool]: Success if registered (idempotent - re-registration is OK).

        """
        if not name:
            return r[bool].fail(f"{category} name cannot be empty")

        key = f"{category}::{name}"
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

    def get_class_plugin(self, category: str, name: str) -> r[t.RegistrablePlugin]:
        """Get plugin from class-level storage.

        Args:
            category: Plugin category
            name: Plugin name

        Returns:
            r[t.RegistrablePlugin]: Success with plugin or failure if not found.

        """
        key = f"{category}::{name}"
        cls = type(self)
        if key not in cls._class_registered_keys:
            available = [
                k.split("::")[1]
                for k in cls._class_registered_keys
                if k.startswith(f"{category}::")
            ]
            return r[t.RegistrablePlugin].fail(
                f"{category} '{name}' not found. Available: {available}",
            )
        return r[t.RegistrablePlugin].ok(cls._class_plugin_storage[key])

    def list_class_plugins(self, category: str) -> r[list[str]]:
        """List class-level plugins in category.

        Args:
            category: Plugin category to list

        Returns:
            r[list[str]]: Success with list of plugin names.

        """
        cls = type(self)
        return r[list[str]].ok([
            k.split("::")[1]
            for k in cls._class_registered_keys
            if k.startswith(f"{category}::")
        ])

    def unregister_class_plugin(self, category: str, name: str) -> r[bool]:
        """Unregister plugin from class-level storage.

        Args:
            category: Plugin category
            name: Plugin name

        Returns:
            r[bool]: Success if unregistered, failure if not found.

        """
        key = f"{category}::{name}"
        cls = type(self)
        if key not in cls._class_registered_keys:
            return r[bool].fail(f"{category} '{name}' not registered")

        del cls._class_plugin_storage[key]
        cls._class_registered_keys.discard(key)
        self.logger.info("Unregistered class plugin %s: %s", category, name)
        return r[bool].ok(value=True)

    @classmethod
    def reset_class_storage(cls) -> None:
        """Reset class-level storage (for testing only).

        This method clears all class-level plugins for this specific class.
        Use in test fixtures to ensure test isolation.
        """
        cls._class_plugin_storage = {}
        cls._class_registered_keys = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_handler_key(
        handler: RegistryHandler,
    ) -> str:
        """Resolve registration key from handler."""
        handler_id = getattr(handler, "handler_id", None)
        return (
            handler_id
            if type(handler_id) is str and handler_id
            else handler.__class__.__name__
        )

    def register(
        self,
        name: str,
        service: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue] | BaseModel,
        metadata: Mapping[str, t.ConfigMapValue] | m.Metadata | None = None,
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
        # Normalize metadata to dict for internal use
        validated_metadata: Mapping[str, t.ConfigMapValue] | dict[str, t.ConfigMapValue] | None = None
        if metadata is not None:
            meta = metadata
            if type(meta) is m.Metadata:
                attrs = meta.attributes
                validated_metadata = {str(k): v for k, v in attrs.items()} if attrs else None
            elif type(meta) is dict:
                validated_metadata = meta
            else:
                _items = getattr(meta, "items", None)
                if callable(_items):
                    validated_metadata = dict(_items())
                else:
                    return r[bool].fail(
                        f"metadata must be dict or m.Metadata, got {type(metadata).__name__}",
                    )

        # Store metadata if provided (for future use)
        if validated_metadata is not None and (
            type(validated_metadata) is dict
            or (hasattr(validated_metadata, "keys") and hasattr(validated_metadata, "__getitem__"))
        ):
            metadata_dict: dict[str, t.ConfigMapValue] = dict(validated_metadata.items())
            metadata_keys_str: str = ",".join(metadata_dict.keys())
            self.logger.debug(
                "Registering service with metadata",
                operation="with_service",
                service_name=name,
                has_metadata=True,
                metadata_keys=metadata_keys_str,
            )

        # Delegate to container (x.container returns FlextContainer)
        # Use with_service for fluent API compatibility (returns Self)
        try:
            # service is already valid registerable type (from method signature)
            # with_service returns Self for fluent chaining, but we don't need the return value
            _ = self.container.with_service(name, service)
            return r[bool].ok(value=True)
        except ValueError as e:
            error_str = str(e)
            return r[bool].fail(error_str)

    # =========================================================================
    # Protocol Implementations: RegistrationTracker, BatchProcessor
    # =========================================================================

    def register_item(
        self,
        name: str,
        item: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue] | BaseModel,
    ) -> r[bool]:
        """Register item (RegistrationTracker protocol)."""
        # Direct implementation without try/except - use FlextResult for error handling
        return self.register(name, item)

    def get_item(self, name: str) -> r[t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue] | BaseModel]:
        """Get registered item (RegistrationTracker protocol)."""
        try:
            return r[t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue] | BaseModel].ok(
                getattr(self, name),
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue] | BaseModel].fail(
                str(e),
            )

    def list_items(self) -> r[list[str]]:
        """List registered items (RegistrationTracker protocol)."""
        try:
            keys = list(getattr(self, "_registered_keys", []))
            return r[list[str]].ok(keys)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[list[str]].fail(str(e))

    @staticmethod
    def get_batch_size() -> int:
        """Get batch size (BatchProcessor protocol)."""
        return 100


__all__ = ["FlextRegistry"]
