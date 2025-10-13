"""Handler registration and discovery utilities.

This module provides FlextRegistry, utilities for registering and managing
command/query handlers with FlextDispatcher, including batch registration,
idempotency guarantees, and registration tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""
# pyright: basic

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    Annotated,
    cast,
)

from pydantic import Field, computed_field

from flext_core.constants import FlextConstants
from flext_core.dispatcher import FlextDispatcher
from flext_core.handlers import FlextHandlers
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextRegistry(FlextMixins):
    """Handler registration and discovery utilities.

    Provides utilities for registering and managing command/query handlers
    with FlextDispatcher, including batch registration, idempotency guarantees,
    and registration tracking.

    Features:
    - Handler registration with dispatcher integration
    - Batch registration for multiple handlers
    - Registration deduplication and idempotency
    - Registration summary and statistics tracking
    - Handler discovery and enumeration
    - Registration status validation
    - Handler unregistration and cleanup
    - Registration error tracking and reporting

    Usage:
        >>> from flext_core import FlextRegistry, FlextDispatcher
        >>>
        >>> dispatcher = FlextDispatcher()
        >>> registry = FlextRegistry(dispatcher)
        >>>
        >>> # Register single handler
        >>> result = registry.register_handler(CreateUserHandler())
        >>>
        >>> # Batch register multiple handlers
        >>> handlers = [CreateUserHandler(), UpdateUserHandler()]
        >>> summary = registry.register_handlers(handlers)
    """

    class Summary(FlextModels.Value):
        """Aggregated outcome for batch handler registration tracking.

        Provides comprehensive summary of batch handler registration operations,
        tracking successful, skipped, and failed registrations with detailed
        metadata and computed success indicators.

        This immutable value object is used by FlextRegistry to report the
        results of batch handler registration operations, enabling idempotent
        registration with full auditability.

        Attributes:
            registered: List of successfully registered handlers with details
            skipped: List of handler identifiers that were skipped (already registered)
            errors: List of error messages for failed registrations

        Computed Properties:
            is_success: True if no errors occurred during registration
            successful_registrations: Count of successfully registered handlers
            failed_registrations: Count of failed registration attempts

        Examples:
            >>> from pydantic import Field
            >>> summary = FlextRegistry.Summary(
            ...     registered=[
            ...         FlextModels.RegistrationDetails(
            ...             registration_id="reg-001",
            ...             handler_mode="command",
            ...             timestamp="2025-01-01T00:00:00Z",
            ...             status="running",
            ...         )
            ...     ],
            ...     skipped=["CreateUserCommand"],
            ...     errors=[],
            ... )
            >>> summary.is_success
            True
            >>> summary.successful_registrations
            1

        """

        registered: Annotated[
            list[FlextModels.RegistrationDetails],
            Field(
                default_factory=list,
                description="Successfully registered handlers with registration details",
            ),
        ] = Field(default_factory=list)
        skipped: Annotated[
            FlextTypes.StringList,
            Field(
                default_factory=list,
                description="Handler identifiers that were skipped (already registered)",
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = Field(default_factory=list)
        errors: Annotated[
            FlextTypes.StringList,
            Field(
                default_factory=list,
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = Field(default_factory=list)

        @computed_field
        @property
        def is_success(self) -> bool:
            """Indicate whether the batch registration fully succeeded.

            Returns:
                True if no errors occurred, False otherwise

            Examples:
                >>> summary = FlextRegistry.Summary(registered=[...], errors=[])
                >>> summary.is_success
                True

            """
            return not self.errors

        @computed_field
        @property
        def successful_registrations(self) -> int:
            """Number of successful registrations.

            Returns:
                Count of successfully registered handlers

            Examples:
                >>> summary = FlextRegistry.Summary(registered=[detail1, detail2])
                >>> summary.successful_registrations
                2

            """
            return len(self.registered)

        @computed_field
        @property
        def failed_registrations(self) -> int:
            """Number of failed registrations.

            Returns:
                Count of failed registration attempts

            Examples:
                >>> summary = FlextRegistry.Summary(errors=["error1", "error2"])
                >>> summary.failed_registrations
                2

            """
            return len(self.errors)

    def __init__(self, dispatcher: FlextDispatcher) -> None:
        """Initialize the registry with a FlextDispatcher instance."""
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_registry")

        # Enrich context with registry metadata for observability
        self._enrich_context(
            service_type="registry",
            dispatcher_type=type(dispatcher).__name__,
            supports_batch_registration=True,
            idempotent_registration=True,
        )

        self._dispatcher = dispatcher
        self._registered_keys: set[str] = set()

    def _safe_get_handler_mode(self, value: object) -> FlextConstants.HandlerModeSimple:
        """Safely extract and validate handler mode from dict value."""
        if value == "query":
            return "query"
        if value == "command":
            return "command"
        # Default to command for invalid values
        return "command"

    def _safe_get_status(self, value: object) -> FlextConstants.Status:
        """Safely extract and validate status from dict value."""
        if value == "active":
            return "running"
        if value == "inactive":
            return "completed"
        # Default to running for invalid values
        return "running"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_handler(
        self,
        handler: FlextHandlers[object, object] | None,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register an already-constructed handler instance.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        # Propagate context for distributed tracing
        handler_name = handler.__class__.__name__ if handler else "unknown"
        self._propagate_context(f"register_handler_{handler_name}")

        # Validate handler is not None
        if handler is None:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                "Handler cannot be None",
            )

        key = self._resolve_handler_key(handler)
        if key in self._registered_keys:
            # Return successful registration details for idempotent registration
            return FlextResult[FlextModels.RegistrationDetails].ok(
                FlextModels.RegistrationDetails(
                    registration_id=key,
                    handler_mode="command",
                    timestamp="",  # Will be set by model if needed
                    status="running",
                ),
            )

        # Handler is already the correct type
        registration = self._dispatcher.register_handler(handler)
        if registration.is_success:
            self._registered_keys.add(key)
            # Convert dict to RegistrationDetails
            reg_data = registration.value
            reg_details = FlextModels.RegistrationDetails(
                registration_id=str(reg_data.get("registration_id", key)),
                handler_mode=self._safe_get_handler_mode(
                    reg_data.get(
                        "handler_mode",
                        FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                    ),
                ),
                timestamp=str(reg_data.get("timestamp", "")),
                status=self._safe_get_status(
                    reg_data.get(
                        "status",
                        FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                    ),
                ),
            )
            return FlextResult[FlextModels.RegistrationDetails].ok(reg_details)
        return FlextResult[FlextModels.RegistrationDetails].fail(
            registration.error or "Unknown error",
        )

    def register_handlers(
        self,
        handlers: Iterable[FlextHandlers[object, object]],
    ) -> FlextResult[FlextRegistry.Summary]:
        """Register multiple handlers in one shot using railway pattern.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with registration summary.

        """
        # Propagate context for distributed tracing
        self._propagate_context("register_handlers_batch")

        summary = FlextRegistry.Summary()
        for handler in handlers:
            result: FlextResult[None] = self._process_single_handler(handler, summary)
            if result.is_failure:
                return FlextResult[FlextRegistry.Summary].fail(
                    result.error or "Handler processing failed",
                )
        return self._finalize_summary(summary)

    def _process_single_handler(
        self,
        handler: FlextHandlers[object, object],
        summary: FlextRegistry.Summary,
    ) -> FlextResult[None]:
        """Process a single handler registration.

        Returns:
            FlextResult[None]: Success result if registration succeeds.

        """
        key = self._resolve_handler_key(handler)
        if key in self._registered_keys:
            summary.skipped.append(key)
            return FlextResult[None].ok(None)

        # Handler is already the correct type
        registration_result: FlextResult[FlextTypes.Dict] = (
            self._dispatcher.register_handler(handler)
        )
        if registration_result.is_success:
            # Convert dict to RegistrationDetails
            reg_data = registration_result.value
            reg_details = FlextModels.RegistrationDetails(
                registration_id=str(reg_data.get("registration_id", key)),
                handler_mode=self._safe_get_handler_mode(
                    reg_data.get(
                        "handler_mode",
                        FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                    ),
                ),
                timestamp=str(reg_data.get("timestamp", "")),
                status=self._safe_get_status(
                    reg_data.get(
                        "status",
                        FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                    ),
                ),
            )
            self._add_successful_registration(key, reg_details, summary)
            return FlextResult[None].ok(None)
        self._add_registration_error(key, registration_result.error or "", summary)
        return FlextResult[None].fail(
            registration_result.error or "Registration failed",
        )

    def _add_successful_registration(
        self,
        key: str,
        registration: FlextModels.RegistrationDetails,
        summary: FlextRegistry.Summary,
    ) -> None:
        """Add successful registration to summary."""
        self._registered_keys.add(key)
        summary.registered.append(
            registration,
        )

    def _add_registration_error(
        self,
        key: str,
        error: str,
        summary: FlextRegistry.Summary,
    ) -> str:
        """Add registration error to summary.

        Returns:
            str: The error message that was added.

        """
        summary.errors.append(str(error) or f"Failed to register handler '{key}'")
        return error

    def _finalize_summary(
        self,
        summary: FlextRegistry.Summary,
    ) -> FlextResult[FlextRegistry.Summary]:
        """Finalize summary based on error state.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return FlextResult[FlextRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return FlextResult[FlextRegistry.Summary].ok(summary)

    def register_bindings(
        self,
        bindings: Sequence[tuple[type[object], FlextHandlers[object, object]]],
    ) -> FlextResult[FlextRegistry.Summary]:
        """Register handlers bound to explicit message types.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextRegistry.Summary()
        for message_type, handler in bindings:
            key = self._resolve_binding_key(handler, message_type)
            if key in self._registered_keys:
                summary.skipped.append(key)
                continue

            # Handler is already the correct type
            registration = self._dispatcher.register_command(message_type, handler)
            if registration.is_failure:
                summary.errors.append(
                    str(registration.error)
                    or f"Failed to register handler '{key}' for '{message_type.__name__}'",
                )
                continue

            self._registered_keys.add(key)
            # Convert dict to RegistrationDetails
            reg_data = registration.value
            reg_details = FlextModels.RegistrationDetails(
                registration_id=str(reg_data.get("registration_id", key)),
                handler_mode=self._safe_get_handler_mode(
                    reg_data.get(
                        "handler_mode",
                        FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                    ),
                ),
                timestamp=str(reg_data.get("timestamp", "")),
                status=self._safe_get_status(
                    reg_data.get(
                        "status",
                        FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                    ),
                ),
            )
            summary.registered.append(reg_details)

        if summary.errors:
            return FlextResult[FlextRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return FlextResult[FlextRegistry.Summary].ok(summary)

    def register_function_map(
        self,
        mapping: Mapping[
            type[object],
            FlextHandlers[object, object]
            | tuple[
                Callable[[object], object | FlextResult[object]],
                object | FlextResult[object],
            ]
            | FlextTypes.Dict
            | tuple[object, ...]
            | None,
        ],
    ) -> FlextResult[FlextRegistry.Summary]:
        """Register plain callables or pre-built handlers for message types.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextRegistry.Summary()
        for message_type, entry in mapping.items():
            try:
                key = self._resolve_binding_key_from_entry(entry, message_type)
                if key in self._registered_keys:
                    summary.skipped.append(key)
                    continue

                if isinstance(entry, (tuple, FlextHandlers)):
                    # Handle tuple (function, config) or FlextHandlers instance
                    if isinstance(entry, tuple):
                        handler_func, handler_config = entry
                        # Create handler from function
                        handler_result = self._dispatcher.create_handler_from_function(
                            cast(
                                "Callable[[object], object | FlextResult[object]]",
                                handler_func,
                            ),
                            cast("FlextTypes.Dict | None", handler_config),
                            "command",
                        )
                        if handler_result.is_success:
                            handler = handler_result.value
                            # Register with dispatcher
                            register_result = self._dispatcher.register_handler(handler)
                            if register_result.is_success:
                                reg_details = FlextModels.RegistrationDetails(
                                    registration_id=key,
                                    handler_mode="command",
                                    timestamp="",
                                    status="running",
                                )
                                summary.registered.append(reg_details)
                                self._registered_keys.add(key)
                            else:
                                summary.errors.append(
                                    f"Failed to register handler: {register_result.error}",
                                )
                        else:
                            summary.errors.append(
                                f"Failed to create handler: {handler_result.error}",
                            )
                    else:
                        # Handle FlextHandlers instance
                        register_result = self._dispatcher.register_handler(entry)
                        if register_result.is_success:
                            reg_details = FlextModels.RegistrationDetails(
                                registration_id=key,
                                handler_mode="command",
                                timestamp="",
                                status="running",
                            )
                            summary.registered.append(reg_details)
                            self._registered_keys.add(key)
                        else:
                            summary.errors.append(
                                f"Failed to register handler: {register_result.error}",
                            )
                else:
                    # Handle dict or other types
                    reg_details = FlextModels.RegistrationDetails(
                        registration_id=key,
                        handler_mode="command",
                        timestamp="",
                        status="running",
                    )
                    summary.registered.append(reg_details)
                    self._registered_keys.add(key)

            except Exception as e:
                error_msg = f"Failed to register handler for {message_type}: {e}"
                summary.errors.append(error_msg)
                continue

        return FlextResult[FlextRegistry.Summary].ok(summary)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_handler_key(self, handler: FlextHandlers[object, object]) -> str:
        handler_id = getattr(handler, "handler_id", None)
        if isinstance(handler_id, str) and handler_id:
            return handler_id
        return handler.__class__.__name__

    def _resolve_binding_key(
        self,
        handler: FlextHandlers[object, object],
        message_type: type[object],
    ) -> str:
        base_key = self._resolve_handler_key(handler)
        # Handle both type objects and string keys gracefully
        if hasattr(message_type, "__name__"):
            type_name = message_type.__name__
        else:
            # Fallback for string keys or other non-type objects
            type_name = str(message_type)
        return f"{base_key}::{type_name}"

    def _resolve_binding_key_from_entry(
        self,
        entry: FlextHandlers[object, object]
        | tuple[
            Callable[[object], object | FlextResult[object]],
            object | FlextResult[object],
        ]
        | FlextTypes.Dict
        | tuple[object, ...]
        | None,
        message_type: type[object],
    ) -> str:
        if isinstance(entry, tuple):
            handler_func = entry[0]
            handler_name = getattr(handler_func, "__name__", str(handler_func))
            # Handle both type objects and string keys gracefully
            if hasattr(message_type, "__name__"):
                type_name = message_type.__name__
            else:
                # Fallback for string keys or other non-type objects
                type_name = str(message_type)
            return f"{handler_name}::{type_name}"
        if isinstance(entry, FlextHandlers):
            return self._resolve_binding_key(entry, message_type)
        # Handle dict or other types
        return str(entry)

    def register(
        self,
        name: str,
        service: object,
        metadata: FlextTypes.Dict | None = None,
    ) -> FlextResult[None]:
        """Register a service component with optional metadata.

        Delegates to the container's register method for dependency injection.
        Metadata is currently stored for future use but not actively used.

        Args:
            name: Service name for later retrieval
            service: Service instance to register
            metadata: Optional metadata about the service

        Returns:
            FlextResult[None]: Success if registered or failure with error details.


        """
        # Store metadata if provided (for future use)
        if metadata:
            # Could store in a metadata registry, but for now just log
            self.logger.debug(
                f"Registering service '{name}' with metadata",
                extra={"service_name": name, "metadata": metadata},
            )

        # Delegate to container
        return self.container.register(name, service)


__all__ = ["FlextRegistry"]
