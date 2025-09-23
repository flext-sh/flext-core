"""Dispatcher registration helpers aligned with the 1.0.0 modernization plan.

The registry implements the handler bootstrapping workflow called out in
``README.md`` and ``docs/architecture.md`` so CLI and connector packages can
standardise idempotent registrations during the unified dispatcher rollout.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from flext_core.constants import FlextConstants
from flext_core.dispatcher import FlextDispatcher
from flext_core.handlers import FlextHandlers
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

HandlerModeLiteral = Literal["command", "query"]
HandlerTypeLiteral = Literal["command", "query"]
RegistrationStatusLiteral = Literal["active", "inactive"]


class FlextRegistry:
    """Stateful helper that reports on handler adoption across packages.

    By wrapping ``FlextDispatcher`` it provides registration summaries and
    idempotency guarantees that feed ecosystem migration dashboards during the
    1.0.0 rollout.
    """

    @dataclass(slots=True)
    class Summary:
        """Aggregated outcome used for 1.0.0 handler adoption tracking."""

        registered: list[FlextModels.RegistrationDetails] = field(default_factory=list)
        skipped: list[str] = field(default_factory=list)
        errors: list[str] = field(default_factory=list)

        @property
        def is_success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            return not self.errors

        @property
        def successful_registrations(self) -> int:
            """Number of successful registrations."""
            return len(self.registered)

        @property
        def failed_registrations(self) -> int:
            """Number of failed registrations."""
            return len(self.errors)

    def __init__(self, dispatcher: FlextDispatcher) -> None:
        """Initialize the registry with a FlextDispatcher instance."""
        # No super() call needed as this class doesn't inherit from anything
        self._dispatcher = dispatcher
        self._registered_keys: set[str] = set()

    def _safe_get_handler_mode(self, value: object) -> HandlerModeLiteral:
        """Safely extract and validate handler mode from dict value."""
        if value == "query":
            return "query"
        if value == "command":
            return "command"
        # Default to command for invalid values
        return "command"

    def _safe_get_status(self, value: object) -> Literal["active", "inactive"]:
        """Safely extract and validate status from dict value."""
        if value == "active":
            return "active"
        if value == "inactive":
            return "inactive"
        # Default to active for invalid values
        return "active"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_handler(
        self,
        handler: FlextHandlers[object, object],
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register an already-constructed handler instance.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        key = self._resolve_handler_key(handler)
        if key in self._registered_keys:
            # Return successful registration details for idempotent registration
            return FlextResult[FlextModels.RegistrationDetails].ok(
                FlextModels.RegistrationDetails(
                    registration_id=key,
                    handler_mode="command",
                    timestamp="",  # Will be set by model if needed
                    status="active",
                )
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
                    )
                ),
                timestamp=str(reg_data.get("timestamp", "")),
                status=self._safe_get_status(
                    reg_data.get(
                        "status",
                        FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                    )
                ),
            )
            return FlextResult[FlextModels.RegistrationDetails].ok(reg_details)
        return FlextResult[FlextModels.RegistrationDetails].fail(
            registration.error or "Unknown error"
        )

    def register_handlers(
        self,
        handlers: Iterable[FlextHandlers[object, object]],
    ) -> FlextResult[FlextRegistry.Summary]:
        """Register multiple handlers in one shot using railway pattern.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextRegistry.Summary()
        for handler in handlers:
            result = self._process_single_handler(handler, summary)
            if result.is_failure:
                return FlextResult[FlextRegistry.Summary].fail(
                    result.error or "Handler processing failed"
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
        registration_result = self._dispatcher.register_handler(handler)
        if registration_result.is_success:
            # Convert dict to RegistrationDetails
            reg_data = registration_result.value
            reg_details = FlextModels.RegistrationDetails(
                registration_id=str(reg_data.get("registration_id", key)),
                handler_mode=self._safe_get_handler_mode(
                    reg_data.get(
                        "handler_mode",
                        FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                    )
                ),
                timestamp=str(reg_data.get("timestamp", "")),
                status=self._safe_get_status(
                    reg_data.get(
                        "status",
                        FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                    )
                ),
            )
            self._add_successful_registration(key, reg_details, summary)
            return FlextResult[None].ok(None)
        self._add_registration_error(key, registration_result.error or "", summary)
        return FlextResult[None].fail(
            registration_result.error or "Registration failed"
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
                    )
                ),
                timestamp=str(reg_data.get("timestamp", "")),
                status=self._safe_get_status(
                    reg_data.get(
                        "status",
                        FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                    )
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
                dict[str, object] | None,
            ],
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

                if isinstance(entry, tuple):
                    tuple_handler_func: Callable[[object], object] = entry[0]
                    tuple_handler_config: FlextTypes.Core.Dict | None = entry[1]
                    # Convert function to handler object for registration

                    # Convert dict config to proper Handler config or use None
                    config_obj = None
                    handler_mode: HandlerModeLiteral = "command"

                    if tuple_handler_config:
                        try:
                            # Ensure proper types for literal fields
                            handler_type_val: HandlerTypeLiteral = "command"
                            handler_mode_val: HandlerModeLiteral = "command"

                            handler_type_raw = (
                                tuple_handler_config.get("handler_type")
                                if tuple_handler_config
                                else None
                            )
                            if isinstance(handler_type_raw, str):
                                raw_type: str = handler_type_raw
                                if raw_type in {
                                    FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
                                    FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
                                }:
                                    handler_type_val = cast(
                                        "HandlerTypeLiteral", raw_type
                                    )

                            handler_mode_raw = (
                                tuple_handler_config.get("handler_mode")
                                if tuple_handler_config
                                else None
                            )
                            if isinstance(handler_mode_raw, str):
                                raw_mode = handler_mode_raw
                                if raw_mode in {
                                    FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                                    FlextConstants.Dispatcher.HANDLER_MODE_QUERY,
                                }:
                                    handler_mode_val = cast(
                                        "HandlerModeLiteral", raw_mode
                                    )

                            metadata_raw: dict[str, object] = cast(
                                "dict[str, object]",
                                tuple_handler_config.get("metadata", {})
                                if tuple_handler_config
                                else {},
                            )
                            metadata_val = metadata_raw

                            config_obj = FlextModels.CqrsConfig.Handler(
                                handler_id=str(
                                    tuple_handler_config.get("handler_id", "default")
                                    if tuple_handler_config
                                    else "default"
                                ),
                                handler_name=str(
                                    tuple_handler_config.get("handler_name", "unnamed")
                                    if tuple_handler_config
                                    else "unnamed"
                                ),
                                handler_type=handler_type_val,
                                handler_mode=handler_mode_val,
                                metadata=metadata_val,
                            )
                            # Derive handler_mode from handler_type if not explicitly set
                            if (
                                tuple_handler_config
                                and "handler_mode" in tuple_handler_config
                                and isinstance(
                                    tuple_handler_config.get("handler_mode"), str
                                )
                                and tuple_handler_config.get("handler_mode")
                                in {
                                    FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                                    FlextConstants.Dispatcher.HANDLER_MODE_QUERY,
                                }
                            ):
                                handler_mode = handler_mode_val
                            elif handler_type_val == "command":
                                handler_mode = "command"
                            elif handler_type_val == "query":
                                handler_mode = "query"
                            # If neither handler_mode nor handler_type_val are valid, use default
                        except Exception:
                            config_obj = None

                    # Convert config_obj to dict if it's a Handler object
                    config_dict = None
                    if config_obj is not None and hasattr(config_obj, "model_dump"):
                        config_dict = config_obj.model_dump()

                    # Create a handler wrapper for the function
                    handler_result = self._dispatcher.register_function(
                        message_type,
                        tuple_handler_func,
                        handler_config=config_dict,
                        mode=handler_mode,
                    )
                elif isinstance(entry, dict) and "handler" in entry:
                    # Handle the test case format: {"handler": ..., "metadata": ...}
                    dict_handler_func: Callable[[object], object] = cast(
                        "Callable[[object], object]", entry["handler"]
                    )

                    def _get_default(_k: str, d: object) -> object:
                        return d

                    dict_handler_config: FlextTypes.Core.Dict = cast(
                        "FlextTypes.Core.Dict",
                        getattr(entry, "get", _get_default)("metadata", {}),
                    )

                    # Create a handler wrapper for the function
                    handler_result = self._dispatcher.register_function(
                        message_type,
                        dict_handler_func,
                        handler_config=dict_handler_config,
                        mode="command",
                    )
                else:
                    # Entry is already the correct type
                    handler_result = self._dispatcher.register_command(
                        message_type, entry, handler_config=None
                    )

                if handler_result.is_failure:
                    summary.errors.append(
                        str(handler_result.error)
                        or f"Failed to register function handler '{key}'",
                    )
                    continue

                self._registered_keys.add(key)
                # Convert dict to RegistrationDetails
                reg_data = handler_result.value
                reg_details = FlextModels.RegistrationDetails(
                    registration_id=str(reg_data.get("registration_id", key)),
                    handler_mode=self._safe_get_handler_mode(
                        reg_data.get(
                            "handler_mode",
                            FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                        )
                    ),
                    timestamp=str(reg_data.get("timestamp", "")),
                    status=self._safe_get_status(
                        reg_data.get(
                            "status",
                            FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                        )
                    ),
                )
                summary.registered.append(reg_details)
            except Exception as e:
                # Handle any exceptions gracefully by adding to errors
                error_msg = f"Failed to process entry for '{message_type}': {e!s}"
                summary.errors.append(error_msg)
                continue

        # Always return success with summary containing both successes and errors
        # This allows callers to inspect what succeeded and what failed
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
            dict[str, object] | None,
        ],
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
        return self._resolve_binding_key(entry, message_type)


__all__ = ["FlextRegistry"]
