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
from typing import cast

from flext_core.dispatcher import FlextDispatcher
from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult
from flext_core.typings import MessageT, ResultT


class FlextDispatcherRegistry:
    """Stateful helper that reports on handler adoption across packages.

    By wrapping ``FlextDispatcher`` it provides registration summaries and
    idempotency guarantees that feed ecosystem migration dashboards during the
    1.0.0 rollout.
    """

    @dataclass(slots=True)
    class Summary:
        """Aggregated outcome used for 1.0.0 handler adoption tracking."""

        registered: list[FlextDispatcher.Registration[object, object]] = field(
            default_factory=list,
        )
        skipped: list[str] = field(default_factory=list)
        errors: list[str] = field(default_factory=list)

        @property
        def is_success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            return not self.errors

    def __init__(self, dispatcher: FlextDispatcher) -> None:
        """Initialize the registry with a FlextDispatcher instance."""
        self._dispatcher = dispatcher
        self._registered_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_handler(
        self,
        handler: FlextHandlers[MessageT, ResultT],
    ) -> FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]:
        """Register an already-constructed handler instance.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.
        
        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        key = self._resolve_handler_key(handler)
        if key in self._registered_keys:
            return FlextResult[FlextDispatcher.Registration[MessageT, ResultT]].ok(
                FlextDispatcher.Registration(None, handler),
            )

        registration = self._dispatcher.register_handler(handler)
        if registration.is_success:
            self._registered_keys.add(key)
        return registration

    def register_handlers(
        self,
        handlers: Iterable[FlextHandlers[MessageT, ResultT]],
    ) -> FlextResult[FlextDispatcherRegistry.Summary]:
        """Register multiple handlers in one shot using railway pattern.
        
        Returns:
            FlextResult[FlextDispatcherRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextDispatcherRegistry.Summary()
        for handler in handlers:
            result = self._process_single_handler(handler, summary)
            if result.is_failure:
                return FlextResult[FlextDispatcherRegistry.Summary].fail(
                    result.error or "Handler processing failed"
                )
        return self._finalize_summary(summary)

    def _process_single_handler(
        self,
        handler: FlextHandlers[MessageT, ResultT],
        summary: FlextDispatcherRegistry.Summary,
    ) -> FlextResult[None]:
        """Process a single handler registration.
        
        Returns:
            FlextResult[None]: Success result if registration succeeds.

        """
        key = self._resolve_handler_key(handler)
        if key in self._registered_keys:
            summary.skipped.append(key)
            return FlextResult[None].ok(None)

        registration_result = self._dispatcher.register_handler(handler)
        if registration_result.is_success:
            self._add_successful_registration(key, registration_result.value, summary)
            return FlextResult[None].ok(None)
        self._add_registration_error(key, registration_result.error or "", summary)
        return FlextResult[None].fail(
            registration_result.error or "Registration failed"
        )

    def _add_successful_registration(
        self,
        key: str,
        registration: FlextDispatcher.Registration[MessageT, ResultT],
        summary: FlextDispatcherRegistry.Summary,
    ) -> None:
        """Add successful registration to summary."""
        self._registered_keys.add(key)
        summary.registered.append(
            cast("FlextDispatcher.Registration[object, object]", registration),
        )

    def _add_registration_error(
        self,
        key: str,
        error: str,
        summary: FlextDispatcherRegistry.Summary,
    ) -> str:
        """Add registration error to summary.
        
        Returns:
            str: The error message that was added.

        """
        summary.errors.append(error or f"Failed to register handler '{key}'")
        return error

    def _finalize_summary(
        self,
        summary: FlextDispatcherRegistry.Summary,
    ) -> FlextResult[FlextDispatcherRegistry.Summary]:
        """Finalize summary based on error state.
        
        Returns:
            FlextResult[FlextDispatcherRegistry.Summary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return FlextResult[FlextDispatcherRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return FlextResult[FlextDispatcherRegistry.Summary].ok(summary)

    def register_bindings(
        self,
        bindings: Sequence[tuple[type[MessageT], FlextHandlers[MessageT, ResultT]]],
    ) -> FlextResult[FlextDispatcherRegistry.Summary]:
        """Register handlers bound to explicit message types.
        
        Returns:
            FlextResult[FlextDispatcherRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextDispatcherRegistry.Summary()
        for message_type, handler in bindings:
            key = self._resolve_binding_key(handler, message_type)
            if key in self._registered_keys:
                summary.skipped.append(key)
                continue

            registration = self._dispatcher.register_command(message_type, handler)
            if registration.is_failure:
                summary.errors.append(
                    registration.error
                    or f"Failed to register handler '{key}' for '{message_type.__name__}'",
                )
                continue

            self._registered_keys.add(key)
            summary.registered.append(
                cast(
                    "FlextDispatcher.Registration[object, object]",
                    registration.value,
                ),
            )

        if summary.errors:
            return FlextResult[FlextDispatcherRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return FlextResult[FlextDispatcherRegistry.Summary].ok(summary)

    def register_function_map(
        self,
        mapping: Mapping[
            type[MessageT],
            FlextHandlers[MessageT, ResultT]
            | tuple[
                Callable[[MessageT], ResultT | FlextResult[ResultT]],
                dict[str, object] | None,
            ],
        ],
    ) -> FlextResult[FlextDispatcherRegistry.Summary]:
        """Register plain callables or pre-built handlers for message types.
        
        Returns:
            FlextResult[FlextDispatcherRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextDispatcherRegistry.Summary()
        for message_type, entry in mapping.items():
            key = self._resolve_binding_key_from_entry(entry, message_type)
            if key in self._registered_keys:
                summary.skipped.append(key)
                continue

            if isinstance(entry, tuple):
                handler_func, handler_config = entry
                handler_result = self._dispatcher.register_function(
                    message_type,
                    handler_func,
                    handler_config=handler_config,
                )
            else:
                handler_result = self._dispatcher.register_command(message_type, entry)

            if handler_result.is_failure:
                summary.errors.append(
                    handler_result.error
                    or f"Failed to register function handler '{key}'",
                )
                continue

            self._registered_keys.add(key)
            summary.registered.append(
                cast(
                    "FlextDispatcher.Registration[object, object]",
                    handler_result.value,
                ),
            )

        if summary.errors:
            return FlextResult[FlextDispatcherRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return FlextResult[FlextDispatcherRegistry.Summary].ok(summary)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_handler_key(self, handler: FlextHandlers[MessageT, ResultT]) -> str:
        handler_id = getattr(handler, "handler_id", None)
        if isinstance(handler_id, str) and handler_id:
            return handler_id
        return handler.__class__.__name__

    def _resolve_binding_key(
        self,
        handler: FlextHandlers[MessageT, ResultT],
        message_type: type[MessageT],
    ) -> str:
        base_key = self._resolve_handler_key(handler)
        return f"{base_key}::{message_type.__name__}"

    def _resolve_binding_key_from_entry(
        self,
        entry: FlextHandlers[MessageT, ResultT]
        | tuple[
            Callable[[MessageT], ResultT | FlextResult[ResultT]],
            dict[str, object] | None,
        ],
        message_type: type[MessageT],
    ) -> str:
        if isinstance(entry, tuple):
            handler_func = entry[0]
            handler_name = getattr(handler_func, "__name__", str(handler_func))
            return f"{handler_name}::{message_type.__name__}"
        return self._resolve_binding_key(entry, message_type)


__all__ = ["FlextDispatcherRegistry"]
