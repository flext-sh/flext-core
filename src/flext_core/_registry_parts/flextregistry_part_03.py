"""Handler registration and discovery utilities.

Provides handler registration with binding, tracking, and batch operations
for the FLEXT dispatcher system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, e, m, p, r, t

from .flextregistry_part_02 import (
    FlextRegistry as FlextRegistryPart02,
)


class FlextRegistry(FlextRegistryPart02):
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
            handler_name = getattr(handler, "__name__", type(handler).__name__)
            if not isinstance(handler_name, str):
                handler_name = type(handler).__name__
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
            handler_name = getattr(handler, "__name__", None)
            key = (
                handler_name
                if isinstance(handler_name, str)
                else type(handler).__name__
            )
            if result.success:
                self._add_successful_registration(key, result.value, summary)
            else:
                summary.errors.append(
                    result.error or f"Failed to register handler '{key}'",
                )
        return self._finalize_summary(summary)


__all__: list[str] = ["FlextRegistry"]
