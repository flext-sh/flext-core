"""Extracted mixin for flext_tests."""

from __future__ import annotations

from flext_tests import c, m


class FlextTestsHandlerHelpersUtilitiesMixin:
    """Helpers for handler testing."""

    @staticmethod
    def create_handler_config(
        handler_id: str,
        handler_name: str,
        handler_type: c.HandlerType | None = None,
        handler_mode: c.HandlerType | None = None,
        command_timeout: int | None = None,
        max_command_retries: int | None = None,
        metadata: m.Metadata | None = None,
    ) -> m.Handler:
        """Create a handler configuration model using canonical model defaults."""
        resolved_handler_type = handler_type or c.HandlerType.COMMAND
        handler: m.Handler = m.Handler.model_validate({
            "handler_id": handler_id,
            "handler_name": handler_name,
            "handler_type": resolved_handler_type,
            "handler_mode": handler_mode or resolved_handler_type,
            **(
                {"command_timeout": command_timeout}
                if command_timeout is not None
                else {}
            ),
            **(
                {"max_command_retries": max_command_retries}
                if max_command_retries is not None
                else {}
            ),
            **({"metadata": metadata} if metadata is not None else {}),
        })
        return handler
