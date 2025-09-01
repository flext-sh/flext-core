"""Context facade (new) to operate with FlextContextConfig and headers.

Single-class module: defines `FlextContextCore` only.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from flext_core.context_new.config import FlextContextConfig
from flext_core.context_new.headers import FlextContextHeaders
from flext_core.result import FlextResult


class FlextContextCore:
    """Facade for typed context configuration and header mapping."""

    @staticmethod
    def default_config() -> FlextContextConfig:
        return FlextContextConfig()

    @staticmethod
    def validate_config(
        config: Mapping[str, object],
    ) -> FlextResult[FlextContextConfig]:
        try:
            model = FlextContextConfig.model_validate(dict(config))
            return FlextResult[FlextContextConfig].ok(model)
        except (ValidationError, ValueError, TypeError) as e:
            return FlextResult[FlextContextConfig].fail(f"Invalid context config: {e}")

    @staticmethod
    def to_header_context(context: Mapping[str, object]) -> dict[str, str]:
        headers: dict[str, str] = {}
        if (v := context.get("correlation_id")) and isinstance(v, str):
            headers[FlextContextHeaders.CORRELATION_ID] = v
        if (v := context.get("parent_correlation_id")) and isinstance(v, str):
            headers[FlextContextHeaders.PARENT_CORRELATION_ID] = v
        if (v := context.get("service_name")) and isinstance(v, str):
            headers[FlextContextHeaders.SERVICE_NAME] = v
        if (v := context.get("user_id")) and isinstance(v, str):
            headers[FlextContextHeaders.USER_ID] = v
        return headers

    @staticmethod
    def from_header_context(headers: Mapping[str, str]) -> dict[str, object]:
        """Normalize incoming header mapping to standard context keys."""
        ctx: dict[str, object] = {}
        if v := headers.get(FlextContextHeaders.CORRELATION_ID):
            ctx["correlation_id"] = v
        if v := headers.get(FlextContextHeaders.PARENT_CORRELATION_ID):
            ctx["parent_correlation_id"] = v
        if v := headers.get(FlextContextHeaders.SERVICE_NAME):
            ctx["service_name"] = v
        if v := headers.get(FlextContextHeaders.USER_ID):
            ctx["user_id"] = v
        return ctx
