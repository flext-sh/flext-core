"""Example 11 service models."""

from __future__ import annotations

from typing import override

from flext_core import FlextSettings, r, t


class Ex11HandlerLikeService(FlextSettings):
    """Service-like handler stub."""

    @classmethod
    @override
    def validate(cls, value: t.ContainerValue) -> Ex11HandlerLikeService:
        """Validate service payload."""
        return cls.model_validate(value)

    def can_handle(self, message_type: type) -> bool:
        """Check whether message type is handled."""
        return bool(message_type)

    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        """Handle service message."""
        return r[t.ContainerValue].ok(message)
