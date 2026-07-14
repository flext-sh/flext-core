"""Decorator facade implementation."""

from __future__ import annotations

from ._runtime import FlextDecoratorsRuntime


class FlextDecorators(FlextDecoratorsRuntime):
    """Automation decorators for infrastructure concerns."""


__all__: tuple[str, ...] = ("FlextDecorators",)
