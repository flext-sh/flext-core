"""Serialization module for FLEXT Core."""

from flext_core.serialization.msgspec_adapters import (
    HighPerformanceSerializer,
    get_serializer,
)

__all__ = [
    "HighPerformanceSerializer",
    "get_serializer",
]
