"""High-performance serialization adapters using msgspec."""

import json
from abc import ABC, abstractmethod
from typing import Any


class SerializerInterface(ABC):
    """Interface for serializers."""

    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""


class HighPerformanceSerializer(SerializerInterface):
    """High-performance serializer implementation."""

    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON bytes."""
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to data."""
        return json.loads(data.decode("utf-8"))

    def serialize_to_str(self, data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data, default=str)

    def deserialize_from_str(self, data: str) -> Any:
        """Deserialize JSON string to data."""
        return json.loads(data)


_default_serializer = HighPerformanceSerializer()


def get_serializer() -> HighPerformanceSerializer:
    """Get the default serializer instance."""
    return _default_serializer
