"""High-performance JSON serialization using msgspec.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Provides optimized JSON serialization for FLEXT applications,
specifically designed for WebSocket and API communication.
"""

from __future__ import annotations

from typing import Any

import msgspec


class MsgspecJSONSerializer:
    """High-performance JSON serializer using msgspec."""

    def __init__(self) -> None:
        """Initialize the msgspec JSON encoder/decoder."""
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder()

    def encode_json_str(self, data: Any) -> str:
        """Encode data to JSON string.

        Args:
            data: Data to encode (dict, list, etc.)

        Returns:
            JSON string representation

        """
        json_bytes = self._encoder.encode(data)
        return json_bytes.decode("utf-8")

    def decode_json_str(self, json_str: str) -> Any:
        """Decode JSON string to Python object.

        Args:
            json_str: JSON string to decode

        Returns:
            Decoded Python object

        """
        return self._decoder.decode(json_str.encode("utf-8"))

    def encode(self, data: Any) -> bytes:
        """Encode data to JSON bytes.

        Args:
            data: Data to encode

        Returns:
            JSON bytes

        """
        return self._encoder.encode(data)

    def decode(self, data: bytes) -> Any:
        """Decode JSON bytes to Python object.

        Args:
            data: JSON bytes to decode

        Returns:
            Decoded Python object

        """
        return self._decoder.decode(data)


def get_serializer() -> MsgspecJSONSerializer:
    """Get the high-performance msgspec JSON serializer.

    Returns:
        Configured msgspec JSON serializer instance

    """
    return MsgspecJSONSerializer()
