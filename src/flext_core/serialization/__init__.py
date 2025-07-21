"""High-performance serialization utilities for FLEXT Core.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides msgspec-based serialization for optimal performance
in data-intensive operations like WebSocket communication and API responses.
"""

from __future__ import annotations

from flext_core.serialization.msgspec_adapters import get_serializer

__all__ = ["get_serializer"]
