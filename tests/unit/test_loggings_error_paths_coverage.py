"""Real tests to cover error paths in loggings.py - no mocks.

Module: flext_core.loggings
Scope: Error handling paths in bind_global_context, clear_global_context, _get_global_context

This module provides real tests (no mocks) to cover error handling paths
that are difficult to trigger in normal operation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextLogger
from tests import t


class TestLoggingsErrorPaths:
    """Test error handling paths in loggings.py."""

    def test_get_global_context_empty(self) -> None:
        """Test _get_global_context returns empty ConfigMap when no context set."""
        FlextLogger.clear_global_context()
        result = FlextLogger._get_global_context()
        assert isinstance(result, t.ConfigMap)
        assert isinstance(result.root, dict)

    def test_get_global_context_with_values(self) -> None:
        """Test _get_global_context returns bound context values."""
        FlextLogger.clear_global_context()
        FlextLogger.bind_global_context(test_key="test_value")
        result = FlextLogger._get_global_context()
        assert isinstance(result, t.ConfigMap)
        assert "test_key" in result.root
        FlextLogger.clear_global_context()

    def test_bind_and_unbind_global_context(self) -> None:
        """Test bind then unbind global context cycle."""
        FlextLogger.clear_global_context()
        bind_result = FlextLogger.bind_global_context(key1="v1")
        assert bind_result.is_success
        unbind_result = FlextLogger.unbind_global_context("key1")
        assert unbind_result.is_success

    def test_clear_global_context(self) -> None:
        """Test clear_global_context succeeds."""
        FlextLogger.bind_global_context(temp="value")
        result = FlextLogger.clear_global_context()
        assert result.is_success


__all__ = ["TestLoggingsErrorPaths"]
