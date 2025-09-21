"""Simplified mixins tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextMixins, FlextModels


class TestMixinsSimple:
    """Test mixins functionality."""

    def test_serializable_exists(self) -> None:
        """Test that Serializable mixin exists."""
        assert hasattr(FlextMixins, "Serializable")

    def test_loggable_exists(self) -> None:
        """Test that Loggable mixin exists."""
        assert hasattr(FlextMixins, "Loggable")

    def test_to_json_works(self) -> None:
        """Test that to_json method works."""
        request = FlextModels.SerializationRequest(data={"test": "data"})
        result = FlextMixins.to_json(request)
        assert "test" in result
