"""Simplified mixins tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime

from flext_core import FlextMixins, FlextModels
from flext_core.config import FlextConfig


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

    def test_timestamp_auto_update_disabled_via_config(self) -> None:
        """Ensure disabling auto-update globally prevents timestamp changes."""

        class DummyTimestamp:
            def __init__(self) -> None:
                self.created_at = None
                self.updated_at = None

        original_config = FlextConfig.get_global_instance()
        try:
            disabled_config = original_config.model_copy(
                update={"timestamp_auto_update": False}
            )
            FlextConfig.set_global_instance(disabled_config)

            obj = DummyTimestamp()
            timestamp_config = FlextModels.TimestampConfig(obj=obj)

            FlextMixins.create_timestamp_fields(timestamp_config)
            assert obj.created_at is not None
            assert obj.updated_at is None

            FlextMixins.update_timestamp(timestamp_config)
            assert obj.updated_at is None
        finally:
            FlextConfig.set_global_instance(original_config)

    def test_timestamp_auto_update_enabled_via_config(self) -> None:
        """Ensure enabling auto-update globally updates timestamps automatically."""

        class DummyTimestamp:
            def __init__(self) -> None:
                self.created_at = None
                self.updated_at = datetime(2000, 1, 1, tzinfo=UTC)

        original_config = FlextConfig.get_global_instance()
        try:
            enabled_config = original_config.model_copy(
                update={"timestamp_auto_update": True}
            )
            FlextConfig.set_global_instance(enabled_config)

            obj = DummyTimestamp()
            timestamp_config = FlextModels.TimestampConfig(obj=obj)

            FlextMixins.create_timestamp_fields(timestamp_config)
            assert obj.updated_at is not None

            previous_timestamp = obj.updated_at
            FlextMixins.update_timestamp(timestamp_config)
            assert obj.updated_at is not None
            assert obj.updated_at != previous_timestamp
            assert obj.updated_at > previous_timestamp
        finally:
            FlextConfig.set_global_instance(original_config)
