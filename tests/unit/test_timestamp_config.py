"""Tests covering TimestampConfig defaults sourced from FlextConfig."""

from __future__ import annotations

from datetime import datetime

from flext_core import FlextConfig, FlextMixins, FlextModels


class TestTimestampConfigPropagation:
    """Validate that timestamp configuration honors global settings."""

    def test_timestamp_config_reflects_custom_auto_update(self) -> None:
        """A custom timestamp_auto_update must propagate to TimestampConfig."""

        original_config = FlextConfig.get_global_instance()
        try:
            custom_config = FlextConfig.create(
                use_utc_timestamps=False,
                timestamp_auto_update=True,
            )
            FlextConfig.set_global_instance(custom_config)

            target = type("TimestampTarget", (), {})()
            timestamp_config = FlextModels.TimestampConfig(obj=target)

            assert timestamp_config.use_utc is False
            assert timestamp_config.auto_update is True
        finally:
            FlextConfig.set_global_instance(original_config)

    def test_create_timestamp_fields_respects_auto_update(self) -> None:
        """Mixins must honour custom auto_update defaults when creating fields."""

        original_config = FlextConfig.get_global_instance()
        try:
            custom_config = FlextConfig.create(
                use_utc_timestamps=True,
                timestamp_auto_update=False,
            )
            FlextConfig.set_global_instance(custom_config)

            class TimestampTarget:
                def __init__(self) -> None:
                    self.created_at = None
                    self.updated_at = None

            target = TimestampTarget()
            timestamp_config = FlextModels.TimestampConfig(obj=target)

            FlextMixins.create_timestamp_fields(timestamp_config)

            assert isinstance(target.created_at, datetime)
            assert target.updated_at is None
        finally:
            FlextConfig.set_global_instance(original_config)
