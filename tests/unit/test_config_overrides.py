"""Tests ensuring FlextConfig overrides propagate to dependent helpers."""

from __future__ import annotations

from flext_core.config import FlextConfig
from flext_core.loggings import FlextLogger
from flext_core.processing import FlextProcessing


class TestFlextConfigOverrides:
    """Verify that FlextConfig values influence processing and logging helpers."""

    def setup_method(self) -> None:  # noqa: D401 - pytest fixture style
        """Reset the global configuration before each test."""

        FlextConfig.reset_global_instance()

    def teardown_method(self) -> None:  # noqa: D401 - pytest fixture style
        """Ensure the global configuration is cleared after each test."""

        FlextConfig.reset_global_instance()

    def test_processing_helpers_pick_up_overrides(self) -> None:
        """FlextProcessing.Config should reflect runtime overrides."""

        config = FlextConfig.create(
            timeout_seconds=42,
            max_batch_size=256,
            max_handlers=25,
            log_verbosity="concise",
        )
        FlextConfig.set_global_instance(config)

        assert FlextProcessing.Config.get_default_timeout() == 42.0
        assert FlextProcessing.Config.get_max_batch_size() == 256
        assert FlextProcessing.Config.get_max_handlers() == 25

        logging_config = FlextLogger.get_configuration()
        assert logging_config["log_verbosity"] == "concise"

    def test_default_timeout_alias_updates_primary_field(self) -> None:
        """Updating the alias keeps ``timeout_seconds`` synchronized."""

        config = FlextConfig()
        config.default_timeout = 17
        assert config.timeout_seconds == 17
