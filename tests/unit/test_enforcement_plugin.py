"""Tests for the flext-core pytest enforcement plugin.

The plugin is auto-registered via the ``flext_core_enforcement`` pytest11
entry-point. These tests assert the entry-point exists and that loading the
module registers the expected runtime-warning contribution with the central
flext-tests dispatcher.
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from flext_tests._fixtures._enforcement_parts.registry import (
    clear,
    get,
    warning_categories,
)

from flext_core._constants.enforcement import (
    FlextMroViolation,
    FlextSmellViolation,
)
from flext_core._fixtures import enforcement as flext_core_enforcement_plugin


class TestsFlextCoreEnforcementPlugin:
    """Entry-point and registration contract for the flext-core plugin."""

    @pytest.fixture
    def _clear_registry(self) -> None:
        """Keep the shared contribution registry isolated between cases."""
        clear()
        yield
        clear()

    def test_pytest11_entry_point_is_registered(self, _clear_registry: None) -> None:
        """The plugin is discoverable through the pytest11 entry-point group."""
        eps = entry_points(group="pytest11")
        names = {ep.name for ep in eps}

        assert "flext_core_enforcement" in names

    def test_plugin_registers_runtime_warning_contribution(
        self,
        _clear_registry: None,
    ) -> None:
        """Loading the module registers the runtime-warning contribution."""
        flext_core_enforcement_plugin._register()

        contribution = get("flext_core_runtime_warning")
        assert contribution is not None
        assert contribution.source_kind == "runtime_warning"
        assert contribution.builder is None
        assert set(contribution.warning_categories) == {
            FlextMroViolation,
            FlextSmellViolation,
        }

    def test_warning_categories_are_exposed_to_dispatcher(
        self,
        _clear_registry: None,
    ) -> None:
        """The registered categories participate in the dispatcher union."""
        flext_core_enforcement_plugin._register()

        assert FlextMroViolation in warning_categories()
        assert FlextSmellViolation in warning_categories()
