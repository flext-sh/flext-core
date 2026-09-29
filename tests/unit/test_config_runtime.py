"""Behavior contract for flext_core config/settings — ADR-005 canonical singletons.

Asserts the observable contract: ``config`` and ``settings`` are pre-instantiated
singletons imported directly (``from flext_core import config, settings``), ``config``
rejects mutation, and each settings subclass owns an independent singleton slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import tm

from flext_core import FlextConfig, FlextSettings, config, settings
from tests import m

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextCoreConfigSettingsCanonical:
    """Public-contract behaviour for the canonical config/settings singletons."""

    def test_config_is_preinstantiated_frozen_singleton(self) -> None:
        """S1: ``config`` is a ready-to-use frozen FlextConfig instance; mutation raises."""
        assert isinstance(config, FlextConfig)
        tm.rejects_assignment(config, "anything", "mutated", expected=m.ValidationError)

    def test_settings_is_preinstantiated_usable_singleton(self) -> None:
        """S2: ``settings`` is a ready-to-use FlextSettings instance used directly."""
        assert isinstance(settings, FlextSettings)
        assert isinstance(settings.model_dump(), dict)

    def test_config_subclasses_keep_independent_singletons(
        self, tmp_path: Path
    ) -> None:
        """Creating and resetting a child never borrows or resets its parent slot."""

        class ParentConfig(FlextConfig):
            CONFIG_DIR: str = str(tmp_path)

        parent = ParentConfig.fetch_global()

        class ChildConfig(ParentConfig):
            pass

        class SiblingConfig(ParentConfig):
            pass

        child = ChildConfig.fetch_global()
        sibling = SiblingConfig.fetch_global()
        assert type(parent) is ParentConfig
        assert type(child) is ChildConfig
        assert type(sibling) is SiblingConfig
        assert ChildConfig.fetch_global() is child
        ChildConfig.reset_for_testing()
        assert ChildConfig.fetch_global() is not child
        assert ParentConfig.fetch_global() is parent
        assert SiblingConfig.fetch_global() is sibling
