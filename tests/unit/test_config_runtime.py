"""Behavior contract for flext_core config/settings — ADR-005 canonical singletons.

Asserts the observable contract: ``config`` and ``settings`` are pre-instantiated
singletons imported directly (``from flext_core import config, settings``), ``config``
rejects mutation, and each settings subclass owns an independent singleton slot.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import FlextConfig, FlextSettings, config, settings
from tests import m, tm

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextCoreConfigSettingsCanonical:
    """Public-contract behaviour for the canonical config/settings singletons."""

    @staticmethod
    def test_config_is_preinstantiated_frozen_singleton() -> None:
        """``config`` is a ready-to-use frozen ``FlextConfig`` instance.

        S1: mutation raises.
        """
        assert isinstance(config, FlextConfig)
        tm.rejects_assignment(config, "anything", "mutated", expected=m.ValidationError)

    @staticmethod
    def test_settings_is_preinstantiated_usable_singleton() -> None:
        """S2: ``settings`` is a ready-to-use FlextSettings instance used directly."""
        assert isinstance(settings, FlextSettings)
        assert isinstance(settings.model_dump(), dict)

    @staticmethod
    def test_settings_config_composition_preserves_public_contract() -> None:
        """The settings-first diamond retains validation and frozen config state."""

        class ComposedConfig(FlextSettings, FlextConfig):
            pass

        composed = ComposedConfig.fetch_global()
        assert isinstance(composed, FlextSettings)
        assert isinstance(composed, FlextConfig)
        assert ComposedConfig.fetch_global() is composed
        assert composed.model_dump()["debug"] == composed.debug
        tm.rejects_assignment(
            composed,
            "debug",
            not composed.debug,
            expected=m.ValidationError,
        )
        with ComposedConfig.singleton_disabled(), pytest.raises(m.ValidationError):
            ComposedConfig(debug=False, trace=True)
        ComposedConfig.reset_for_testing()
        assert ComposedConfig.fetch_global() is not composed

    @staticmethod
    def test_config_subclasses_keep_independent_singletons(
        tmp_path: Path,
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
