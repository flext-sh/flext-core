"""Behavior contract for flext_core config/settings — ADR-005 canonical singletons.

Asserts the observable contract: ``config`` and ``settings`` are pre-instantiated
singletons imported directly (``from flext_core import config, settings``), ``config``
rejects mutation, and each settings subclass owns an independent singleton slot.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextConfig, FlextCore, FlextSettings, config, settings
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
        tm.that(isinstance(config, FlextConfig), truth=True)
        tm.rejects_assignment(config, "anything", "mutated", expected=m.ValidationError)

    @staticmethod
    def test_settings_is_preinstantiated_usable_singleton() -> None:
        """S2: ``settings`` is a ready-to-use FlextSettings instance used directly."""
        tm.that(isinstance(settings, FlextSettings), truth=True)
        tm.that(isinstance(settings.model_dump(), dict), truth=True)

    @staticmethod
    def test_custom_runtime_settings_preserve_frozen_config_and_defaults() -> None:
        """Runtime overrides do not merge or mutate the independent config owner."""
        configuration = config.model_dump()
        default_debug = settings.debug
        customized = settings.clone(debug=not default_debug)
        facade = FlextCore(runtime_settings=customized)

        tm.that(facade.settings is customized, eq=True)
        tm.that(settings.debug, eq=default_debug)
        tm.that(config.model_dump(), eq=configuration)
        tm.rejects_assignment(config, "anything", "mutated", expected=m.ValidationError)

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
        tm.that(type(parent) is ParentConfig, truth=True)
        tm.that(type(child) is ChildConfig, truth=True)
        tm.that(type(sibling) is SiblingConfig, truth=True)
        tm.that(ChildConfig.fetch_global() is child, truth=True)
        ChildConfig.reset_for_testing()
        tm.that(ChildConfig.fetch_global() is not child, truth=True)
        tm.that(ParentConfig.fetch_global() is parent, truth=True)
        tm.that(SiblingConfig.fetch_global() is sibling, truth=True)
