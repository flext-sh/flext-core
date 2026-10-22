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

from flext_core import FlextConfig, FlextCore, FlextSettings, config, settings
from tests import m, tm, u

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextCoreConfigSettingsCanonical:
    """Public-contract behaviour for the canonical config/settings singletons."""

    @staticmethod
    def test_config_is_preinstantiated_frozen_singleton() -> None:
        """``config`` is a ready-to-use frozen ``FlextConfig`` instance.

        S1: mutation raises.
        """
        tm.that(config, is_=FlextConfig)
        tm.rejects_assignment(config, "anything", "mutated", expected=m.ValidationError)

    @staticmethod
    def test_settings_is_preinstantiated_usable_singleton() -> None:
        """S2: ``settings`` is a ready-to-use FlextSettings instance used directly."""
        tm.that(settings, is_=FlextSettings)
        tm.that(settings.model_dump(), is_=dict)

    @staticmethod
    def test_custom_runtime_settings_preserve_frozen_config_and_defaults() -> None:
        """Runtime overrides do not merge or mutate the independent config owner."""
        configuration = config.model_dump()
        default_debug = settings.debug
        customized = settings.clone(debug=not default_debug)
        facade = FlextCore(runtime_settings=customized)

        tm.ok(facade.settings is customized)
        tm.that(settings.debug, eq=default_debug)
        tm.that(config.model_dump(), eq=configuration)
        tm.rejects_assignment(config, "anything", "mutated", expected=m.ValidationError)

    @staticmethod
    def test_config_subclasses_keep_independent_singletons(
        tmp_path: Path,
    ) -> None:
        """Creating and resetting a child never borrows or resets its parent slot."""

        class ParentConfig(FlextConfig):
            marker: str = "default"

        with u.Tests.env_vars_context(
            env_vars={"TESTS_CONFIG_DIR": str(tmp_path)},
        ):
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
        tm.that(ChildConfig.fetch_global() is child, truth=True)
        tm.that(ParentConfig.fetch_global() is parent, truth=True)
        tm.that(SiblingConfig.fetch_global() is sibling, truth=True)

    @staticmethod
    def test_imported_config_repoints_between_yaml_roots_and_restores(
        tmp_path: Path,
    ) -> None:
        """The exported frozen root keeps identity across real source relocations."""
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "rules.yaml").write_text(
            "lifecycle_marker: first\nfirst_only: true\n", encoding="utf-8"
        )
        (second_dir / "rules.yaml").write_text(
            "lifecycle_marker: second\nsecond_only: true\n", encoding="utf-8"
        )
        original = config.model_dump()
        try:
            for directory, marker, absent in (
                (first_dir, "first", "second_only"),
                (second_dir, "second", "first_only"),
            ):
                with u.Tests.env_vars_context(
                    env_vars={
                        "FLEXT_CORE_CONFIG_DIR": str(directory),
                        "XDG_CONFIG_HOME": str(tmp_path / "preferences"),
                    },
                ):
                    FlextConfig.reset_for_testing()
                    tm.ok(config is FlextConfig.fetch_global())
                    tm.that(config.model_dump()["lifecycle_marker"], eq=marker)
                    tm.that(config.model_dump(), lacks=absent)
                    tm.rejects_assignment(
                        config,
                        "lifecycle_marker",
                        "mutated",
                        expected=m.ValidationError,
                    )
        finally:
            FlextConfig.reset_for_testing()
        tm.ok(config is FlextConfig.fetch_global())
        tm.that(config.model_dump(), eq=original)

    @staticmethod
    @pytest.mark.parametrize("contents", ["limit: invalid\n", "limit: -1\n"])
    def test_invalid_config_reset_preserves_published_value_and_frozen_state(
        tmp_path: Path,
        contents: str,
    ) -> None:
        """Typed YAML failures cannot replace or partially mutate a bound config."""

        class ValidatedConfig(FlextConfig):
            limit: int = m.Field(default=1, ge=0)

        directory = tmp_path / "rules"
        directory.mkdir()
        rules = directory / "rules.yaml"
        rules.write_text("limit: 3\n", encoding="utf-8")
        with u.Tests.env_vars_context(
            env_vars={
                "TESTS_CONFIG_DIR": str(directory),
                "XDG_CONFIG_HOME": str(tmp_path / "preferences"),
            },
        ):
            published = ValidatedConfig.fetch_global()
            snapshot = published.model_copy(deep=True)
            original = published.model_dump()
            rules.write_text(contents, encoding="utf-8")
            with pytest.raises(m.ValidationError, match="limit"):
                ValidatedConfig.reset_for_testing()
            with pytest.raises(m.ValidationError, match="limit"):
                ValidatedConfig()
            tm.ok(published is ValidatedConfig.fetch_global())
            tm.that(published.model_dump(), eq=original)
            tm.rejects_assignment(published, "limit", 4, expected=m.ValidationError)
            selected_limit = snapshot.limit + 2
            rules.write_text(f"limit: {selected_limit}\n", encoding="utf-8")
            ValidatedConfig.reset_for_testing()
            tm.ok(published is ValidatedConfig.fetch_global())
            tm.that(published.limit, eq=selected_limit)
            tm.that(snapshot.model_dump(), eq=original)

    @staticmethod
    def test_unused_config_stays_lazy_and_failed_construction_never_publishes(
        tmp_path: Path,
    ) -> None:
        """Uninitialized classes remain unpublished until a successful fetch."""

        class RequiredConfig(FlextConfig):
            required_value: int

        directory = tmp_path / "rules"
        directory.mkdir()
        rules = directory / "rules.yaml"
        rules.write_text("required_value: invalid\n", encoding="utf-8")
        with u.Tests.env_vars_context(
            env_vars={
                "TESTS_CONFIG_DIR": str(directory),
                "XDG_CONFIG_HOME": str(tmp_path / "preferences"),
            },
        ):
            RequiredConfig.reset_for_testing()
            with pytest.raises(m.ValidationError, match="required_value"):
                RequiredConfig.model_validate({})
            with pytest.raises(m.ValidationError, match="required_value"):
                RequiredConfig.fetch_global()
            rules.write_text("required_value: 7\n", encoding="utf-8")
            direct = RequiredConfig(required_value=7)
            selected_value = direct.required_value + 2
            rules.write_text(f"required_value: {selected_value}\n", encoding="utf-8")
            published = RequiredConfig.fetch_global()
            tm.ok(published is not direct)
            tm.that(published.required_value, eq=selected_value)
