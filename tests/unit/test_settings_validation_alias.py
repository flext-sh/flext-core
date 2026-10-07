"""Behavioral contract for FlextSettings override helpers with validation_alias fields.

Regression origin (bd-9sg): a subclass field declared only via ``validation_alias``
with ``extra='forbid'`` must survive ``update_global`` / ``clone`` revalidation
without raising "Extra inputs are not permitted". These tests assert the observable
public contract only (return values, singleton propagation, error raising) — never
internal revalidation mechanics.

Public contract under test (flext_core.FlextSettings):
- fetch_global() -> shared singleton
- update_global(**overrides) -> new singleton, propagates to fetch_global()
- clone(**overrides) -> isolated copy, does not mutate the global
- validate_overrides / update_global reject unknown field keys with ValueError

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

import pytest

from flext_core import FlextSettings, m, t


class TestsFlextCoreSettingsValidationAlias:
    """Public override-helper behavior for settings carrying validation_alias fields."""

    class _AliasFieldSettings(FlextSettings):
        """Minimal subclass: one field declared only via validation_alias."""

        model_config = m.SettingsConfigDict(extra="forbid", populate_by_name=False)

        pandoc_bin: Annotated[
            str,
            m.Field(validation_alias=t.AliasChoices("PANDOC", "FLEXT_PANDOC")),
        ] = "pandoc"

    @staticmethod
    def setup_method() -> None:
        """Provide ``setup_method``."""
        TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.reset_for_testing()

    @staticmethod
    def teardown_method() -> None:
        """Provide ``teardown_method``."""
        TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.reset_for_testing()

    @staticmethod
    def test_default_value_when_no_override_applied() -> None:
        # Arrange / Act
        """Test default value when no override applied."""
        settings = (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global()
        )

        # Assert — declared default surfaces through the public field.
        assert settings.pandoc_bin == "pandoc"

    @pytest.mark.parametrize(
        "override_value",
        ["custom_pandoc", "/usr/bin/pandoc", "pandoc-3.1", "pandoc"],
    )
    @staticmethod
    def test_update_global_applies_and_propagates_override(
        override_value: str,
    ) -> None:
        # Act — must not raise "Extra inputs are not permitted".
        """Test update global applies and propagates override."""
        returned = (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.update_global(
                pandoc_bin=override_value,
            )
        )

        # Assert — returned value carries the override AND it propagates.
        assert returned.pandoc_bin == override_value
        assert (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global().pandoc_bin
            == override_value
        )

    @staticmethod
    def test_update_global_is_idempotent_across_repeated_calls() -> None:
        # Act
        """Test update global is idempotent across repeated calls."""
        first = TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.update_global(
            pandoc_bin="pandoc-a",
        )
        second = (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.update_global(
                pandoc_bin="pandoc-a",
            )
        )

        # Assert — repeated identical override yields the same observable state.
        assert first.pandoc_bin == "pandoc-a"
        assert second.pandoc_bin == "pandoc-a"
        assert (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global().pandoc_bin
            == "pandoc-a"
        )

    @staticmethod
    def test_clone_override_does_not_mutate_global_singleton() -> None:
        # Arrange
        """Test clone override does not mutate global singleton."""
        base = TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global()

        # Act
        cloned = base.clone(pandoc_bin="cloned_pandoc")

        # Assert — clone is isolated; global keeps its prior value.
        assert cloned.pandoc_bin == "cloned_pandoc"
        assert base.pandoc_bin == "pandoc"
        assert (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global().pandoc_bin
            == "pandoc"
        )

    @staticmethod
    def test_clone_without_overrides_is_independent_copy() -> None:
        # Arrange
        """Test clone without overrides is independent copy."""
        TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.update_global(
            pandoc_bin="global_pandoc",
        )
        base = TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global()

        # Act
        copy = base.clone()

        # Assert — value equality but distinct instances (deep copy contract).
        assert copy.pandoc_bin == "global_pandoc"
        assert copy is not base

    @staticmethod
    def test_fetch_global_overrides_yield_isolated_snapshot() -> None:
        # Arrange — materialize the singleton so overrides route through clone.
        """Test fetch global overrides yield isolated snapshot."""
        TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global()

        # Act — overrides on fetch_global must not touch the shared singleton.
        snapshot = (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global(
                overrides={"pandoc_bin": "snap"},
            )
        )

        # Assert
        assert snapshot.pandoc_bin == "snap"
        assert (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.fetch_global().pandoc_bin
            == "pandoc"
        )

    @staticmethod
    def test_model_dump_exposes_override_through_public_field_name() -> None:
        # Arrange
        """Test model dump exposes override through public field name."""
        settings = (
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.update_global(
                pandoc_bin="dumped_pandoc",
            )
        )

        # Act
        dumped = settings.model_dump()

        # Assert — public serialization keys on the field name, not the alias.
        assert dumped["pandoc_bin"] == "dumped_pandoc"

    @staticmethod
    def test_unknown_override_key_raises_value_error() -> None:
        # Act / Assert — typo guard rejects undeclared fields at the boundary.
        """Test unknown override key raises value error."""
        with pytest.raises(ValueError, match="Unknown settings override"):
            TestsFlextCoreSettingsValidationAlias._AliasFieldSettings.update_global(
                not_a_field="x",
            )
