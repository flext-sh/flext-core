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
"""

from __future__ import annotations

from typing import Annotated

import pytest

from flext_core import FlextSettings, m, t


class _AliasFieldSettings(FlextSettings):
    """Minimal subclass: one field declared only via validation_alias (no populate_by_name)."""

    model_config = m.SettingsConfigDict(extra="forbid", populate_by_name=False)

    pandoc_bin: Annotated[
        str, m.Field(validation_alias=t.AliasChoices("PANDOC", "FLEXT_PANDOC"))
    ] = "pandoc"


class TestsFlextCoreSettingsValidationAlias:
    """Public override-helper behavior for settings carrying validation_alias fields."""

    def setup_method(self) -> None:
        _AliasFieldSettings.reset_for_testing()

    def teardown_method(self) -> None:
        _AliasFieldSettings.reset_for_testing()

    def test_default_value_when_no_override_applied(self) -> None:
        # Arrange / Act
        settings = _AliasFieldSettings.fetch_global()

        # Assert — declared default surfaces through the public field.
        assert settings.pandoc_bin == "pandoc"

    @pytest.mark.parametrize(
        "override_value", ["custom_pandoc", "/usr/bin/pandoc", "pandoc-3.1", "pandoc"]
    )
    def test_update_global_applies_and_propagates_override(
        self, override_value: str
    ) -> None:
        # Act — must not raise "Extra inputs are not permitted".
        returned = _AliasFieldSettings.update_global(pandoc_bin=override_value)

        # Assert — returned value carries the override AND it propagates.
        assert returned.pandoc_bin == override_value
        assert _AliasFieldSettings.fetch_global().pandoc_bin == override_value

    def test_update_global_is_idempotent_across_repeated_calls(self) -> None:
        # Act
        first = _AliasFieldSettings.update_global(pandoc_bin="pandoc-a")
        second = _AliasFieldSettings.update_global(pandoc_bin="pandoc-a")

        # Assert — repeated identical override yields the same observable state.
        assert first.pandoc_bin == "pandoc-a"
        assert second.pandoc_bin == "pandoc-a"
        assert _AliasFieldSettings.fetch_global().pandoc_bin == "pandoc-a"

    def test_clone_override_does_not_mutate_global_singleton(self) -> None:
        # Arrange
        base = _AliasFieldSettings.fetch_global()

        # Act
        cloned = base.clone(pandoc_bin="cloned_pandoc")

        # Assert — clone is isolated; global keeps its prior value.
        assert cloned.pandoc_bin == "cloned_pandoc"
        assert base.pandoc_bin == "pandoc"
        assert _AliasFieldSettings.fetch_global().pandoc_bin == "pandoc"

    def test_clone_without_overrides_is_independent_copy(self) -> None:
        # Arrange
        _AliasFieldSettings.update_global(pandoc_bin="global_pandoc")
        base = _AliasFieldSettings.fetch_global()

        # Act
        copy = base.clone()

        # Assert — value equality but distinct instances (deep copy contract).
        assert copy.pandoc_bin == "global_pandoc"
        assert copy is not base

    def test_fetch_global_overrides_yield_isolated_snapshot(self) -> None:
        # Arrange — materialize the singleton so overrides route through clone.
        _AliasFieldSettings.fetch_global()

        # Act — overrides on fetch_global must not touch the shared singleton.
        snapshot = _AliasFieldSettings.fetch_global(overrides={"pandoc_bin": "snap"})

        # Assert
        assert snapshot.pandoc_bin == "snap"
        assert _AliasFieldSettings.fetch_global().pandoc_bin == "pandoc"

    def test_model_dump_exposes_override_through_public_field_name(self) -> None:
        # Arrange
        settings = _AliasFieldSettings.update_global(pandoc_bin="dumped_pandoc")

        # Act
        dumped = settings.model_dump()

        # Assert — public serialization keys on the field name, not the alias.
        assert dumped["pandoc_bin"] == "dumped_pandoc"

    def test_unknown_override_key_raises_value_error(self) -> None:
        # Act / Assert — typo guard rejects undeclared fields at the boundary.
        with pytest.raises(ValueError, match="Unknown settings override"):
            _AliasFieldSettings.update_global(not_a_field="x")


__all__: t.MutableSequenceOf[str] = ["TestsFlextCoreSettingsValidationAlias"]
