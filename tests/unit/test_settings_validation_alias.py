"""bd-9sg: update_global/clone revalidation must not fail with validation_alias + extra='forbid'.

Repro: a subclass adds a field with AliasChoices validation_alias.
FlextSettingsBase.update_global calls validate_python(__dict__) which uses Python
field names as input keys; with extra='forbid' the validator sees field names as
extras and raises "Extra inputs are not permitted".

Fix target: FlextSettingsBase.clone / FlextSettingsBase.update_global in
flext_core/_settings/_base_parts/flextsettingsbase_part_01.py
"""

from __future__ import annotations

from typing import Annotated

from flext_core import FlextSettings, m, t


class _AliasFieldSettings(FlextSettings):
    """Minimal subclass: one field declared only via validation_alias (no populate_by_name)."""

    model_config = m.SettingsConfigDict(
        extra="forbid",
        populate_by_name=False,
    )

    pandoc_bin: Annotated[
        str,
        m.Field(validation_alias=t.AliasChoices("PANDOC", "FLEXT_PANDOC")),
    ] = "pandoc"


class TestUpdateGlobalWithValidationAlias:
    """update_global and clone must not raise with validation_alias fields."""

    def setup_method(self) -> None:
        _AliasFieldSettings.reset_for_testing()

    def teardown_method(self) -> None:
        _AliasFieldSettings.reset_for_testing()

    def test_update_global_succeeds_with_validation_alias_field(self) -> None:
        """update_global must not raise 'Extra inputs are not permitted'."""
        settings = _AliasFieldSettings.update_global(pandoc_bin="custom_pandoc")
        assert settings.pandoc_bin == "custom_pandoc"

    def test_clone_succeeds_with_validation_alias_field(self) -> None:
        """Clone must not raise 'Extra inputs are not permitted'."""
        base = _AliasFieldSettings.fetch_global()
        cloned = base.clone(pandoc_bin="custom_pandoc_clone")
        assert cloned.pandoc_bin == "custom_pandoc_clone"
        assert base.pandoc_bin == "pandoc"

    def test_update_global_override_wins_over_env(self) -> None:
        """The explicit override value must not be clobbered by attr-reading."""
        settings = _AliasFieldSettings.update_global(pandoc_bin="/usr/bin/pandoc")
        assert settings.pandoc_bin == "/usr/bin/pandoc"
        assert _AliasFieldSettings.fetch_global().pandoc_bin == "/usr/bin/pandoc"


__all__: t.MutableSequenceOf[str] = ["TestUpdateGlobalWithValidationAlias"]
