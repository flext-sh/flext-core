"""Behavioral contract for the project-metadata constants exposed flat on ``c.*``.

Source under test: ``flext_core._constants.project_metadata`` — surfaced through
the ``FlextConstants`` facade and re-exposed on the tests ``c`` facade. These
tests assert the *public* constant contract a caller depends on (values, mapping
contents, immutability), never the module's internal layout.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import pytest

from tests.constants import c


class TestsFlextCoreConstantsProjectMetadata:
    """Public contract for fixed project-metadata constants."""

    def test_pyproject_filename_is_the_standard_manifest_name(self) -> None:
        # Arrange / Act
        filename = c.PYPROJECT_FILENAME

        # Assert — callers use this to locate a project's manifest on disk.
        assert filename == "pyproject.toml"

    @pytest.mark.parametrize(
        ("project_name", "expected_facade"),
        [
            ("flext", "FlextRoot"),
            ("flext-core", "Flext"),
        ],
    )
    def test_special_name_overrides_maps_project_to_canonical_facade(
        self,
        project_name: str,
        expected_facade: str,
    ) -> None:
        # Act
        override = c.SPECIAL_NAME_OVERRIDES[project_name]

        # Assert — the override table drives facade-class naming for these roots.
        assert override == expected_facade

    def test_special_name_overrides_reports_membership_for_known_and_unknown_keys(
        self,
    ) -> None:
        # Assert — only the declared roots are overridden; anything else is absent
        # so callers fall back to their default naming strategy.
        assert "flext" in c.SPECIAL_NAME_OVERRIDES
        assert "some-unknown-project" not in c.SPECIAL_NAME_OVERRIDES

    def test_special_name_overrides_exposes_exactly_the_declared_roots(self) -> None:
        # Assert — the full public contents, independent of insertion order.
        assert dict(c.SPECIAL_NAME_OVERRIDES) == {
            "flext": "FlextRoot",
            "flext-core": "Flext",
        }

    def test_special_name_overrides_is_a_read_only_mapping(self) -> None:
        # Assert — callers may safely share the constant: it is a read-only
        # Mapping, not a MutableMapping, so no consumer can corrupt it.
        assert isinstance(c.SPECIAL_NAME_OVERRIDES, Mapping)
        assert not isinstance(c.SPECIAL_NAME_OVERRIDES, MutableMapping)

    def test_unknown_override_lookup_raises_key_error(self) -> None:
        # Assert — absent keys raise, so callers must handle the miss explicitly
        # rather than receiving an invented default.
        with pytest.raises(KeyError):
            _ = c.SPECIAL_NAME_OVERRIDES["not-a-flext-project"]
