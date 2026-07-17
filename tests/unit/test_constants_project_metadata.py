"""Behavioral contract for the project-metadata constants exposed flat on ``c.*``.

Source under test: ``flext_core._constants.project_metadata`` — surfaced through
the ``FlextConstants`` facade and re-exposed on the tests ``c`` facade. These
tests assert the *public* constant contract a caller depends on (values, override
table contents, immutability), never the module's internal layout.
"""

from __future__ import annotations

import pytest

from tests import c


class TestsFlextCoreConstantsProjectMetadata:
    """Public contract for fixed project-metadata constants."""

    def test_pyproject_filename_is_the_standard_manifest_name(self) -> None:
        assert c.PYPROJECT_FILENAME == "pyproject.toml"

    @pytest.mark.parametrize(
        ("project_name", "expected_facade"),
        [("flext", "FlextRoot"), ("flext-core", "Flext")],
    )
    def test_special_name_overrides_maps_project_to_canonical_facade(
        self, project_name: str, expected_facade: str
    ) -> None:
        # The override table is an immutable tuple of (name, facade) pairs;
        # callers materialize it as a mapping to resolve a root's facade class.
        assert dict(c.SPECIAL_NAME_OVERRIDES)[project_name] == expected_facade

    def test_special_name_overrides_reports_membership_for_known_and_unknown_keys(
        self,
    ) -> None:
        overrides = dict(c.SPECIAL_NAME_OVERRIDES)
        assert "flext" in overrides
        assert "some-unknown-project" not in overrides

    def test_special_name_overrides_exposes_exactly_the_declared_roots(self) -> None:
        assert dict(c.SPECIAL_NAME_OVERRIDES) == {
            "flext": "FlextRoot",
            "flext-core": "Flext",
        }

    def test_special_name_overrides_is_an_immutable_tuple(self) -> None:
        # Callers may safely share the constant: it is an immutable tuple of
        # pairs, so no consumer can corrupt the override table.
        assert isinstance(c.SPECIAL_NAME_OVERRIDES, tuple)
        assert all(isinstance(pair, tuple) for pair in c.SPECIAL_NAME_OVERRIDES)

    def test_unknown_override_lookup_is_absent(self) -> None:
        # Absent roots are simply not present, so callers fall back to their
        # default naming strategy rather than receiving an invented value.
        assert "not-a-flext-project" not in dict(c.SPECIAL_NAME_OVERRIDES)
