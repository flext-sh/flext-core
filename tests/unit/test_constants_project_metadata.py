"""Behavior contract for c.* project-metadata constants."""

from __future__ import annotations

from tests import c


class TestsFlextConstantsProjectMetadata:
    """Behavior contract for fixed project-metadata constants."""

    def test_project_metadata_constants_are_flat_on_c(self) -> None:
        assert c.PYPROJECT_FILENAME == "pyproject.toml"
