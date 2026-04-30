"""Behavior contract for fixed c.* project-metadata constants."""

from __future__ import annotations

from tests import c


class TestsFlextConstantsProjectMetadata:
    """Behavior contract for fixed project-metadata constants."""

    def test_pyproject_filename_is_the_only_fixed_project_metadata_constant(
        self,
    ) -> None:
        assert c.PYPROJECT_FILENAME == "pyproject.toml"
