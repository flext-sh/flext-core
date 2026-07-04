"""Project metadata read utility tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.unit._project_metadata_support import write_pyproject
from tests.utilities import u

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextUtilitiesProjectMetadataRead:
    @pytest.mark.parametrize(
        ("project_name", "expected_stem"),
        [
            (c.Tests.SAMPLE_PROJECT_NAME, c.Tests.SAMPLE_PROJECT_CLASS_STEM),
            ("flext-core", "Flext"),
        ],
    )
    def test_derive_class_stem_produces_pascal_case_from_project_name(
        self,
        project_name: str,
        expected_stem: str,
    ) -> None:
        tm.that(u.derive_class_stem(project_name), eq=expected_stem)

    def test_pascalize_converts_dashes_and_underscores_to_camel_case(self) -> None:
        tm.that(
            m.pascalize(c.Tests.SAMPLE_PROJECT_NAME),
            eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM,
        )
        tm.that(m.pascalize("flext_ldif"), eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM)
        tm.that(m.pascalize(""), eq="")

    def test_derive_class_stem_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            u.derive_class_stem("")

    def test_read_project_metadata_parses_minimal_pyproject(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"
            description = "LDIF"
            """,
        )
        meta = u.read_project_metadata(root)
        tm.that(meta, is_=m.ProjectMetadata)
        tm.that(meta.name, eq=c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(meta.class_stem, eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM)

    def test_read_project_metadata_accepts_spdx_license_dict(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = {{text = "{c.Tests.SAMPLE_PROJECT_LICENSE}"}}
            """,
        )
        tm.that(
            u.read_project_metadata(root).license,
            eq=c.Tests.SAMPLE_PROJECT_LICENSE,
        )

    def test_read_project_metadata_extracts_author_names_from_project_table(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"
            authors = [
                {{name = "{c.Tests.SAMPLE_AUTHOR_ALICE}", email = "alice@example.com"}},
                {{name = "{c.Tests.SAMPLE_AUTHOR_BOB}"}},
            ]
            """,
        )
        tm.that(
            u.read_project_metadata(root).authors,
            eq=(c.Tests.SAMPLE_AUTHOR_ALICE, c.Tests.SAMPLE_AUTHOR_BOB),
        )

    def test_read_project_constants_returns_installed_project_metadata_values(
        self,
    ) -> None:
        constants = u.read_project_constants("flext-core")
        tm.that(constants.PACKAGE_NAME, eq="flext-core")
        tm.that(constants.PYTHON_PACKAGE_NAME, eq="flext_core")
        tm.that(constants.CLASS_STEM, eq="Flext")
        tm.that("c" in constants.RUNTIME_ALIAS_NAMES, eq=True)

    def test_read_project_metadata_raises_on_missing_pyproject(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            u.read_project_metadata(tmp_path)

    @pytest.mark.parametrize(
        ("body", "match_pattern"),
        [
            ('[project]\nversion="0.12.0"\n', r"missing.*name"),
            ('[project]\nname="x"\n', r"missing.*version"),
        ],
        ids=["missing_name", "missing_version"],
    )
    def test_read_project_metadata_rejects_incomplete_pyproject(
        self,
        tmp_path: Path,
        body: str,
        match_pattern: str,
    ) -> None:
        root = write_pyproject(tmp_path, body)
        with pytest.raises(ValueError, match=match_pattern):
            u.read_project_metadata(root)
