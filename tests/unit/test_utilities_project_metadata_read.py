"""Project metadata read utility tests.

``u.read_project_metadata(root)`` returns ``r[m.ProjectMetadata]`` — a
Result-wrapped, frozen model whose PEP 621 payload lives under the nested
``project`` field. Tests assert the observable success value and the
Result failure contract for missing/incomplete pyproject inputs.
"""

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
        self, project_name: str, expected_stem: str
    ) -> None:
        tm.that(u.derive_class_stem(project_name), eq=expected_stem)

    def test_derive_class_stem_returns_empty_for_empty_input(self) -> None:
        tm.that(u.derive_class_stem(""), eq="")

    def test_read_project_metadata_parses_minimal_pyproject(
        self, tmp_path: Path
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            description = "LDIF"
            """,
        )
        meta = u.read_project_metadata(root).unwrap()
        tm.that(meta, is_=m.ProjectMetadata)
        tm.that(meta.project.name, eq=c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(meta.class_stem, eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM)

    def test_read_project_metadata_extracts_author_names_from_project_table(
        self, tmp_path: Path
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            authors = [
                {{name = "{c.Tests.SAMPLE_AUTHOR_ALICE}", email = "alice@example.com"}},
                {{name = "{c.Tests.SAMPLE_AUTHOR_BOB}"}},
            ]
            """,
        )
        meta = u.read_project_metadata(root).unwrap()
        tm.that(
            tuple(author.name for author in meta.project.authors),
            eq=(c.Tests.SAMPLE_AUTHOR_ALICE, c.Tests.SAMPLE_AUTHOR_BOB),
        )

    def test_read_project_metadata_derives_package_name_and_stem_from_name(
        self, tmp_path: Path
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            """,
        )
        meta = u.read_project_metadata(root).unwrap()
        tm.that(meta.package_name, eq=c.Tests.SAMPLE_PROJECT_NAME.replace("-", "_"))
        tm.that(meta.class_stem, eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM)

    def test_read_project_metadata_extracts_optional_url_and_requires_python(
        self, tmp_path: Path
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            requires-python = ">=3.13"
            urls = {{Homepage = "https://example.com"}}
            """,
        )
        meta = u.read_project_metadata(root).unwrap()
        tm.that(meta.project.requires_python, eq=">=3.13")
        tm.that(meta.project.urls.homepage, eq="https://example.com")

    def test_read_project_metadata_defaults_optional_fields_when_absent(
        self, tmp_path: Path
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            """,
        )
        meta = u.read_project_metadata(root).unwrap()
        tm.that(meta.project.requires_python, eq="")
        tm.that(meta.project.urls.homepage, eq="")
        tm.that(meta.project.authors, eq=())

    def test_project_metadata_is_immutable(self, tmp_path: Path) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            """,
        )
        meta = u.read_project_metadata(root).unwrap()
        with pytest.raises(m.ValidationError):
            meta.package_name = "mutated"

    def test_read_project_metadata_fails_on_missing_pyproject(
        self, tmp_path: Path
    ) -> None:
        result = u.read_project_metadata(tmp_path)
        tm.that(result.failure, eq=True)

    @pytest.mark.parametrize(
        ("body", "match_pattern"),
        [
            ('[project]\nversion="0.12.0"\n', "name"),
            ('[project]\nname="x"\n', "version"),
        ],
        ids=["missing_name", "missing_version"],
    )
    def test_read_project_metadata_fails_on_incomplete_pyproject(
        self, tmp_path: Path, body: str, match_pattern: str
    ) -> None:
        root = write_pyproject(tmp_path, body)
        result = u.read_project_metadata(root)
        tm.that(result.failure, eq=True)
        tm.that(match_pattern in result.error, eq=True)
