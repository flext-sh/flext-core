"""Project metadata read utility tests.

``_read(root)`` returns ``p.Result[m.ProjectMetadata]`` — a
Result-wrapped, frozen model whose PEP 621 payload lives under the nested
``project`` field. Tests assert the observable success value and the
Result failure contract for missing/incomplete pyproject inputs.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

import pytest
from flext_tests import tm

from flext_core import r
from tests import c, m, p, u
from tests.unit._project_metadata_support import write_pyproject


def _read(root: Path) -> p.ResultView[m.ProjectMetadata]:
    """Read project metadata through the canonical owner chain.

    Returns:
        The resulting ``p.ResultView[m.ProjectMetadata]``.

    """
    resolved = root.resolve()
    try:
        document = u.read_project_document_cached(resolved)
        meta = u.build_project_metadata(resolved, document)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        return cast(
            "p.ResultView[m.ProjectMetadata]",
            r[m.ProjectMetadata].fail(
                f"cannot read project metadata from {resolved}: {exc}",
                exception=exc,
            ),
        )
    return cast("p.ResultView[m.ProjectMetadata]", r[m.ProjectMetadata].ok(meta))


class TestsFlextUtilitiesProjectMetadataRead:
    """Tests for ``FlextUtilitiesProjectMetadataRead``."""

    @pytest.mark.parametrize(
        ("project_name", "expected_stem"),
        [
            (c.Tests.SAMPLE_PROJECT_NAME, c.Tests.SAMPLE_PROJECT_CLASS_STEM),
            ("flext-core", "Flext"),
        ],
    )
    @staticmethod
    def test_derive_class_stem_produces_pascal_case_from_project_name(
        project_name: str,
        expected_stem: str,
    ) -> None:
        """Test derive class stem produces pascal case from project name."""
        tm.that(u.derive_class_stem(project_name), eq=expected_stem)

    @staticmethod
    def test_derive_class_stem_returns_empty_for_empty_input() -> None:
        """Test derive class stem returns empty for empty input."""
        tm.that(u.derive_class_stem(""), eq="")

    @staticmethod
    def test_read_project_metadata_parses_minimal_pyproject(
        tmp_path: Path,
    ) -> None:
        """Test read project metadata parses minimal pyproject."""
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            description = "LDIF"
            """,
        )
        meta = _read(root).value
        tm.that(meta, is_=m.ProjectMetadata)
        tm.that(meta.project.name, eq=c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(meta.class_stem, eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM)

    @staticmethod
    def test_read_project_metadata_extracts_author_names_from_project_table(
        tmp_path: Path,
    ) -> None:
        """Test read project metadata extracts author names from project table."""
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
        meta = _read(root).value
        tm.that(
            tuple(author.name for author in meta.project.authors),
            eq=(c.Tests.SAMPLE_AUTHOR_ALICE, c.Tests.SAMPLE_AUTHOR_BOB),
        )

    @staticmethod
    def test_read_project_metadata_derives_package_name_and_stem_from_name(
        tmp_path: Path,
    ) -> None:
        """Test read project metadata derives package name and stem from name."""
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            """,
        )
        meta = _read(root).value
        tm.that(meta.package_name, eq=c.Tests.SAMPLE_PROJECT_NAME.replace("-", "_"))
        tm.that(meta.class_stem, eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM)

    @staticmethod
    def test_read_project_metadata_extracts_optional_url_and_requires_python(
        tmp_path: Path,
    ) -> None:
        """Test read project metadata extracts optional url and requires python."""
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
        meta = _read(root).value
        tm.that(meta.project.requires_python, eq=">=3.13")
        tm.that(meta.project.urls.homepage, eq="https://example.com")

    @staticmethod
    def test_read_project_metadata_defaults_optional_fields_when_absent(
        tmp_path: Path,
    ) -> None:
        """Test read project metadata defaults optional fields when absent."""
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            """,
        )
        meta = _read(root).value
        tm.that(meta.project.requires_python, eq="")
        tm.that(meta.project.urls.homepage, eq="")
        tm.that(meta.project.authors, eq=())

    @staticmethod
    def test_project_metadata_is_immutable(tmp_path: Path) -> None:
        """Test project metadata is immutable."""
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            """,
        )
        meta = _read(root).value
        tm.rejects_assignment(
            meta,
            "package_name",
            "mutated",
            expected=m.ValidationError,
        )

    @staticmethod
    def test_read_project_metadata_fails_on_missing_pyproject(
        tmp_path: Path,
    ) -> None:
        """Test read project metadata fails on missing pyproject."""
        result = _read(tmp_path)
        tm.that(result.failure, eq=True)

    @pytest.mark.parametrize(
        ("body", "match_pattern"),
        [
            ('[project]\nversion="0.12.0"\n', "name"),
            ('[project]\nname="x"\n', "version"),
        ],
        ids=["missing_name", "missing_version"],
    )
    @staticmethod
    def test_read_project_metadata_fails_on_incomplete_pyproject(
        tmp_path: Path,
        body: str,
        match_pattern: str,
    ) -> None:
        """Test read project metadata fails on incomplete pyproject."""
        root = write_pyproject(tmp_path, body)
        result = _read(root)
        tm.that(result.failure, eq=True)
        tm.that(match_pattern in (result.error or ""), eq=True)
