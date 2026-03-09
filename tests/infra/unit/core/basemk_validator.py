"""Tests for FlextInfraBaseMkValidator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import m
from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
from flext_tests import tm


class TestBaseMkValidatorCore:
    """Core validation tests for FlextInfraBaseMkValidator."""

    def test_validate_missing_root_basemk(self, tmp_path: Path) -> None:
        """Missing root base.mk returns not-passed report."""
        validator = FlextInfraBaseMkValidator()
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=False)
        tm.that(report.summary, contains="missing root base.mk")

    def test_validate_matching_basemk(self, tmp_path: Path) -> None:
        """Matching base.mk files return success report."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# root content")
        proj = tmp_path / "project1"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")
        (proj / "base.mk").write_text("# root content")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(isinstance(report, m.Infra.Core.ValidationReport), eq=True)

    def test_validate_mismatched_basemk(self, tmp_path: Path) -> None:
        """Mismatched base.mk files produce violations."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# root content")
        proj = tmp_path / "project1"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")
        (proj / "base.mk").write_text("# different content")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=False)

    def test_validate_multiple_projects(self, tmp_path: Path) -> None:
        """Validate checks all projects in workspace."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# root content")
        for i in range(3):
            proj = tmp_path / f"project{i}"
            proj.mkdir()
            (proj / "pyproject.toml").write_text("")
            (proj / "base.mk").write_text("# root content")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=True)
        tm.that(report.summary, contains="3 checked")

    def test_validate_no_vendored_basemk(self, tmp_path: Path) -> None:
        """Projects without base.mk are skipped."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# root content")
        proj = tmp_path / "project1"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=True)

    def test_validate_empty_workspace(self, tmp_path: Path) -> None:
        """Empty workspace with root base.mk passes."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# content")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=True)


class TestBaseMkValidatorEdgeCases:
    """Edge case tests for FlextInfraBaseMkValidator."""

    def test_validate_skips_projects_without_pyproject(self, tmp_path: Path) -> None:
        """Projects without pyproject.toml are skipped."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# content")
        proj = tmp_path / "project1"
        proj.mkdir()
        (proj / "base.mk").write_text("# different")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=True)

    def test_validate_reports_relative_paths(self, tmp_path: Path) -> None:
        """Violations include relative paths."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# root")
        proj = tmp_path / "project1"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")
        (proj / "base.mk").write_text("# different")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=False)
        tm.that(report.violations[0], contains="project1/base.mk")

    def test_validate_reports_all_mismatches(self, tmp_path: Path) -> None:
        """All mismatched files are reported."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# root content")
        for i in range(2):
            proj = tmp_path / f"project{i}"
            proj.mkdir()
            (proj / "pyproject.toml").write_text("")
            (proj / "base.mk").write_text(f"# different content {i}")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=False)
        tm.that(report.violations, length=2)

    def test_validate_passes_when_all_match(self, tmp_path: Path) -> None:
        """All matching files produce pass summary."""
        validator = FlextInfraBaseMkValidator()
        content = "# shared base.mk content"
        (tmp_path / "base.mk").write_text(content)
        for i in range(5):
            proj = tmp_path / f"project{i}"
            proj.mkdir()
            (proj / "pyproject.toml").write_text("")
            (proj / "base.mk").write_text(content)
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=True)
        tm.that(report.summary, contains="all vendored base.mk copies in sync")

    def test_validate_fails_when_any_mismatch(self, tmp_path: Path) -> None:
        """One mismatch among many causes failure."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# root")
        proj1 = tmp_path / "project1"
        proj1.mkdir()
        (proj1 / "pyproject.toml").write_text("")
        (proj1 / "base.mk").write_text("# root")
        proj2 = tmp_path / "project2"
        proj2.mkdir()
        (proj2 / "pyproject.toml").write_text("")
        (proj2 / "base.mk").write_text("# different")
        report = tm.ok(validator.validate(tmp_path))
        tm.that(report.passed, eq=False)

    def test_validate_oserror_returns_failure(self, tmp_path: Path) -> None:
        """OSError on unreadable file returns failure."""
        validator = FlextInfraBaseMkValidator()
        (tmp_path / "base.mk").write_text("# content")
        proj = tmp_path / "project1"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")
        basemk = proj / "base.mk"
        basemk.write_text("# content")
        basemk.chmod(0)
        try:
            result = validator.validate(tmp_path)
            tm.that(result.is_failure, eq=True)
        finally:
            basemk.chmod(0o644)


class TestBaseMkValidatorSha256:
    """Tests for _sha256 hash computation."""

    def test_sha256_computes_file_hash(self, tmp_path: Path) -> None:
        """Hash is a 64-char hex string."""
        f = tmp_path / "test.txt"
        f.write_text("content")
        h = FlextInfraBaseMkValidator._sha256(f)
        tm.that(isinstance(h, str), eq=True)
        tm.that(h, length=64)

    def test_sha256_same_content_same_hash(self, tmp_path: Path) -> None:
        """Identical content produces identical hash."""
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("same content")
        f2.write_text("same content")
        tm.that(
            FlextInfraBaseMkValidator._sha256(f1),
            eq=FlextInfraBaseMkValidator._sha256(f2),
        )

    def test_sha256_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different content produces different hash."""
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("content1")
        f2.write_text("content2")
        tm.that(
            FlextInfraBaseMkValidator._sha256(f1)
            != FlextInfraBaseMkValidator._sha256(f2),
            eq=True,
        )


__all__: list[str] = []
