"""Tests for FlextInfraBaseMkValidator.

Exemplar test file: uses tf (file ops), tm (matchers), fixtures, parametrize.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
from flext_tests import tf, tm
from tests.infra.models import m

_ROOT = "# root content"


def _workspace(
    base: Path,
    root: str,
    projects: Mapping[str, str | None],
) -> Path:
    """Build workspace: root base.mk + project dirs with optional vendored copy.

    Uses tf.create_in() for all file I/O. Projects with None content skip base.mk.
    """
    tf.create_in(root, "base.mk", base)
    for name, content in projects.items():
        proj = base / name
        proj.mkdir()
        tf.create_in("", "pyproject.toml", proj)
        if content is not None:
            tf.create_in(content, "base.mk", proj)
    return base


@pytest.fixture
def v() -> FlextInfraBaseMkValidator:
    """Shared validator instance — eliminates 15× construction."""
    return FlextInfraBaseMkValidator()


class TestBaseMkValidatorCore:
    """Core validation: pass/fail scenarios via parametrize."""

    def test_missing_root_basemk_fails(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
    ) -> None:
        report = tm.ok(v.validate(tmp_path))
        tm.that(report.passed, eq=False)
        tm.that(report.summary, has="missing root base.mk")

    def test_matching_basemk_returns_report_model(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
    ) -> None:
        ws = _workspace(tmp_path, _ROOT, {"project1": _ROOT})
        report = tm.ok(v.validate(ws))
        assert isinstance(report, m.Infra.Core.ValidationReport)

    @pytest.mark.parametrize(
        ("projects", "expect_pass", "summary_has"),
        [
            ({"p1": _ROOT, "p2": _ROOT, "p3": _ROOT}, True, "3 checked"),
            ({"p1": "# different"}, False, None),
            ({}, True, None),
        ],
        ids=["all-match-3", "one-mismatch", "empty-workspace"],
    )
    def test_validate_projects(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
        projects: dict[str, str | None],
        expect_pass: bool,
        summary_has: str | None,
    ) -> None:
        ws = _workspace(tmp_path, _ROOT, projects)
        report = tm.ok(v.validate(ws))
        tm.that(report.passed, eq=expect_pass)
        if summary_has:
            tm.that(report.summary, has=summary_has)

    def test_projects_without_basemk_skipped(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
    ) -> None:
        ws = _workspace(tmp_path, _ROOT, {"project1": None})
        report = tm.ok(v.validate(ws))
        tm.that(report.passed, eq=True)


class TestBaseMkValidatorEdgeCases:
    """Edge cases: skip rules, violation reporting, OS errors."""

    def test_skips_projects_without_pyproject(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
    ) -> None:
        tf.create_in("# content", "base.mk", tmp_path)
        proj = tmp_path / "project1"
        proj.mkdir()
        tf.create_in("# different", "base.mk", proj)
        report = tm.ok(v.validate(tmp_path))
        tm.that(report.passed, eq=True)

    def test_violations_include_relative_paths(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
    ) -> None:
        ws = _workspace(tmp_path, "# root", {"project1": "# different"})
        report = tm.ok(v.validate(ws))
        tm.that(report.passed, eq=False)
        tm.that(report.violations[0], has="project1/base.mk")

    def test_reports_all_mismatches(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
    ) -> None:
        projects = {f"p{i}": f"# different {i}" for i in range(2)}
        ws = _workspace(tmp_path, _ROOT, projects)
        report = tm.ok(v.validate(ws))
        tm.that(report.passed, eq=False)
        tm.that(report.violations, length=2)

    @pytest.mark.parametrize(
        ("match_content", "expect_pass", "summary_has"),
        [
            ("# shared", True, "all vendored base.mk copies in sync"),
            (None, False, None),
        ],
        ids=["all-5-match", "one-mismatch-among-many"],
    )
    def test_multi_project_scenarios(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
        match_content: str | None,
        expect_pass: bool,
        summary_has: str | None,
    ) -> None:
        root = match_content or "# root"
        projects: dict[str, str | None] = {}
        for i in range(5 if match_content else 2):
            projects[f"p{i}"] = match_content or ("# root" if i == 0 else "# diff")
        ws = _workspace(tmp_path, root, projects)
        report = tm.ok(v.validate(ws))
        tm.that(report.passed, eq=expect_pass)
        if summary_has:
            tm.that(report.summary, has=summary_has)

    def test_oserror_returns_failure(
        self,
        tmp_path: Path,
        v: FlextInfraBaseMkValidator,
    ) -> None:
        ws = _workspace(tmp_path, "# content", {"project1": "# content"})
        basemk = ws / "project1" / "base.mk"
        basemk.chmod(0)
        try:
            result = v.validate(ws)
            tm.that(result.is_failure, eq=True)
        finally:
            basemk.chmod(0o644)


class TestBaseMkValidatorSha256:
    """Hash computation: determinism, length, collision resistance."""

    @staticmethod
    def _sha(path: Path) -> str:
        return FlextInfraBaseMkValidator._sha256(path)

    def test_hash_is_64char_hex(self, tmp_path: Path) -> None:
        f = tf.create_in("content", "test.txt", tmp_path)
        h = self._sha(f)
        assert isinstance(h, str)
        tm.that(h, length=64)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tf.create_in("same", "f1.txt", tmp_path)
        f2 = tf.create_in("same", "f2.txt", tmp_path)
        tm.that(self._sha(f1), eq=self._sha(f2))

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tf.create_in("content1", "f1.txt", tmp_path)
        f2 = tf.create_in("content2", "f2.txt", tmp_path)
        tm.that(self._sha(f1), ne=self._sha(f2))


__all__: list[str] = []
