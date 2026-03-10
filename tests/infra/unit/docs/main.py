"""Tests for documentation CLI — _run_audit and _run_fix handlers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.docs import __main__ as docs_main
from flext_infra.docs.__main__ import _run_audit, _run_fix
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_tests import tm


def _audit_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        "check": "all",
        "strict": 1,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fix_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        "apply": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRunAudit:
    def test_success_no_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="root",
            items=[],
            checks=["links"],
            strict=True,
            passed=True,
        )

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([report])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        tm.that(_run_audit(_audit_args()), eq=0)

    def test_run_audit_success_with_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_audit returns 1 when audit has failures."""
        report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="root",
            items=[],
            checks=["links"],
            strict=True,
            passed=False,
        )

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([report])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        tm.that(_run_audit(_audit_args()), eq=1)

    def test_run_audit_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_audit returns 1 on audit failure."""

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].fail("audit error")

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        monkeypatch.setattr(
            "flext_infra.docs.__main__.output",
            type("O", (), {"error": staticmethod(lambda *a: None)})(),
        )
        tm.that(_run_audit(_audit_args()), eq=1)

    def test_run_audit_with_project_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_audit passes project filter to auditor."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        _run_audit(_audit_args(project="test-project"))
        tm.that(captured_kwargs.get("project"), eq="test-project")

    def test_run_audit_with_projects_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_audit passes projects filter to auditor."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        _run_audit(_audit_args(projects="proj1,proj2"))
        tm.that(captured_kwargs.get("projects"), eq="proj1,proj2")

    def test_run_audit_with_check_links(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_audit passes check parameter."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        _run_audit(_audit_args(check="links"))
        tm.that(captured_kwargs.get("check"), eq="links")

    def test_run_audit_strict_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_audit passes strict parameter."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        _run_audit(_audit_args(strict=0))
        tm.that(captured_kwargs.get("strict"), eq=False)


class TestRunFix:
    """Tests for _run_fix handler."""

    def test_run_fix_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_fix returns 0 on success."""

        def mock_fix(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].ok([])

        monkeypatch.setattr(FlextInfraDocFixer, "fix", mock_fix)
        tm.that(_run_fix(_fix_args()), eq=0)

    def test_run_fix_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_fix returns 1 on failure."""

        def mock_fix(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].fail("fix error")

        monkeypatch.setattr(FlextInfraDocFixer, "fix", mock_fix)
        monkeypatch.setattr(
            docs_main,
            "output",
            type("O", (), {"error": staticmethod(lambda *a: None)})(),
        )
        tm.that(_run_fix(_fix_args()), eq=1)

    def test_run_fix_with_apply_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_fix passes apply flag."""
        captured_kwargs: dict[str, object] = {}

        def mock_fix(*a: object, **kw: object) -> r[list[object]]:
            captured_kwargs.update(kw)
            return r[list[object]].ok([])

        monkeypatch.setattr(FlextInfraDocFixer, "fix", mock_fix)
        _run_fix(_fix_args(apply=True))
        tm.that(captured_kwargs.get("apply"), eq=True)
