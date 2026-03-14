"""Tests for FlextInfraDocAuditor — CLI main() and scope failure paths.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flext_core import r, t
from flext_infra.docs.auditor import FlextInfraDocAuditor, main
from flext_infra.docs.shared import FlextInfraDocsShared
from flext_tests import tm
from tests.infra.models import m


class TestAuditorScopeFailure:
    """Tests for audit scope build failure path."""

    def test_audit_with_scope_build_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test audit() when scope building fails."""
        auditor = FlextInfraDocAuditor()

        def mock_build_scopes(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.FlextInfraDocScope]]:
            return r[list[m.Infra.Docs.FlextInfraDocScope]].fail("scope build error")

        monkeypatch.setattr(FlextInfraDocsShared, "build_scopes", mock_build_scopes)
        result = auditor.audit(tmp_path)
        tm.fail(result, has="scope build error")


class TestAuditorMainCli:
    """Tests for auditor main() CLI entry point."""

    def test_main_with_failure_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main() CLI entry point with audit failure."""

        def mock_audit(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].fail("audit error")

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        monkeypatch.setattr(sys, "argv", ["auditor", "--root", str(tmp_path)])
        result = main()
        tm.that(result, eq=1)

    def test_main_with_failed_reports(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main() returns 1 when reports have failures."""
        failed_report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="test",
            items=[],
            checks=["links"],
            strict=True,
            passed=False,
        )

        def mock_audit(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([failed_report])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        monkeypatch.setattr(sys, "argv", ["auditor", "--root", str(tmp_path)])
        result = main()
        tm.that(result, eq=1)

    def test_main_with_success_reports(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main() returns 0 when all reports pass."""
        passed_report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="test",
            items=[],
            checks=["links"],
            strict=True,
            passed=True,
        )

        def mock_audit(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([passed_report])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        monkeypatch.setattr(sys, "argv", ["auditor", "--root", str(tmp_path)])
        result = main()
        tm.that(result, eq=0)

    def test_main_with_all_cli_arguments(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main() CLI with all arguments."""
        passed_report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="test",
            items=[],
            checks=["links"],
            strict=True,
            passed=True,
        )

        def mock_audit(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([passed_report])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "auditor",
                "--root",
                str(tmp_path),
                "--project",
                "test-proj",
                "--output-dir",
                str(tmp_path / "output"),
                "--check",
                "links",
                "--strict",
                "0",
            ],
        )
        result = main()
        tm.that(result, eq=0)

    def test_main_entry_point_returns_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test main() returns 0 on success."""
        passed_report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="test",
            items=[],
            checks=["links"],
            strict=True,
            passed=True,
        )

        def mock_audit(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([passed_report])

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        monkeypatch.setattr(sys, "argv", ["auditor", "--root", str(tmp_path)])
        result = main()
        tm.that(result, eq=0)
