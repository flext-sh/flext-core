"""Tests for documentation CLI — _run_audit and _run_fix handlers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest

from flext_core import r
from flext_infra.docs import __main__ as docs_main
from flext_infra.docs.__main__ import _run_audit, _run_fix
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_tests import tm
from tests.infra.helpers import h
from tests.infra.models import m
from tests.infra.typings import t


def _audit_args(**overrides: t.ContainerValue) -> argparse.Namespace:
    defaults: dict[str, t.ContainerValue | None] = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        "check": "all",
        "strict": 1,
    }
    defaults.update(overrides)
    return h.ns(**defaults)


def _fix_args(**overrides: t.ContainerValue) -> argparse.Namespace:
    defaults: dict[str, t.ContainerValue | None] = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        "apply": False,
    }
    defaults.update(overrides)
    return h.ns(**defaults)


def _ok(
    val: list[m.Infra.Docs.DocsPhaseReport],
) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(
        _self: t.ContainerValue,
        *_a: t.ContainerValue,
        **_kw: t.ContainerValue,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a, _kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok(val)

    return _fn


def _fail_report(err: str) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(
        _self: t.ContainerValue,
        *_a: t.ContainerValue,
        **_kw: t.ContainerValue,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a, _kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].fail(err)

    return _fn


def _ok_list(val: list[t.ContainerValue]) -> Callable[..., r[list[t.ContainerValue]]]:
    def _fn(
        _self: t.ContainerValue,
        *_a: t.ContainerValue,
        **_kw: t.ContainerValue,
    ) -> r[list[t.ContainerValue]]:
        _ = (_self, _a, _kw)
        return r[list[t.ContainerValue]].ok(val)

    return _fn


def _fail_list(err: str) -> Callable[..., r[list[t.ContainerValue]]]:
    def _fn(
        _self: t.ContainerValue,
        *_a: t.ContainerValue,
        **_kw: t.ContainerValue,
    ) -> r[list[t.ContainerValue]]:
        _ = (_self, _a, _kw)
        return r[list[t.ContainerValue]].fail(err)

    return _fn


_SILENT = type("O", (), {"error": staticmethod(lambda *a: None)})()


def _capturing(
    captured: dict[str, t.ContainerValue],
) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(
        _self: t.ContainerValue,
        *_a: t.ContainerValue,
        **kw: t.ContainerValue,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a)
        captured.update(kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

    return _fn


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

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _ok([report]))
        tm.that(_run_audit(_audit_args()), eq=0)

    def test_run_audit_success_with_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="root",
            items=[],
            checks=["links"],
            strict=True,
            passed=False,
        )

        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _ok([report]))
        tm.that(_run_audit(_audit_args()), eq=1)

    def test_run_audit_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _fail_report("audit error"))
        monkeypatch.setattr(docs_main, "output", _SILENT)
        tm.that(_run_audit(_audit_args()), eq=1)

    def test_run_audit_with_project_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs: dict[str, t.ContainerValue] = {}
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capturing(captured_kwargs))
        _run_audit(_audit_args(project="test-project"))
        tm.that(captured_kwargs.get("project"), eq="test-project")

    def test_run_audit_with_projects_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs: dict[str, t.ContainerValue] = {}
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capturing(captured_kwargs))
        _run_audit(_audit_args(projects="proj1,proj2"))
        tm.that(captured_kwargs.get("projects"), eq="proj1,proj2")

    def test_run_audit_with_check_links(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: dict[str, t.ContainerValue] = {}
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capturing(captured_kwargs))
        _run_audit(_audit_args(check="links"))
        tm.that(captured_kwargs.get("check"), eq="links")

    def test_run_audit_strict_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: dict[str, t.ContainerValue] = {}
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capturing(captured_kwargs))
        _run_audit(_audit_args(strict=0))
        tm.that(captured_kwargs.get("strict"), eq=False)


class TestRunFix:
    def test_run_fix_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocFixer, "fix", _ok_list([]))
        tm.that(_run_fix(_fix_args()), eq=0)

    def test_run_fix_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocFixer, "fix", _fail_list("fix error"))
        monkeypatch.setattr(docs_main, "output", _SILENT)
        tm.that(_run_fix(_fix_args()), eq=1)

    def test_run_fix_with_apply_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: dict[str, t.ContainerValue] = {}

        def mock_fix(
            _self: t.ContainerValue,
            *_a: t.ContainerValue,
            **kw: t.ContainerValue,
        ) -> r[list[t.ContainerValue]]:
            _ = (_self, _a)
            captured_kwargs.update(kw)
            return r[list[t.ContainerValue]].ok([])

        monkeypatch.setattr(FlextInfraDocFixer, "fix", mock_fix)
        _run_fix(_fix_args(apply=True))
        tm.that(captured_kwargs.get("apply"), eq=True)
