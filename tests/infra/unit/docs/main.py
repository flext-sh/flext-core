"""Tests for documentation CLI — _run_audit and _run_fix handlers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest

from flext_core import r, t
from flext_infra.docs import __main__ as docs_main
from flext_infra.docs.__main__ import _run_audit, _run_fix
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_tests import tm
from tests.infra.helpers import h
from tests.infra.models import m


def _audit_args(**overrides: t.Scalar | None) -> argparse.Namespace:
    defaults: dict[str, t.Scalar | None] = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        "check": "all",
        "strict": 1,
    }
    defaults.update(overrides)
    return h.ns(**defaults)


def _fix_args(**overrides: t.Scalar | None) -> argparse.Namespace:
    defaults: dict[str, t.Scalar | None] = {
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
        _self: t.Scalar,
        *_a: t.Scalar,
        **_kw: t.Scalar,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a, _kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok(val)

    return _fn


def _fail_report(err: str) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(
        _self: t.Scalar,
        *_a: t.Scalar,
        **_kw: t.Scalar,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a, _kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].fail(err)

    return _fn


def _ok_list(
    val: list[m.Infra.Docs.DocsPhaseReport],
) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(
        _self: t.Scalar,
        *_a: t.Scalar,
        **_kw: t.Scalar,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a, _kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok(val)

    return _fn


def _fail_list(err: str) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(
        _self: t.Scalar,
        *_a: t.Scalar,
        **_kw: t.Scalar,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a, _kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].fail(err)

    return _fn


_SILENT = type("O", (), {"error": staticmethod(lambda *a: None)})()


def _capturing(
    captured: dict[str, t.Scalar],
) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(
        _self: t.Scalar,
        *_a: t.Scalar,
        **kw: t.Scalar,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        _ = (_self, _a)
        captured.update(kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

    return _fn


class TestRunAudit:
    @pytest.mark.parametrize(
        ("passed", "expected"),
        [(True, 0), (False, 1)],
    )
    def test_run_audit_report_exit_code(
        self,
        monkeypatch: pytest.MonkeyPatch,
        passed: bool,
        expected: int,
    ) -> None:
        report = m.Infra.Docs.DocsPhaseReport(
            phase="audit",
            scope="root",
            items=[],
            checks=["links"],
            strict=True,
            passed=passed,
        )
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _ok([report]))
        tm.that(_run_audit(_audit_args()), eq=expected)

    def test_run_audit_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _fail_report("audit error"))
        monkeypatch.setattr(docs_main, "output", _SILENT)
        tm.that(_run_audit(_audit_args()), eq=1)

    @pytest.mark.parametrize(
        ("kwargs", "field", "expected"),
        [
            ({"project": "test-project"}, "project", "test-project"),
            ({"projects": "proj1,proj2"}, "projects", "proj1,proj2"),
            ({"check": "links"}, "check", "links"),
            ({"strict": 0}, "strict", False),
        ],
    )
    def test_run_audit_forwards_arguments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict[str, t.Scalar],
        field: str,
        expected: t.Scalar,
    ) -> None:
        captured_kwargs: dict[str, t.Scalar] = {}
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capturing(captured_kwargs))
        _run_audit(_audit_args(**kwargs))
        tm.that(captured_kwargs.get(field), eq=expected)


class TestRunFix:
    def test_run_fix_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocFixer, "fix", _ok_list([]))
        tm.that(_run_fix(_fix_args()), eq=0)

    def test_run_fix_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocFixer, "fix", _fail_list("fix error"))
        monkeypatch.setattr(docs_main, "output", _SILENT)
        tm.that(_run_fix(_fix_args()), eq=1)

    @pytest.mark.parametrize(("apply", "expected"), [(True, True), (False, False)])
    def test_run_fix_forwards_apply_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        apply: bool,
        expected: bool,
    ) -> None:
        captured_kwargs: dict[str, t.Scalar] = {}

        def mock_fix(
            _self: t.Scalar,
            *_a: t.Scalar,
            **kw: t.Scalar,
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            _ = (_self, _a)
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(FlextInfraDocFixer, "fix", mock_fix)
        _run_fix(_fix_args(apply=apply))
        assert captured_kwargs.get("apply") == expected
