"""Tests for documentation CLI — main() entry point routing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from flext_core import r
from flext_infra.docs.__main__ import main
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_infra.docs.validator import FlextInfraDocValidator
from flext_tests import tm
from tests.infra.models import m
from tests.infra.typings import t


def _ok_empty(*a: t.Scalar, **kw: t.Scalar) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
    return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])


def _ok_audit(*a: t.Scalar, **kw: t.Scalar) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
    return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])


class TestMainRouting:
    def test_main_with_audit_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--root", "."])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _ok_audit)
        tm.that(main(), eq=0)

    def test_main_with_fix_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "fix", "--root", "."])
        monkeypatch.setattr(FlextInfraDocFixer, "fix", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_build_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "build", "--root", "."])
        monkeypatch.setattr(FlextInfraDocBuilder, "build", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_generate_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "generate", "--root", "."])
        monkeypatch.setattr(FlextInfraDocGenerator, "generate", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_validate_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--root", "."])
        monkeypatch.setattr(FlextInfraDocValidator, "validate", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_no_command_prints_help(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["prog"])
        tm.that(main(), eq=1)

    def test_main_with_help_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_audit_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_fix_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "fix", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_build_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "build", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_generate_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "generate", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_validate_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)


def _capture_audit(
    store: dict[str, t.Scalar],
) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(*a: t.Scalar, **kw: t.Scalar) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        store.update(kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

    return _fn


def _capture_simple(
    store: dict[str, t.Scalar],
) -> Callable[..., r[list[m.Infra.Docs.DocsPhaseReport]]]:
    def _fn(*a: t.Scalar, **kw: t.Scalar) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        store.update(kw)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

    return _fn


class TestMainWithFlags:
    def test_audit_custom_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--root", "/custom/path"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capture_audit(kw))
        main()
        tm.that(str(kw.get("root", "")).endswith("custom/path"), eq=True)

    def test_audit_project_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--project", "test-proj"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capture_audit(kw))
        main()
        tm.that(kw.get("project"), eq="test-proj")

    def test_audit_strict_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--strict", "0"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capture_audit(kw))
        main()
        tm.that(kw.get("strict"), eq=False)

    def test_fix_apply_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "fix", "--apply"])
        monkeypatch.setattr(FlextInfraDocFixer, "fix", _capture_simple(kw))
        main()
        tm.that(kw.get("apply"), eq=True)

    def test_generate_apply_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "generate", "--apply"])
        monkeypatch.setattr(FlextInfraDocGenerator, "generate", _capture_simple(kw))
        main()
        tm.that(kw.get("apply"), eq=True)

    def test_validate_apply_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--apply"])
        monkeypatch.setattr(FlextInfraDocValidator, "validate", _capture_simple(kw))
        main()
        tm.that(kw.get("apply"), eq=True)

    def test_audit_check_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--check", "links"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _capture_audit(kw))
        main()
        tm.that(kw.get("check"), eq="links")

    def test_validate_check_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        kw: dict[str, t.Scalar] = {}
        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--check", "links"])
        monkeypatch.setattr(FlextInfraDocValidator, "validate", _capture_simple(kw))
        main()
        tm.that(kw.get("check"), eq="links")
