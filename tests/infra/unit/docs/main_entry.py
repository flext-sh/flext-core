"""Tests for documentation CLI — main() entry point routing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.docs.__main__ import main
from flext_infra.docs.auditor import FlextInfraDocAuditor
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.fixer import FlextInfraDocFixer
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_infra.docs.validator import FlextInfraDocValidator
from flext_tests import tm


def _ok_empty(*a: object, **kw: object) -> r[list[object]]:
    return r[list[object]].ok([])


def _ok_audit(*a: object, **kw: object) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
    return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])


class TestMainRouting:
    """Tests for main() command routing."""

    def test_main_with_audit_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() routes audit command."""
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--root", "."])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", _ok_audit)
        tm.that(main(), eq=0)

    def test_main_with_fix_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() routes fix command."""
        monkeypatch.setattr(sys, "argv", ["prog", "fix", "--root", "."])
        monkeypatch.setattr(FlextInfraDocFixer, "fix", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_build_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() routes build command."""
        monkeypatch.setattr(sys, "argv", ["prog", "build", "--root", "."])
        monkeypatch.setattr(FlextInfraDocBuilder, "build", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_generate_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() routes generate command."""
        monkeypatch.setattr(sys, "argv", ["prog", "generate", "--root", "."])
        monkeypatch.setattr(FlextInfraDocGenerator, "generate", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_validate_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() routes validate command."""
        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--root", "."])
        monkeypatch.setattr(FlextInfraDocValidator, "validate", _ok_empty)
        tm.that(main(), eq=0)

    def test_main_with_no_command_prints_help(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main() prints help when no command given."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        tm.that(main(), eq=1)

    def test_main_with_help_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() with --help flag."""
        monkeypatch.setattr(sys, "argv", ["prog", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_audit_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() with audit --help."""
        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_fix_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() with fix --help."""
        monkeypatch.setattr(sys, "argv", ["prog", "fix", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_build_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() with build --help."""
        monkeypatch.setattr(sys, "argv", ["prog", "build", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_generate_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() with generate --help."""
        monkeypatch.setattr(sys, "argv", ["prog", "generate", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)

    def test_main_with_validate_help(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() with validate --help."""
        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        tm.that(exc_info.value.code, eq=0)


class TestMainWithFlags:
    """Tests for main() with specific CLI flags."""

    def test_main_audit_with_custom_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() audit with custom root path."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--root", "/custom/path"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        main()
        tm.that(str(captured_kwargs.get("root", "")).endswith("custom/path"), eq=True)

    def test_main_audit_with_project_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main() audit with project filter."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--project", "test-proj"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        main()
        tm.that(captured_kwargs.get("project"), eq="test-proj")

    def test_main_audit_with_strict_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() audit with strict flag."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--strict", "0"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        main()
        tm.that(captured_kwargs.get("strict"), eq=False)

    def test_main_fix_with_apply_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main() fix with apply flag."""
        captured_kwargs: dict[str, object] = {}

        def mock_fix(*a: object, **kw: object) -> r[list[object]]:
            captured_kwargs.update(kw)
            return r[list[object]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "fix", "--apply"])
        monkeypatch.setattr(FlextInfraDocFixer, "fix", mock_fix)
        main()
        tm.that(captured_kwargs.get("apply"), eq=True)

    def test_main_generate_with_apply_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main() generate with apply flag."""
        captured_kwargs: dict[str, object] = {}

        def mock_gen(*a: object, **kw: object) -> r[list[object]]:
            captured_kwargs.update(kw)
            return r[list[object]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "generate", "--apply"])
        monkeypatch.setattr(FlextInfraDocGenerator, "generate", mock_gen)
        main()
        tm.that(captured_kwargs.get("apply"), eq=True)

    def test_main_validate_with_apply_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main() validate with apply flag."""
        captured_kwargs: dict[str, object] = {}

        def mock_val(*a: object, **kw: object) -> r[list[object]]:
            captured_kwargs.update(kw)
            return r[list[object]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--apply"])
        monkeypatch.setattr(FlextInfraDocValidator, "validate", mock_val)
        main()
        tm.that(captured_kwargs.get("apply"), eq=True)

    def test_main_audit_with_check_parameter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main() audit with check parameter."""
        captured_kwargs: dict[str, object] = {}

        def mock_audit(
            *a: object, **kw: object
        ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
            captured_kwargs.update(kw)
            return r[list[m.Infra.Docs.DocsPhaseReport]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "audit", "--check", "links"])
        monkeypatch.setattr(FlextInfraDocAuditor, "audit", mock_audit)
        main()
        tm.that(captured_kwargs.get("check"), eq="links")

    def test_main_validate_with_check_parameter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main() validate with check parameter."""
        captured_kwargs: dict[str, object] = {}

        def mock_val(*a: object, **kw: object) -> r[list[object]]:
            captured_kwargs.update(kw)
            return r[list[object]].ok([])

        monkeypatch.setattr(sys, "argv", ["prog", "validate", "--check", "links"])
        monkeypatch.setattr(FlextInfraDocValidator, "validate", mock_val)
        main()
        tm.that(captured_kwargs.get("check"), eq="links")
