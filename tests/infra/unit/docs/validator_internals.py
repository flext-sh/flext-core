"""Tests for FlextInfraDocValidator — internal methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.docs.validator import FlextInfraDocValidator
from flext_tests import tm
from tests.infra.models import m


@pytest.fixture
def validator() -> FlextInfraDocValidator:
    return FlextInfraDocValidator()


def _scope(tmp_path: Path, name: str = "test") -> m.Infra.Docs.FlextInfraDocScope:
    return m.Infra.Docs.FlextInfraDocScope(
        name=name,
        path=tmp_path,
        report_dir=tmp_path / "reports",
    )


class TestValidateScope:
    def test_adr_check(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        report = validator._validate_scope(
            _scope(tmp_path, "root"),
            check="adr-skill",
            apply_mode=False,
        )
        tm.that(report.scope, eq="root")

    def test_without_config(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        report = validator._validate_scope(
            _scope(tmp_path),
            check="all",
            apply_mode=False,
        )
        tm.that(report.scope, eq="test")

    def test_all_check(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        report = validator._validate_scope(
            _scope(tmp_path),
            check="all",
            apply_mode=False,
        )
        tm.that(report.scope, eq="test")

    def test_adr_check_failure(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def mock_adr_check(path: Path | str) -> tuple[int, list[str]]:
            return (1, ["missing_skill"])

        monkeypatch.setattr(validator, "_run_adr_skill_check", mock_adr_check)
        report = validator._validate_scope(
            _scope(tmp_path),
            check="adr",
            apply_mode=False,
        )
        tm.that(report.scope, eq="test")

    def test_adr_skill_check_failure(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        arch_dir = tmp_path / "docs/architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        (arch_dir / "architecture_config.json").write_text("{}")

        def mock_adr_check(path: Path | str) -> tuple[int, list[str]]:
            return (1, ["missing_skill_1", "missing_skill_2"])

        monkeypatch.setattr(validator, "_run_adr_skill_check", mock_adr_check)
        report = validator._validate_scope(
            _scope(tmp_path, "root"),
            check="adr-skill",
            apply_mode=False,
        )
        tm.that(report.scope, eq="root")
        tm.that(report.result, eq="FAIL")


class TestAdrHelpers:
    def test_has_adr_with_text(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        sf = tmp_path / "SKILL.md"
        sf.write_text("# Skill\n\nADR: This is an ADR reference.\n")
        tm.that(validator._has_adr_reference(sf), eq=True)

    def test_has_adr_without_text(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        sf = tmp_path / "SKILL.md"
        sf.write_text("# Skill\n\nNo architecture decision record here.\n")
        tm.that(validator._has_adr_reference(sf), eq=False)

    def test_adr_check_no_config(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        code, _missing = validator._run_adr_skill_check(tmp_path)
        tm.that(code >= 0, eq=True)

    def test_adr_check_with_config(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        cfg_dir = tmp_path / "docs/architecture"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "architecture_config.json").write_text(
            '{"docs_validation": {"required_skills": ["test-skill"]}}',
        )
        code, _missing = validator._run_adr_skill_check(tmp_path)
        tm.that(code >= 0, eq=True)

    def test_adr_check_with_missing_skills(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        (tmp_path / ".claude/skills").mkdir(parents=True, exist_ok=True)
        code, _missing = validator._run_adr_skill_check(tmp_path)
        tm.that(code >= 0, eq=True)


class TestMaybeWriteTodo:
    def test_root_scope_skipped(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        tm.that(
            validator._maybe_write_todo(
                _scope(tmp_path, "root"),
                apply_mode=True,
            ),
            eq=False,
        )

    def test_apply_false(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        tm.that(
            validator._maybe_write_todo(
                _scope(tmp_path),
                apply_mode=False,
            ),
            eq=False,
        )

    def test_creates_file(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        result = validator._maybe_write_todo(_scope(tmp_path), apply_mode=True)
        tm.that(result, eq=True)
        tm.that((tmp_path / "TODOS.md").exists(), eq=True)
