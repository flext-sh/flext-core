"""Tests for census models, excluded projects, and violation pattern.

Validates CensusViolation/CensusReport models, project exclusion set,
and the compiled violation regex pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.constants import c
from flext_infra.models import FlextInfraModels
from flext_tests import tm


class TestExcludedProjects:
    def test_flexcore_in_excluded_set(self) -> None:
        tm.that("flexcore" in c.Infra.Codegen.EXCLUDED_PROJECTS, eq=True)

    def test_excluded_set_is_frozenset(self) -> None:
        tm.that(type(c.Infra.Codegen.EXCLUDED_PROJECTS).__name__, eq="frozenset")


class TestViolationPattern:
    def test_named_groups_present(self) -> None:
        match = c.Infra.Codegen.VIOLATION_PATTERN.match(
            "[NS-001-001] src/file.py:10 — msg"
        )
        tm.that(match is not None, eq=True)
        assert match is not None
        assert set(match.groupdict().keys()) == {"rule", "module", "line", "message"}


_CV = FlextInfraModels.Infra.Codegen.CensusViolation
_CR = FlextInfraModels.Infra.Codegen.CensusReport


class TestCensusViolationModel:
    def test_model_fields(self) -> None:
        v = _CV(
            module="src/file.py",
            rule="NS-001",
            line=10,
            message="Test message",
            fixable=True,
        )
        tm.that(v.module, eq="src/file.py")
        tm.that(v.rule, eq="NS-001")
        tm.that(v.line, eq=10)
        tm.that(v.message, eq="Test message")
        tm.that(v.fixable, eq=True)


class TestCensusReportModel:
    def test_empty_report(self) -> None:
        report = _CR(project="test-project", violations=[], total=0, fixable=0)
        tm.that(report.project, eq="test-project")
        tm.that(report.total, eq=0)
        tm.that(report.fixable, eq=0)
        tm.that(report.violations, eq=[])

    def test_report_with_mixed_violations(self) -> None:
        violations = [
            _CV(module="src/a.py", rule="NS-000", line=1, message="m1", fixable=False),
            _CV(module="src/b.py", rule="NS-001", line=2, message="m2", fixable=True),
            _CV(module="src/c.py", rule="NS-002", line=3, message="m3", fixable=True),
        ]
        report = _CR(
            project="test-project",
            violations=violations,
            total=len(violations),
            fixable=sum(1 for v in violations if v.fixable),
        )
        tm.that(report.total, eq=3)
        tm.that(report.fixable, eq=2)


__all__: list[str] = []
