"""Tests for constants-quality-gate CLI wiring.

Validates CLI dispatch, argument parsing, and verdict classification
using real service instances with temporary workspaces.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.codegen import __main__ as codegen_main
from flext_infra.codegen.constants_quality_gate import (
    FlextInfraCodegenConstantsQualityGate,
)
from flext_tests import tm


class TestConstantsQualityGateCLIDispatch:
    """CLI dispatch and argument parsing for constants-quality-gate."""

    def test_dispatch_returns_int(self, tmp_path: Path) -> None:
        """main() dispatches constants-quality-gate command to handler."""
        argv = ["constants-quality-gate", "--workspace", str(tmp_path)]
        result = codegen_main.main(argv)
        tm.that(result, is_=int)

    def test_parses_before_report_flag(self, tmp_path: Path) -> None:
        """main() parses baseline comparison flags for quality gate."""
        baseline = tmp_path / "before.json"
        argv = [
            "constants-quality-gate",
            "--workspace",
            str(tmp_path),
            "--before-report",
            str(baseline),
            "--format",
            "json",
        ]
        result = codegen_main.main(argv)
        tm.that(result, is_=int)

    def test_json_format_exits_with_int(self, tmp_path: Path) -> None:
        """JSON mode returns an integer exit code."""
        argv = [
            "constants-quality-gate",
            "--workspace",
            str(tmp_path),
            "--format",
            "json",
        ]
        result = codegen_main.main(argv)
        tm.that(result, is_=int)

    def test_text_format_exits_with_int(self, tmp_path: Path) -> None:
        """Text mode returns an integer exit code."""
        argv = [
            "constants-quality-gate",
            "--workspace",
            str(tmp_path),
            "--format",
            "text",
        ]
        result = codegen_main.main(argv)
        tm.that(result, is_=int)


class TestConstantsQualityGateVerdict:
    """Verdict classification and real workspace execution."""

    def test_success_verdict_accepts_pass(self) -> None:
        """is_success_verdict returns True for PASS."""
        tm.that(
            FlextInfraCodegenConstantsQualityGate.is_success_verdict("PASS"),
            eq=True,
        )

    def test_success_verdict_accepts_conditional_pass(self) -> None:
        """is_success_verdict returns True for CONDITIONAL_PASS."""
        tm.that(
            FlextInfraCodegenConstantsQualityGate.is_success_verdict(
                "CONDITIONAL_PASS"
            ),
            eq=True,
        )

    def test_success_verdict_rejects_fail(self) -> None:
        """is_success_verdict returns False for FAIL."""
        tm.that(
            FlextInfraCodegenConstantsQualityGate.is_success_verdict("FAIL"),
            eq=False,
        )

    def test_real_workspace_run_returns_report(self, tmp_path: Path) -> None:
        """Quality gate runs on real empty workspace without errors."""
        gate = FlextInfraCodegenConstantsQualityGate(workspace_root=tmp_path)
        report = gate.run()
        assert isinstance(report, dict)
        assert "verdict" in report


__all__: list[str] = []
