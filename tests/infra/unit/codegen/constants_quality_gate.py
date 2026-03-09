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


def test_main_constants_quality_gate_dispatch(tmp_path: Path) -> None:
    """main() dispatches constants-quality-gate command to handler."""
    argv = ["constants-quality-gate", "--workspace", str(tmp_path)]
    result = codegen_main.main(argv)
    tm.that(isinstance(result, int), eq=True)


def test_main_constants_quality_gate_parses_before_report(tmp_path: Path) -> None:
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
    tm.that(isinstance(result, int), eq=True)


def test_handle_constants_quality_gate_json_exits_with_int(
    tmp_path: Path,
) -> None:
    """JSON mode returns an integer exit code."""
    argv = ["constants-quality-gate", "--workspace", str(tmp_path), "--format", "json"]
    result = codegen_main.main(argv)
    tm.that(isinstance(result, int), eq=True)


def test_handle_constants_quality_gate_text_exits_with_int(
    tmp_path: Path,
) -> None:
    """Text mode returns an integer exit code."""
    argv = ["constants-quality-gate", "--workspace", str(tmp_path), "--format", "text"]
    result = codegen_main.main(argv)
    tm.that(isinstance(result, int), eq=True)


def test_quality_gate_success_verdict_helper() -> None:
    """Success helper accepts PASS and CONDITIONAL_PASS only."""
    tm.that(
        FlextInfraCodegenConstantsQualityGate.is_success_verdict("PASS"),
        eq=True,
    )
    tm.that(
        FlextInfraCodegenConstantsQualityGate.is_success_verdict("CONDITIONAL_PASS"),
        eq=True,
    )
    tm.that(
        FlextInfraCodegenConstantsQualityGate.is_success_verdict("FAIL"),
        eq=False,
    )


def test_quality_gate_real_workspace_run(tmp_path: Path) -> None:
    """Quality gate runs on real empty workspace without errors."""
    gate = FlextInfraCodegenConstantsQualityGate(workspace_root=tmp_path)
    report = gate.run()
    tm.that(isinstance(report, dict), eq=True)
    tm.that("verdict" in report, eq=True)


__all__: list[str] = []
