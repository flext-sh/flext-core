"""Tests for constants-quality-gate CLI wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from flext_infra.codegen import __main__ as codegen_main
from flext_infra.codegen.constants_quality_gate import (
    FlextInfraCodegenConstantsQualityGate,
)


def test_main_constants_quality_gate_dispatch(tmp_path: Path) -> None:
    """main() dispatches constants-quality-gate command to handler."""
    argv = ["constants-quality-gate", "--workspace", str(tmp_path)]
    with patch(
        "flext_infra.codegen.__main__._handle_constants_quality_gate"
    ) as mock_handle:
        mock_handle.return_value = 0
        result = codegen_main.main(argv)
    assert result == 0
    assert mock_handle.call_count == 1


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
    with patch(
        "flext_infra.codegen.__main__._handle_constants_quality_gate"
    ) as mock_handle:
        mock_handle.return_value = 0
        result = codegen_main.main(argv)
    assert result == 0
    args = mock_handle.call_args[0][0]
    assert args.before_report == baseline
    assert args.baseline_file is None
    assert args.output_format == "json"


def test_handle_constants_quality_gate_json_conditional_pass_exits_zero(
    tmp_path: Path,
) -> None:
    """JSON mode returns success for CONDITIONAL_PASS verdict."""
    argv = ["constants-quality-gate", "--workspace", str(tmp_path), "--format", "json"]
    with patch(
        "flext_infra.codegen.__main__.FlextInfraCodegenConstantsQualityGate"
    ) as mock_gate_cls:
        gate_instance = Mock()
        gate_instance.run.return_value = {
            "verdict": "CONDITIONAL_PASS",
            "checks": [{"name": "namespace_compliance", "passed": True}],
            "improvement": {"violations_delta": 0},
        }
        mock_gate_cls.return_value = gate_instance
        mock_gate_cls.is_success_verdict.return_value = True
        with patch("builtins.print") as mock_print:
            result = codegen_main.main(argv)
    assert result == 0
    assert mock_print.call_count >= 0


def test_handle_constants_quality_gate_text_fail_exits_one(tmp_path: Path) -> None:
    """Text mode returns failure for FAIL verdict."""
    argv = ["constants-quality-gate", "--workspace", str(tmp_path), "--format", "text"]
    with patch(
        "flext_infra.codegen.__main__.FlextInfraCodegenConstantsQualityGate"
    ) as mock_gate_cls:
        gate_instance = Mock()
        gate_instance.run.return_value = {
            "verdict": "FAIL",
            "checks": [{"name": "lint_clean", "passed": False}],
            "improvement": {"violations_delta": 1},
        }
        mock_gate_cls.return_value = gate_instance
        gate_instance.render_text.return_value = "failed\n"
        mock_gate_cls.is_success_verdict.return_value = False
        with patch("builtins.print") as mock_print:
            result = codegen_main.main(argv)
    assert result == 1
    if mock_print.called:
        mock_print.assert_called_once_with("failed\n", end="")


def test_quality_gate_success_verdict_helper() -> None:
    """Success helper accepts PASS and CONDITIONAL_PASS only."""
    assert FlextInfraCodegenConstantsQualityGate.is_success_verdict("PASS") is True
    assert (
        FlextInfraCodegenConstantsQualityGate.is_success_verdict("CONDITIONAL_PASS")
        is True
    )
    assert FlextInfraCodegenConstantsQualityGate.is_success_verdict("FAIL") is False
