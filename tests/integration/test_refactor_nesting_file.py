"""Integration test for single-file class-nesting refactor flow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

try:
    from flext_infra.refactor.rules.class_nesting import ClassNestingRefactorRule
except ImportError as exc:
    pytest.skip(f"class nesting rule unavailable: {exc}", allow_module_level=True)

pytestmark = [pytest.mark.integration]


def test_class_nesting_refactor_single_file_end_to_end(tmp_path: Path) -> None:
    fixture_file = Path("tests/fixtures/namespace_validator/rule0_valid.py")
    source = fixture_file.read_text(encoding="utf-8")
    dispatcher_dir = tmp_path / "_dispatcher"
    dispatcher_dir.mkdir(parents=True, exist_ok=True)
    target_file = dispatcher_dir / "single_file_refactor_target.py"
    target_file.write_text(
        source
        + "\n\n"
        + "from pkg import TimeoutEnforcer\n\n"
        + "class TimeoutEnforcer:\n"
        + "    pass\n\n"
        + "def build(value: TimeoutEnforcer) -> TimeoutEnforcer:\n"
        + "    if isinstance(value, TimeoutEnforcer):\n"
        + "        return TimeoutEnforcer()\n"
        + "    return value\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "class-nesting-mappings.yml"
    config_path.write_text(
        f"confidence_threshold: low\nclass_nesting:\n  - loose_name: TimeoutEnforcer\n    current_file: {target_file.as_posix()}\n    target_namespace: FlextDispatcher\n    target_name: TimeoutEnforcer\n    confidence: high\nhelper_consolidation: []\n",
        encoding="utf-8",
    )
    rule = ClassNestingRefactorRule(config_path=config_path)
    result = rule.apply(target_file, dry_run=False)
    assert result.success
    assert result.modified
    assert result.refactored_code is not None
    refactored_code = target_file.read_text(encoding="utf-8")
    assert "from pkg import FlextDispatcher" in refactored_code
    assert "from pkg import TimeoutEnforcer" not in refactored_code
    assert "class FlextDispatcher:" in refactored_code
    assert "    class TimeoutEnforcer:" in refactored_code
    assert (
        "def build(value: FlextDispatcher.TimeoutEnforcer) -> FlextDispatcher.TimeoutEnforcer:"
        in refactored_code
    )
    assert "return FlextDispatcher.TimeoutEnforcer()" in refactored_code
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(target_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
