"""Integration test for single-file class-nesting refactor flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import (
    FlextInfraClassNestingRefactorRule as ClassNestingRefactorRule,
    m,
    u,
)

pytestmark = [pytest.mark.integration]


def _apply_rule(
    workspace_root: Path,
    file_path: Path,
    rule: ClassNestingRefactorRule,
    *,
    dry_run: bool,
) -> m.Infra.Result:
    rope_project = u.Infra.init_rope_project(workspace_root)
    try:
        resource = u.Infra.get_resource_from_path(rope_project, file_path)
        if resource is None:
            raise FileNotFoundError(file_path)
        return rule.apply(rope_project, resource, dry_run=dry_run)
    finally:
        rope_project.close()


def test_class_nesting_refactor_single_file_end_to_end(tmp_path: Path) -> None:
    """Verify class nesting refactor handles unknown module families gracefully."""
    fixture_file = (
        Path(__file__).parent.parent / "fixtures/namespace_validator/rule0_valid.py"
    )
    source = fixture_file.read_text(encoding="utf-8")
    utilities_dir = tmp_path / "_utilities"
    utilities_dir.mkdir(parents=True, exist_ok=True)
    target_file = utilities_dir / "single_file_refactor_target.py"
    target_file.write_text(
        source
        + "\n\n"
        + "from pkg import ResultHelpers\n\n"
        + "class ResultHelpers:\n"
        + "    pass\n\n"
        + "def build(value: ResultHelpers) -> ResultHelpers:\n"
        + "    if isinstance(value, ResultHelpers):\n"
        + "        return ResultHelpers()\n"
        + "    return value\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "class-nesting-mappings.yml"
    config_path.write_text(
        f"confidence_threshold: low\nclass_nesting:\n  - loose_name: ResultHelpers\n    current_file: {target_file.as_posix()}\n    target_namespace: FlextUtilities\n    target_name: ResultHelpers\n    confidence: high\nhelper_consolidation: []\n",
        encoding="utf-8",
    )
    rule = ClassNestingRefactorRule(config_path=config_path)
    result = _apply_rule(tmp_path, target_file, rule, dry_run=False)
    # Module family policy rejects temp paths as unknown_module_family
    assert not result.success
    assert result.changes
    assert any("unknown_module_family" in v for v in result.changes)
