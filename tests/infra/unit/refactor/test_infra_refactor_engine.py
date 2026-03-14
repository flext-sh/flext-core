"""Unit tests for FlextInfraRefactorEngine rule dispatch and project scanning."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from flext_infra.refactor import (
        FlextInfraRefactorClassReconstructorRule,
        FlextInfraRefactorEngine,
        FlextInfraRefactorEnsureFutureAnnotationsRule,
        FlextInfraRefactorImportModernizerRule,
        FlextInfraRefactorLegacyRemovalRule,
        FlextInfraRefactorMROClassMigrationRule,
        FlextInfraRefactorMRORedundancyChecker,
        FlextInfraRefactorPatternCorrectionsRule,
        FlextInfraRefactorSignaturePropagationRule,
        FlextInfraRefactorSymbolPropagationRule,
    )
except ImportError as exc:
    pytest.skip(f"refactor package unavailable: {exc}", allow_module_level=True)
from flext_tests import tm


def test_rule_dispatch_prefers_fix_action_metadata(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: custom-rule-a\n    enabled: true\n    fix_action: remove\n  - id: custom-rule-b\n    enabled: true\n    fix_action: replace_with_alias\n  - id: custom-rule-c\n    enabled: true\n    fix_action: reorder_methods\n  - id: custom-rule-d\n    enabled: true\n    fix_action: remove_inheritance_keep_class\n  - id: custom-rule-d2\n    enabled: true\n    fix_action: migrate_to_class_mro\n  - id: custom-rule-e\n    enabled: true\n    fix_action: ensure_future_annotations\n  - id: custom-rule-f\n    enabled: true\n    fix_action: propagate_symbol_renames\n    import_symbol_renames:\n      Old: New\n  - id: custom-rule-g\n    enabled: true\n    fix_action: propagate_signature_migrations\n    signature_migrations:\n      - id: migrate-keyword\n        enabled: true\n        target_simple_names:\n          - run\n        keyword_renames:\n          old: new\n  - id: custom-rule-h\n    enabled: true\n    fix_action: convert_dict_to_mapping_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    engine = FlextInfraRefactorEngine(config_path=config_path)
    result = engine.load_rules()
    tm.ok(result)
    assert len(engine.rules) == 9
    assert isinstance(engine.rules[0], FlextInfraRefactorLegacyRemovalRule)
    assert isinstance(engine.rules[1], FlextInfraRefactorImportModernizerRule)
    assert isinstance(engine.rules[2], FlextInfraRefactorClassReconstructorRule)
    assert isinstance(engine.rules[3], FlextInfraRefactorMRORedundancyChecker)
    assert isinstance(engine.rules[4], FlextInfraRefactorMROClassMigrationRule)
    assert isinstance(engine.rules[5], FlextInfraRefactorEnsureFutureAnnotationsRule)
    assert isinstance(engine.rules[6], FlextInfraRefactorSymbolPropagationRule)
    assert isinstance(engine.rules[7], FlextInfraRefactorSignaturePropagationRule)
    assert isinstance(engine.rules[8], FlextInfraRefactorPatternCorrectionsRule)


def test_rule_dispatch_fails_on_invalid_pattern_rule_config(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: custom-pattern-rule\n    enabled: true\n    fix_action: remove_redundant_casts\n".strip()
        + "\n",
        encoding="utf-8",
    )
    engine = FlextInfraRefactorEngine(config_path=config_path)
    result = engine.load_rules()
    tm.fail(result)
    assert result.error is not None
    assert "redundant_type_targets" in result.error


def test_rule_dispatch_fails_on_unknown_rule_mapping(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: custom-unknown-rule\n    enabled: true\n".strip() + "\n",
        encoding="utf-8",
    )
    engine = FlextInfraRefactorEngine(config_path=config_path)
    result = engine.load_rules()
    tm.fail(result)
    assert result.error is not None
    assert "Unknown rule mapping" in result.error


def test_engine_always_enables_class_nesting_file_rule(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: custom-import-rule\n    enabled: true\n    fix_action: replace_with_alias\n".strip()
        + "\n",
        encoding="utf-8",
    )
    engine = FlextInfraRefactorEngine(config_path=config_path)
    engine.set_rule_filters(["custom-import-rule"])
    result = engine.load_rules()
    tm.ok(result)
    assert len(engine.file_rules) == 1


def test_rule_dispatch_keeps_legacy_id_fallback_mapping(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: modernize-import-fallback\n    enabled: true\n".strip()
        + "\n",
        encoding="utf-8",
    )
    engine = FlextInfraRefactorEngine(config_path=config_path)
    result = engine.load_rules()
    tm.ok(result)
    assert len(engine.rules) == 1
    assert isinstance(engine.rules[0], FlextInfraRefactorImportModernizerRule)


def test_refactor_project_scans_tests_and_scripts_dirs(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\nrefactor_engine:\n  project_scan_dirs:\n    - tests\n    - scripts\n".strip()
        + "\n",
        encoding="utf-8",
    )
    project_root = tmp_path / "sample"
    tests_dir = project_root / "tests"
    scripts_dir = project_root / "scripts"
    tests_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text("import os\n", encoding="utf-8")
    (scripts_dir / "task.py").write_text("import sys\n", encoding="utf-8")
    engine = FlextInfraRefactorEngine(config_path=config_path)
    loaded = engine.load_rules()
    tm.ok(loaded)
    results = engine.refactor_project(project_root)
    assert len(results) == 2
    assert all(result.success for result in results)
    assert all(result.modified for result in results)


def test_refactor_files_skips_non_python_inputs(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text('refactor_engine:\n  project_scan_dirs: ["src"]\n')
    py_file = tmp_path / "sample.py"
    py_file.write_text("import os\n", encoding="utf-8")
    md_file = tmp_path / "README.md"
    md_file.write_text("# doc\n", encoding="utf-8")
    engine = FlextInfraRefactorEngine(config_path=config_path)
    loaded = engine.load_rules()
    tm.ok(loaded)
    results = engine.refactor_files([py_file, md_file], dry_run=True)
    assert len(results) == 2
    md_result = next(item for item in results if item.file_path == md_file)
    assert md_result.success
    assert not md_result.modified
    assert "Skipped non-Python file" in md_result.changes
