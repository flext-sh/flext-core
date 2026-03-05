from __future__ import annotations

from pathlib import Path

import libcst as cst

from flext_infra.refactor import (
    FlextInfraRefactorClassReconstructorRule,
    FlextInfraRefactorEngine,
    FlextInfraRefactorEnsureFutureAnnotationsRule,
    FlextInfraRefactorImportModernizerRule,
    FlextInfraRefactorLegacyRemovalRule,
    FlextInfraRefactorMRORedundancyChecker,
    FlextInfraRefactorResult,
    FlextInfraRefactorSignaturePropagationRule,
    FlextInfraRefactorSymbolPropagationRule,
)


def test_import_modernizer_partial_import_keeps_unmapped_symbols() -> None:
    source = (
        "from flext_core.constants import PLATFORM, KEEP\n\n"
        "value = PLATFORM\n"
        "other = KEEP\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "from flext_core import c" in updated
    assert "from flext_core.constants import KEEP" in updated
    assert "value = c.System.PLATFORM" in updated
    assert "other = KEEP" in updated


def test_import_modernizer_updates_aliased_symbol_usage() -> None:
    source = "from flext_core.constants import PLATFORM as P\n\nvalue = P\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "from flext_core.constants import PLATFORM as P" not in updated
    assert "from flext_core import c" in updated
    assert "value = c.System.PLATFORM" in updated


def test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias() -> None:
    source = (
        "from flext_core.constants import PLATFORM as P, KEEP as K\n\n"
        "value = P\n"
        "other = K\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "from flext_core import c" in updated
    assert "from flext_core.constants import KEEP as K" in updated
    assert "value = c.System.PLATFORM" in updated
    assert "other = K" in updated


def test_ensure_future_annotations_after_docstring() -> None:
    source = '"""doc"""\n\nimport os\n'
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorEnsureFutureAnnotationsRule({
        "id": "ensure-future-annotations"
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert '"""doc"""\n\nfrom __future__ import annotations\n\nimport os\n' in updated


def test_import_modernizer_adds_c_when_existing_c_is_aliased() -> None:
    source = (
        "from flext_core import c as consts\n"
        "from flext_core.constants import PLATFORM\n\n"
        "value = PLATFORM\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "from flext_core import c as consts" in updated
    assert "from flext_core import c" in updated
    assert "value = c.System.PLATFORM" in updated


def test_ensure_future_annotations_moves_existing_import_to_top() -> None:
    source = "import os\nfrom __future__ import annotations\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorEnsureFutureAnnotationsRule({
        "id": "ensure-future-annotations"
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert updated.startswith("from __future__ import annotations\n")
    assert "\nimport os\n" in updated


def test_import_modernizer_does_not_rewrite_function_parameter_shadow() -> None:
    source = (
        "from flext_core.constants import PLATFORM as P\n\n"
        "def f(P: str) -> str:\n"
        "    return P\n\n"
        "value = P\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "def f(P: str) -> str:" in updated
    assert "return P" in updated
    assert "value = c.System.PLATFORM" in updated


def test_import_modernizer_does_not_rewrite_rebound_local_name_usage() -> None:
    source = (
        "from flext_core.constants import PLATFORM\n\n"
        'PLATFORM = "local"\n'
        "value = PLATFORM\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "from flext_core.constants import PLATFORM" not in updated
    assert "from flext_core import c" in updated
    assert 'PLATFORM = "local"' in updated
    assert "value = PLATFORM" in updated


def test_legacy_wrapper_function_is_inlined_as_alias() -> None:
    source = "def run(value: int) -> int:\n    return execute(value)\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorLegacyRemovalRule({"id": "remove-wrapper-functions"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "def run" not in updated
    assert "run = execute" in updated


def test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias() -> None:
    source = "def run(a: int, b: int) -> int:\n    return execute(a=a, b=b)\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorLegacyRemovalRule({"id": "remove-wrapper-functions"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "def run" not in updated
    assert "run = execute" in updated


def test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias() -> None:
    source = (
        "def run(a: int, *args: object, **kwargs: object) -> int:\n"
        "    return execute(a, *args, **kwargs)\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorLegacyRemovalRule({"id": "remove-wrapper-functions"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "def run" not in updated
    assert "run = execute" in updated


def test_legacy_wrapper_non_passthrough_is_not_inlined() -> None:
    source = "def run(a: int, b: int) -> int:\n    return execute(a, b + 1)\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorLegacyRemovalRule({"id": "remove-wrapper-functions"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "def run" in updated
    assert "run = execute" not in updated


def test_legacy_rule_uses_fix_action_remove_for_aliases() -> None:
    source = "OldName = NewName\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorLegacyRemovalRule({
        "id": "custom-legacy-rule",
        "fix_action": "remove",
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "OldName = NewName" not in updated


def test_legacy_import_bypass_collapses_to_primary_import() -> None:
    source = (
        "try:\n"
        "    from new_mod import Thing\n"
        "except ImportError:\n"
        "    from old_mod import Thing\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorLegacyRemovalRule({"id": "remove-import-bypasses"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "try:" not in updated
    assert "from new_mod import Thing" in updated
    assert "from old_mod import Thing" not in updated


def test_lazy_import_rule_hoists_import_to_module_level() -> None:
    source = "def build() -> None:\n    import json\n    return None\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({"id": "ban-lazy-imports"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert updated.startswith("import json\n")
    assert "def build() -> None:\n    return None\n" in updated


def test_lazy_import_rule_uses_fix_action_for_hoist() -> None:
    source = "def build() -> None:\n    import json\n    return None\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "custom-lazy-rule",
        "fix_action": "hoist_to_module_top",
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert updated.startswith("import json\n")
    assert "def build() -> None:\n    return None\n" in updated


def test_import_modernizer_skips_when_runtime_alias_name_is_blocked() -> None:
    source = (
        "from flext_infra.constants import c\n"
        "from flext_core.constants import PLATFORM\n\n"
        "value = PLATFORM\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "from flext_infra.constants import c" in updated
    assert "from flext_core.constants import PLATFORM" in updated
    assert "from flext_core import c" not in updated
    assert "value = PLATFORM" in updated


def test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function() -> (
    None
):
    source = (
        "from flext_core.constants import PLATFORM\n\n"
        "def compute(c: object) -> object:\n"
        "    return PLATFORM\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorImportModernizerRule({
        "id": "modernize-constants-import",
        "module": "flext_core.constants",
        "symbol_mapping": {"PLATFORM": "c.System.PLATFORM"},
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "from flext_core import c" not in updated
    assert "from flext_core.constants import PLATFORM" in updated
    assert "return PLATFORM" in updated
    assert "c.System.PLATFORM" not in updated


def test_class_reconstructor_reorders_methods_by_config() -> None:
    source = (
        "class C:\n"
        "    def b(self) -> None:\n"
        "        return None\n\n"
        "    def __init__(self) -> None:\n"
        "        return None\n\n"
        "    def a(self) -> None:\n"
        "        return None\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorClassReconstructorRule({
        "id": "reorder-class-methods",
        "method_order": [
            {"category": "magic", "patterns": [r"^__.+__$"]},
            {"category": "public", "visibility": "public"},
        ],
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert updated.index("def __init__") < updated.index("def a")
    assert updated.index("def a") < updated.index("def b")


def test_mro_redundancy_checker_removes_nested_attribute_inheritance() -> None:
    source = "class Outer:\n    class Inner(Outer.Base):\n        pass\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorMRORedundancyChecker({"id": "fix-mro-redeclaration"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "class Inner:" in updated
    assert "Outer.Base" not in updated


def test_symbol_propagation_renames_import_and_local_references() -> None:
    source = (
        "from flext_infra.refactor import LegacyRemovalRule\n\n"
        "rule_cls = LegacyRemovalRule\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorSymbolPropagationRule({
        "id": "propagate-refactor-api-renames",
        "fix_action": "propagate_symbol_renames",
        "target_modules": ["flext_infra.refactor"],
        "import_symbol_renames": {
            "LegacyRemovalRule": "FlextInfraRefactorLegacyRemovalRule"
        },
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert (
        "from flext_infra.refactor import FlextInfraRefactorLegacyRemovalRule"
        in updated
    )
    assert "rule_cls = FlextInfraRefactorLegacyRemovalRule" in updated


def test_symbol_propagation_keeps_alias_reference_when_asname_used() -> None:
    source = (
        "from flext_infra.refactor import LegacyRemovalRule as Legacy\n\n"
        "rule_cls = Legacy\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorSymbolPropagationRule({
        "id": "propagate-refactor-api-renames",
        "fix_action": "propagate_symbol_renames",
        "target_modules": ["flext_infra.refactor"],
        "import_symbol_renames": {
            "LegacyRemovalRule": "FlextInfraRefactorLegacyRemovalRule"
        },
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert (
        "from flext_infra.refactor import FlextInfraRefactorLegacyRemovalRule as Legacy"
        in updated
    )
    assert "rule_cls = Legacy" in updated


def test_symbol_propagation_updates_mro_base_references() -> None:
    source = (
        "from flext_infra.refactor import LegacyRemovalRule\n\n"
        "class RuleV2(LegacyRemovalRule):\n"
        "    pass\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorSymbolPropagationRule({
        "id": "propagate-refactor-api-renames",
        "fix_action": "propagate_symbol_renames",
        "target_modules": ["flext_infra.refactor"],
        "import_symbol_renames": {
            "LegacyRemovalRule": "FlextInfraRefactorLegacyRemovalRule"
        },
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "class RuleV2(FlextInfraRefactorLegacyRemovalRule):" in updated


def test_signature_propagation_renames_call_keyword() -> None:
    source = "result = migrate(project_root=root, dry_run=True)\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorSignaturePropagationRule({
        "id": "propagate-refactor-signature-migrations",
        "fix_action": "propagate_signature_migrations",
        "signature_migrations": [
            {
                "id": "migrate-project-root-to-workspace-root",
                "enabled": True,
                "target_simple_names": ["migrate"],
                "keyword_renames": {
                    "project_root": "workspace_root",
                },
            }
        ],
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "migrate(workspace_root=root, dry_run=True)" in updated


def test_signature_propagation_removes_and_adds_keywords() -> None:
    source = "run(legacy=True)\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorSignaturePropagationRule({
        "id": "propagate-refactor-signature-migrations",
        "fix_action": "propagate_signature_migrations",
        "signature_migrations": [
            {
                "id": "run-signature-v2",
                "enabled": True,
                "target_simple_names": ["run"],
                "remove_keywords": ["legacy"],
                "add_keywords": {"mode": '"modern"'},
            }
        ],
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "run(mode" in updated
    assert "modern" in updated


def test_rule_dispatch_prefers_fix_action_metadata(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)

    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")

    (rules_dir / "rules.yml").write_text(
        """
rules:
  - id: custom-rule-a
    enabled: true
    fix_action: remove
  - id: custom-rule-b
    enabled: true
    fix_action: replace_with_alias
  - id: custom-rule-c
    enabled: true
    fix_action: reorder_methods
  - id: custom-rule-d
    enabled: true
    fix_action: remove_inheritance_keep_class
  - id: custom-rule-e
    enabled: true
    fix_action: ensure_future_annotations
  - id: custom-rule-f
    enabled: true
    fix_action: propagate_symbol_renames
  - id: custom-rule-g
    enabled: true
    fix_action: propagate_signature_migrations
""".strip()
        + "\n",
        encoding="utf-8",
    )

    engine = FlextInfraRefactorEngine(config_path=config_path)
    result = engine.load_rules()

    assert result.is_success
    assert len(engine.rules) == 7
    assert isinstance(engine.rules[0], FlextInfraRefactorLegacyRemovalRule)
    assert isinstance(engine.rules[1], FlextInfraRefactorImportModernizerRule)
    assert isinstance(engine.rules[2], FlextInfraRefactorClassReconstructorRule)
    assert isinstance(engine.rules[3], FlextInfraRefactorMRORedundancyChecker)
    assert isinstance(engine.rules[4], FlextInfraRefactorEnsureFutureAnnotationsRule)
    assert isinstance(engine.rules[5], FlextInfraRefactorSymbolPropagationRule)
    assert isinstance(engine.rules[6], FlextInfraRefactorSignaturePropagationRule)


def test_rule_dispatch_fails_on_unknown_rule_mapping(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)

    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")

    (rules_dir / "rules.yml").write_text(
        """
rules:
  - id: custom-unknown-rule
    enabled: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    engine = FlextInfraRefactorEngine(config_path=config_path)
    result = engine.load_rules()

    assert not result.is_success
    assert result.error is not None
    assert "Unknown rule mapping" in result.error


def test_rule_dispatch_keeps_legacy_id_fallback_mapping(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)

    config_path = tmp_path / "config.yml"
    config_path.write_text("engine: test\n", encoding="utf-8")

    (rules_dir / "rules.yml").write_text(
        """
rules:
  - id: modernize-import-fallback
    enabled: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    engine = FlextInfraRefactorEngine(config_path=config_path)
    result = engine.load_rules()

    assert result.is_success
    assert len(engine.rules) == 1
    assert isinstance(engine.rules[0], FlextInfraRefactorImportModernizerRule)


def test_class_reconstructor_skips_interleaved_non_method_members() -> None:
    source = (
        "class C:\n"
        "    def b(self) -> None:\n"
        "        return None\n\n"
        "    alias = b\n\n"
        "    def a(self) -> None:\n"
        "        return None\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorClassReconstructorRule({
        "id": "reorder-class-methods",
        "method_order": [
            {"category": "magic", "patterns": [r"^__.+__$"]},
            {"category": "public", "visibility": "public"},
        ],
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert updated == source


def test_class_reconstructor_reorders_each_contiguous_method_block() -> None:
    source = (
        "class C:\n"
        "    def b(self) -> None:\n"
        "        return None\n\n"
        "    def a(self) -> None:\n"
        "        return None\n\n"
        "    alias = a\n\n"
        "    def d(self) -> None:\n"
        "        return None\n\n"
        "    def c(self) -> None:\n"
        "        return None\n"
    )
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorClassReconstructorRule({
        "id": "reorder-class-methods",
        "method_order": [
            {"category": "public", "visibility": "public"},
        ],
    })

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert updated.index("def a") < updated.index("def b")
    assert updated.index("def c") < updated.index("def d")
    assert "alias = a" in updated


def test_mro_checker_keeps_external_attribute_base() -> None:
    source = "class Outer:\n    class Inner(pkg.Base):\n        pass\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorMRORedundancyChecker({"id": "fix-mro-redeclaration"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "class Inner(pkg.Base):" in updated


def test_refactor_project_scans_tests_and_scripts_dirs(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        """
rules:
  - id: ensure-future-annotations
    enabled: true
    fix_action: ensure_future_annotations
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
refactor_engine:
  project_scan_dirs:
    - tests
    - scripts
""".strip()
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
    assert loaded.is_success

    results = engine.refactor_project(project_root)
    assert len(results) == 2
    assert all(result.success for result in results)
    assert all(result.modified for result in results)


def test_build_impact_map_extracts_rename_entries() -> None:
    result = FlextInfraRefactorResult(
        file_path=Path("/tmp/flext-core/src/module.py"),
        success=True,
        modified=True,
        changes=[
            "Renamed imported symbol: LegacyRemovalRule -> FlextInfraRefactorLegacyRemovalRule (local=LegacyRemovalRule)"
        ],
        refactored_code="",
    )

    impact_map = FlextInfraRefactorEngine.build_impact_map([result])

    assert len(impact_map) == 1
    assert impact_map[0]["kind"] == "rename"
    assert impact_map[0]["old"] == "LegacyRemovalRule"
    assert impact_map[0]["new"] == "FlextInfraRefactorLegacyRemovalRule"


def test_build_impact_map_extracts_signature_entries() -> None:
    result = FlextInfraRefactorResult(
        file_path=Path("/tmp/flext-core/src/module.py"),
        success=True,
        modified=True,
        changes=[
            "[run-signature-v2] Removed keyword: legacy",
            '[run-signature-v2] Added keyword: mode="modern"',
        ],
        refactored_code="",
    )

    impact_map = FlextInfraRefactorEngine.build_impact_map([result])

    kinds = {entry["kind"] for entry in impact_map}
    assert "signature_remove" in kinds
    assert "signature_add" in kinds
