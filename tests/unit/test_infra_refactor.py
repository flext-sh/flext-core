"""Unit tests for flext_infra.refactor rules and engine behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import overload, override

import libcst as cst
import pytest

from flext_core import r
from flext_infra import m
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
    FlextInfraRefactorViolationAnalyzer,
)
from flext_infra.refactor.safety import FlextInfraRefactorSafetyManager


def test_import_modernizer_partial_import_keeps_unmapped_symbols() -> None:
    source = "from flext_core.constants import PLATFORM, KEEP\n\nvalue = PLATFORM\nother = KEEP\n"
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
    source = "from flext_core.constants import PLATFORM as P, KEEP as K\n\nvalue = P\nother = K\n"
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
    source = "from flext_core import c as consts\nfrom flext_core.constants import PLATFORM\n\nvalue = PLATFORM\n"
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
    source = "from flext_core.constants import PLATFORM as P\n\ndef f(P: str) -> str:\n    return P\n\nvalue = P\n"
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
    source = 'from flext_core.constants import PLATFORM\n\nPLATFORM = "local"\nvalue = PLATFORM\n'
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
    source = "def run(a: int, *args: object, **kwargs: object) -> int:\n    return execute(a, *args, **kwargs)\n"
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
    source = "try:\n    from new_mod import Thing\nexcept ImportError:\n    from old_mod import Thing\n"
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
    source = "from flext_infra.constants import c\nfrom flext_core.constants import PLATFORM\n\nvalue = PLATFORM\n"
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
    source = "from flext_core.constants import PLATFORM\n\ndef compute(c: object) -> object:\n    return PLATFORM\n"
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
    source = "class C:\n    def b(self) -> None:\n        return None\n\n    def __init__(self) -> None:\n        return None\n\n    def a(self) -> None:\n        return None\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorClassReconstructorRule({
        "id": "reorder-class-methods",
        "method_order": [
            {"category": "magic", "patterns": ["^__.+__$"]},
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
    source = "from flext_infra.refactor import LegacyRemovalRule\n\nrule_cls = LegacyRemovalRule\n"
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
    source = "from flext_infra.refactor import LegacyRemovalRule as Legacy\n\nrule_cls = Legacy\n"
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
    source = "from flext_infra.refactor import LegacyRemovalRule\n\nclass RuleV2(LegacyRemovalRule):\n    pass\n"
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
                "keyword_renames": {"project_root": "workspace_root"},
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


def test_pattern_rule_converts_dict_annotations_to_mapping() -> None:
    source = "def f(data: dict[str, t.ContainerValue]) -> dict[str, t.ContainerValue]:\n    return data\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "fix-container-invariance-annotations",
        "fix_action": "convert_dict_to_mapping_annotations",
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert "from collections.abc import Mapping" in updated
    assert "data: Mapping[str, t.ContainerValue]" in updated
    assert "-> dict[str, t.ContainerValue]" in updated


def test_pattern_rule_optionally_converts_return_annotations_to_mapping() -> None:
    source = "def f(data: dict[str, t.ContainerValue]) -> dict[str, t.ContainerValue]:\n    return data\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "fix-container-invariance-annotations",
        "fix_action": "convert_dict_to_mapping_annotations",
        "include_return_annotations": True,
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert "data: Mapping[str, t.ContainerValue]" in updated
    assert "-> Mapping[str, t.ContainerValue]" in updated


def test_pattern_rule_keeps_dict_param_when_subscript_mutated() -> None:
    source = 'def f(data: dict[str, t.ContainerValue]) -> dict[str, t.ContainerValue]:\n    data["k"] = "v"\n    return data\n'
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "fix-container-invariance-annotations",
        "fix_action": "convert_dict_to_mapping_annotations",
    })
    updated_tree, changes = rule.apply(tree)
    updated = updated_tree.code
    assert updated == source
    assert changes == []


def test_pattern_rule_keeps_dict_param_when_copy_used() -> None:
    source = "def f(data: dict[str, t.ContainerValue]) -> dict[str, t.ContainerValue]:\n    clone = data.copy()\n    return clone\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "fix-container-invariance-annotations",
        "fix_action": "convert_dict_to_mapping_annotations",
    })
    updated_tree, changes = rule.apply(tree)
    updated = updated_tree.code
    assert updated == source
    assert changes == []


def test_pattern_rule_skips_overload_signatures() -> None:
    source = "from typing import overload\n\n@overload\ndef f(data: dict[str, t.ContainerValue]) -> str: ...\n\ndef f(data: dict[str, t.ContainerValue]) -> str:\n    return str(data)\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "fix-container-invariance-annotations",
        "fix_action": "convert_dict_to_mapping_annotations",
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert "@overload" in updated
    assert "def f(data: dict[str, t.ContainerValue]) -> str: ..." in updated
    assert "def f(data: Mapping[str, t.ContainerValue]) -> str:" in updated


def test_pattern_rule_removes_configured_redundant_casts() -> None:
    source = 'value = cast("m.ConfigMap", result.unwrap_or(m.ConfigMap(root={})))\n'
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "remove-validated-redundant-casts",
        "fix_action": "remove_redundant_casts",
        "redundant_type_targets": ["m.ConfigMap"],
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert "cast(" not in updated
    assert "value = result.unwrap_or(m.ConfigMap(root={}))" in updated


def test_pattern_rule_removes_nested_type_object_cast_chain() -> None:
    source = 'value = cast("type", cast("object", FlextSettings))\n'
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "remove-validated-redundant-casts",
        "fix_action": "remove_redundant_casts",
        "redundant_type_targets": ["type"],
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert "cast(" not in updated
    assert "value = FlextSettings" in updated


def test_pattern_rule_keeps_type_cast_when_not_nested_object_cast() -> None:
    source = 'metadata_cls = cast("type", FlextRuntime.Metadata)\n'
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorPatternCorrectionsRule({
        "id": "remove-validated-redundant-casts",
        "fix_action": "remove_redundant_casts",
        "redundant_type_targets": ["type"],
    })
    updated_tree, changes = rule.apply(tree)
    updated = updated_tree.code
    assert updated == source
    assert changes == []


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
    assert result.is_success
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
    assert not result.is_success
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
    assert not result.is_success
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
    _ = engine.set_rule_filters(["custom-import-rule"])
    result = engine.load_rules()
    assert result.is_success
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
    assert result.is_success
    assert len(engine.rules) == 1
    assert isinstance(engine.rules[0], FlextInfraRefactorImportModernizerRule)


def test_class_reconstructor_skips_interleaved_non_method_members() -> None:
    source = "class C:\n    def b(self) -> None:\n        return None\n\n    alias = b\n\n    def a(self) -> None:\n        return None\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorClassReconstructorRule({
        "id": "reorder-class-methods",
        "method_order": [
            {"category": "magic", "patterns": ["^__.+__$"]},
            {"category": "public", "visibility": "public"},
        ],
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert updated == source


def test_class_reconstructor_reorders_each_contiguous_method_block() -> None:
    source = "class C:\n    def b(self) -> None:\n        return None\n\n    def a(self) -> None:\n        return None\n\n    alias = a\n\n    def d(self) -> None:\n        return None\n\n    def c(self) -> None:\n        return None\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorClassReconstructorRule({
        "id": "reorder-class-methods",
        "method_order": [{"category": "public", "visibility": "public"}],
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
    assert loaded.is_success
    results = engine.refactor_project(project_root)
    assert len(results) == 2
    assert all(result.success for result in results)
    assert all(result.modified for result in results)


def test_build_impact_map_extracts_rename_entries() -> None:
    result = m.Infra.Refactor.Result(
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
    result = m.Infra.Refactor.Result(
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


def test_violation_analysis_counts_massive_patterns(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'refactor_engine:\n  project_scan_dirs: ["src"]\n', encoding="utf-8"
    )
    project_root = tmp_path / "project"
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "sample.py"
    target_file.write_text(
        'from typing import Mapping, cast\nfrom flext_core.models import User\n\ndef f(data: dict[str, t.ContainerValue]) -> dict[str, t.ContainerValue]:\n    value = cast("m.ConfigMap", data)\n    return value\n',
        encoding="utf-8",
    )
    engine = FlextInfraRefactorEngine(config_path=config_path)
    _ = engine.load_config()
    files = engine.collect_project_files(project_root)
    result = FlextInfraRefactorViolationAnalyzer.analyze_files(files)
    totals = result.totals
    assert "container_invariance" in totals
    assert "redundant_cast" in totals
    assert "direct_submodule_import" in totals
    assert totals["container_invariance"] >= 2
    assert totals["redundant_cast"] >= 1
    assert totals["direct_submodule_import"] >= 1


def test_main_analyze_violations_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'refactor_engine:\n  project_scan_dirs: ["src"]\n', encoding="utf-8"
    )
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "sample.py"
    target_file.write_text("import os\n", encoding="utf-8")
    original = target_file.read_text(encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refactor-engine",
            "--file",
            str(target_file),
            "--analyze-violations",
            "--config",
            str(config_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        FlextInfraRefactorEngine.main()
    assert exc_info.value.code == 0
    assert target_file.read_text(encoding="utf-8") == original


def test_main_analyze_violations_writes_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        'refactor_engine:\n  project_scan_dirs: ["src"]\n', encoding="utf-8"
    )
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "sample.py"
    target_file.write_text("import os\n", encoding="utf-8")
    report_path = tmp_path / "reports" / "analysis.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refactor-engine",
            "--file",
            str(target_file),
            "--analyze-violations",
            "--analysis-output",
            str(report_path),
            "--config",
            str(config_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        FlextInfraRefactorEngine.main()
    assert exc_info.value.code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["files_scanned"] == 1
    assert payload["totals"] == {}


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
    assert loaded.is_success
    results = engine.refactor_files([py_file, md_file], dry_run=True)
    assert len(results) == 2
    md_result = next(item for item in results if item.file_path == md_file)
    assert md_result.success
    assert not md_result.modified
    assert "Skipped non-Python file" in md_result.changes


def test_violation_analyzer_skips_non_utf8_files(tmp_path: Path) -> None:
    file_path = tmp_path / "binary.py"
    file_path.write_bytes(b"\x80\x81\x82")
    result = FlextInfraRefactorViolationAnalyzer.analyze_files([file_path])
    assert result.files_scanned == 1
    assert result.totals == {}


class EngineSafetyStub(FlextInfraRefactorSafetyManager):
    """Test double for safety manager lifecycle operations."""

    def __init__(self) -> None:
        """Initialize call capture state for assertions."""
        super().__init__()
        self.calls: list[str] = []

    @override
    def create_pre_transformation_stash(
        self, workspace_root: Path, *, label: str = "flext-refactor-pre-transform"
    ) -> r[str]:
        _ = workspace_root
        _ = label
        self.calls.append("stash")
        return r[str].ok("stash@{0}")

    @override
    def save_checkpoint_state(
        self,
        workspace_root: Path,
        *,
        status: str,
        stash_ref: str,
        processed_targets: list[str],
    ) -> r[bool]:
        _ = workspace_root
        _ = status
        _ = stash_ref
        _ = processed_targets
        self.calls.append("checkpoint")
        return r[bool].ok(True)

    @override
    def run_semantic_validation(self, workspace_root: Path) -> r[bool]:
        _ = workspace_root
        self.calls.append("validate")
        return r[bool].ok(True)

    @override
    def clear_checkpoint(self) -> r[bool]:
        self.calls.append("clear")
        return r[bool].ok(True)

    @override
    def request_emergency_stop(self, reason: str) -> None:
        _ = reason
        self.calls.append("stop")

    @overload
    def rollback(self, workspace_root: Path, stash_ref: str = "") -> r[bool]: ...

    @overload
    def rollback(self, workspace_root: str, /) -> None: ...

    @override
    def rollback(
        self, workspace_root: Path | str, stash_ref: str = ""
    ) -> r[bool] | None:
        _ = stash_ref
        self.calls.append("rollback")
        if isinstance(workspace_root, Path):
            return r[bool].ok(True)
        return None

    @override
    def is_emergency_stop_requested(self) -> bool:
        self.calls.append("is_stop")
        return False


def test_refactor_project_integrates_safety_manager(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text('refactor_engine:\n  project_scan_dirs: ["src"]\n')
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "sample.py").write_text("import os\n", encoding="utf-8")
    engine = FlextInfraRefactorEngine(config_path=config_path)
    stub = EngineSafetyStub()
    engine.safety_manager = stub
    loaded = engine.load_rules()
    assert loaded.is_success
    results = engine.refactor_project(tmp_path, dry_run=False, apply_safety=True)
    assert results
    assert all(item.success for item in results)
    assert stub.calls == ["stash", "checkpoint", "validate", "clear"]
