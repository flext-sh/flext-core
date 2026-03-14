"""Unit tests for FlextInfraRefactorImportModernizerRule."""

from __future__ import annotations

import libcst as cst
import pytest

try:
    from flext_infra.refactor import (
        FlextInfraRefactorImportModernizerRule,
    )
except ImportError as exc:
    pytest.skip(f"refactor package unavailable: {exc}", allow_module_level=True)


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
    source = "from flext_core.constants import PLATFORM\n\ndef compute(c):\n    return PLATFORM\n"
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
