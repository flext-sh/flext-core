from __future__ import annotations

import libcst as cst

from flext_infra.refactor import (
    ClassReconstructorRule,
    EnsureFutureAnnotationsRule,
    ImportModernizerRule,
    LegacyRemovalRule,
    MRORedundancyChecker,
)


def test_import_modernizer_partial_import_keeps_unmapped_symbols() -> None:
    source = (
        "from flext_core.constants import PLATFORM, KEEP\n\n"
        "value = PLATFORM\n"
        "other = KEEP\n"
    )
    tree = cst.parse_module(source)
    rule = ImportModernizerRule({
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
    rule = ImportModernizerRule({
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
    rule = ImportModernizerRule({
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
    rule = EnsureFutureAnnotationsRule({"id": "ensure-future-annotations"})

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
    rule = ImportModernizerRule({
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
    rule = EnsureFutureAnnotationsRule({"id": "ensure-future-annotations"})

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
    rule = ImportModernizerRule({
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
    rule = ImportModernizerRule({
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
    rule = LegacyRemovalRule({"id": "remove-wrapper-functions"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "def run" not in updated
    assert "run = execute" in updated


def test_legacy_import_bypass_collapses_to_primary_import() -> None:
    source = (
        "try:\n"
        "    from new_mod import Thing\n"
        "except ImportError:\n"
        "    from old_mod import Thing\n"
    )
    tree = cst.parse_module(source)
    rule = LegacyRemovalRule({"id": "remove-import-bypasses"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "try:" not in updated
    assert "from new_mod import Thing" in updated
    assert "from old_mod import Thing" not in updated


def test_lazy_import_rule_hoists_import_to_module_level() -> None:
    source = "def build() -> None:\n    import json\n    return None\n"
    tree = cst.parse_module(source)
    rule = ImportModernizerRule({"id": "ban-lazy-imports"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert updated.startswith("import json\n")
    assert "def build() -> None:\n    return None\n" in updated


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
    rule = ClassReconstructorRule({
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
    rule = MRORedundancyChecker({"id": "fix-mro-redeclaration"})

    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code

    assert "class Inner:" in updated
    assert "Outer.Base" not in updated
