"""Unit tests for FlextInfraRefactorLegacyRemovalRule and EnsureFutureAnnotationsRule."""

from __future__ import annotations

import libcst as cst
import pytest

try:
    from flext_infra.refactor import (
        FlextInfraRefactorEnsureFutureAnnotationsRule,
        FlextInfraRefactorLegacyRemovalRule,
    )
except ImportError as exc:
    pytest.skip(f"refactor package unavailable: {exc}", allow_module_level=True)


def test_ensure_future_annotations_after_docstring() -> None:
    source = '"""doc"""\n\nimport os\n'
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorEnsureFutureAnnotationsRule({
        "id": "ensure-future-annotations",
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert '"""doc"""\n\nfrom __future__ import annotations\n\nimport os\n' in updated


def test_ensure_future_annotations_moves_existing_import_to_top() -> None:
    source = "import os\nfrom __future__ import annotations\n"
    tree = cst.parse_module(source)
    rule = FlextInfraRefactorEnsureFutureAnnotationsRule({
        "id": "ensure-future-annotations",
    })
    updated_tree, _ = rule.apply(tree)
    updated = updated_tree.code
    assert updated.startswith("from __future__ import annotations\n")
    assert "\nimport os\n" in updated


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
    source = "def run(a: int, *args, **kwargs: t.Scalar) -> int:\n    return execute(a, *args, **kwargs)\n"
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
