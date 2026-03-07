"""Unit tests for class nesting transformer behavior."""

from __future__ import annotations

import libcst as cst

from flext_infra.refactor.transformers.class_nesting import (
    FlextInfraRefactorClassNestingTransformer,
)


def test_class_nesting_moves_top_level_class_into_new_namespace() -> None:
    source = (
        "@decorator\n"
        "class TimeoutEnforcer[T](BaseEnforcer, Generic[T], metaclass=Meta):\n"
        '    """timeout docs"""\n'
        "    value: T\n"
    )

    transformed = cst.parse_module(source).visit(
        FlextInfraRefactorClassNestingTransformer(
            {"TimeoutEnforcer": "FlextDispatcher"}, {}, {}
        )
    )
    code = transformed.code

    assert "class TimeoutEnforcer[T](BaseEnforcer, Generic[T], metaclass=Meta):" in code
    assert (
        "@decorator\n    class TimeoutEnforcer[T](BaseEnforcer, Generic[T], metaclass=Meta):"
        in code
    )
    assert '    """timeout docs"""' in code
    assert "class FlextDispatcher:" in code


def test_class_nesting_appends_to_existing_namespace_and_removes_pass() -> None:
    source = "class FlextDispatcher:\n    pass\n\nclass TimeoutEnforcer:\n    pass\n"

    transformed = cst.parse_module(source).visit(
        FlextInfraRefactorClassNestingTransformer(
            {"TimeoutEnforcer": "FlextDispatcher"}, {}, {}
        )
    )
    code = transformed.code

    assert "class FlextDispatcher:" in code
    assert "    class TimeoutEnforcer:" in code
    assert "class TimeoutEnforcer:\n    pass\n" not in code
    assert "class FlextDispatcher:\n    pass\n" not in code


def test_class_nesting_keeps_unmapped_top_level_classes() -> None:
    source = "class TimeoutEnforcer:\n    pass\n\nclass OtherClass:\n    pass\n"

    transformed = cst.parse_module(source).visit(
        FlextInfraRefactorClassNestingTransformer(
            {"TimeoutEnforcer": "FlextDispatcher"}, {}, {}
        )
    )
    code = transformed.code

    assert "class FlextDispatcher:" in code
    assert "    class TimeoutEnforcer:" in code
    assert "class OtherClass:\n    pass\n" in code
