"""Unit tests for nested class propagation transformer behavior."""

from __future__ import annotations

import libcst as cst
from libcst.metadata import MetadataWrapper

from flext_infra import FlextInfraNestedClassPropagationTransformer


def test_nested_class_propagation_updates_import_annotations_and_calls() -> None:
    source = "from pkg import TimeoutEnforcer\n\nclass Child(TimeoutEnforcer):\n    pass\n\ndef validate(x: TimeoutEnforcer) -> bool:\n    if isinstance(x, TimeoutEnforcer):\n        y = TimeoutEnforcer()\n        return isinstance(y, pkg.TimeoutEnforcer)\n    return False\n"
    transformed = MetadataWrapper(cst.parse_module(source)).visit(
        FlextInfraNestedClassPropagationTransformer({
            "TimeoutEnforcer": "FlextDispatcher.TimeoutEnforcer",
        }),
    )
    code = transformed.code
    assert "from pkg import FlextDispatcher" in code
    assert "class Child(FlextDispatcher.TimeoutEnforcer):" in code
    assert "def validate(x: FlextDispatcher.TimeoutEnforcer) -> bool:" in code
    assert "if isinstance(x, FlextDispatcher.TimeoutEnforcer):" in code
    assert "y = FlextDispatcher.TimeoutEnforcer()" in code
    assert "isinstance(y, pkg.FlextDispatcher.TimeoutEnforcer)" in code


def test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage() -> None:
    source = "from pkg import TimeoutEnforcer as TE\n\nvalue = TE()\n"
    transformed = MetadataWrapper(cst.parse_module(source)).visit(
        FlextInfraNestedClassPropagationTransformer({
            "TimeoutEnforcer": "FlextDispatcher.TimeoutEnforcer",
        }),
    )
    code = transformed.code
    assert "from pkg import FlextDispatcher as TE" in code
    assert "value = TE.TimeoutEnforcer()" in code
