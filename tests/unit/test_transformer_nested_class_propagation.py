"""Unit tests for nested class propagation transformer behavior."""

from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraNestedClassPropagationTransformer, u


def _transform_source(tmp_path: Path, source: str) -> str:
    file_path = tmp_path / "nested_class_propagation.py"
    file_path.write_text(source, encoding="utf-8")
    transformer = FlextInfraNestedClassPropagationTransformer({
        "TimeoutEnforcer": "FlextDispatcher.TimeoutEnforcer",
    })
    rope_project = u.Infra.init_rope_project(tmp_path)
    try:
        resource = u.Infra.get_resource_from_path(rope_project, file_path)
        if resource is None:
            raise FileNotFoundError(file_path)
        transformed, _ = transformer.transform(rope_project, resource)
        return transformed
    finally:
        rope_project.close()


def test_nested_class_propagation_updates_import_annotations_and_calls(
    tmp_path: Path,
) -> None:
    source = "from pkg import TimeoutEnforcer\n\nclass Child(TimeoutEnforcer):\n    pass\n\ndef validate(x: TimeoutEnforcer) -> bool:\n    if isinstance(x, TimeoutEnforcer):\n        y = TimeoutEnforcer()\n        return isinstance(y, pkg.TimeoutEnforcer)\n    return False\n"
    code = _transform_source(tmp_path, source)
    assert "from pkg import FlextDispatcher" in code
    assert "class Child(FlextDispatcher.TimeoutEnforcer):" in code
    assert "def validate(x: FlextDispatcher.TimeoutEnforcer) -> bool:" in code
    assert "if isinstance(x, FlextDispatcher.TimeoutEnforcer):" in code
    assert "y = FlextDispatcher.TimeoutEnforcer()" in code
    assert "isinstance(y, pkg.FlextDispatcher.TimeoutEnforcer)" in code


def test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage(
    tmp_path: Path,
) -> None:
    source = "from pkg import TimeoutEnforcer as TE\n\nvalue = TE()\n"
    code = _transform_source(tmp_path, source)
    assert "from pkg import FlextDispatcher as TE" in code
    assert "value = TE.TimeoutEnforcer()" in code
