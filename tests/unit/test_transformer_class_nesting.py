"""Unit tests for class nesting transformer behavior."""

from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraRefactorClassNestingTransformer, u


def _transform_source(tmp_path: Path, source: str) -> str:
    file_path = tmp_path / "class_nesting.py"
    file_path.write_text(source, encoding="utf-8")
    transformer = FlextInfraRefactorClassNestingTransformer(
        {"TimeoutEnforcer": "FlextDispatcher"},
        {},
        {},
    )
    rope_project = u.Infra.init_rope_project(tmp_path)
    try:
        resource = u.Infra.get_resource_from_path(rope_project, file_path)
        if resource is None:
            raise FileNotFoundError(file_path)
        transformed, _ = transformer.transform(rope_project, resource)
        return transformed
    finally:
        rope_project.close()


def test_class_nesting_moves_top_level_class_into_new_namespace(tmp_path: Path) -> None:
    source = '@decorator\nclass TimeoutEnforcer[T](BaseEnforcer, Generic[T], metaclass=Meta):\n    """timeout docs"""\n    value: T\n'
    code = _transform_source(tmp_path, source)
    assert "class TimeoutEnforcer[T](BaseEnforcer, Generic[T], metaclass=Meta):" in code
    assert (
        "@decorator\n    class TimeoutEnforcer[T](BaseEnforcer, Generic[T], metaclass=Meta):"
        in code
    )
    assert '    """timeout docs"""' in code
    assert "class FlextDispatcher:" in code


def test_class_nesting_appends_to_existing_namespace_and_removes_pass(
    tmp_path: Path,
) -> None:
    source = "class FlextDispatcher:\n    pass\n\nclass TimeoutEnforcer:\n    pass\n"
    code = _transform_source(tmp_path, source)
    assert "class FlextDispatcher:" in code
    assert "    class TimeoutEnforcer:" in code
    assert "class TimeoutEnforcer:\n    pass\n" not in code
    assert "class FlextDispatcher:\n    pass\n" not in code


def test_class_nesting_keeps_unmapped_top_level_classes(tmp_path: Path) -> None:
    source = "class TimeoutEnforcer:\n    pass\n\nclass OtherClass:\n    pass\n"
    code = _transform_source(tmp_path, source)
    assert "class FlextDispatcher:" in code
    assert "    class TimeoutEnforcer:" in code
    assert "class OtherClass:\n    pass\n" in code
