"""Tests for FlextInfraNamespaceValidator."""

from __future__ import annotations

import re
from pathlib import Path

from flext_infra import m
from flext_infra.core.namespace_validator import FlextInfraNamespaceValidator

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "namespace_validator"


def _read_fixture(name: str) -> str:
    return (_FIXTURES_DIR / name).read_text(encoding="utf-8")


def _make_project_with_module(
    tmp_path: Path,
    *,
    module_source: str,
    module_name: str,
) -> Path:
    project_root = tmp_path / "project"
    package_dir = project_root / "src" / "flext_test"
    package_dir.mkdir(parents=True)
    _ = (package_dir / "__init__.py").write_text("", encoding="utf-8")
    _ = (package_dir / module_name).write_text(module_source, encoding="utf-8")
    return project_root


class TestFlextInfraNamespaceValidator:
    """Test suite for namespace validator rules 0-2."""

    def test_rule0_valid_module_passes(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule0_valid.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert result.value.passed
        assert result.value.violations == []

    def test_rule0_multiple_classes_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule0_multiple_classes.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any("Multiple outer classes found" in v for v in result.value.violations)

    def test_rule0_no_class_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule0_no_class.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any("No outer class found" in v for v in result.value.violations)

    def test_rule0_wrong_prefix_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule0_wrong_prefix.py"),
            module_name="constants.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any(
            "does not start with prefix 'FlextTest'" in v
            for v in result.value.violations
        )

    def test_rule0_loose_items_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule0_loose_items.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any(
            "Disallowed top-level statement: FunctionDef" in v
            for v in result.value.violations
        )

    def test_rule1_valid_constants_passes(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        module_source = (
            "from __future__ import annotations\n"
            "\n"
            "class FlextTestConstants(Constants):\n"
            "    class Limits:\n"
            "        MAX_RETRIES = 3\n"
        )
        root = _make_project_with_module(
            tmp_path,
            module_source=module_source,
            module_name="constants.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert result.value.passed

    def test_rule1_loose_constant_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule1_loose_constant.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any("Loose Final constant" in v for v in result.value.violations)

    def test_rule1_loose_enum_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule1_loose_enum.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any("Multiple outer classes found" in v for v in result.value.violations)

    def test_rule1_method_in_constants_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule1_method_in_constants.py"),
            module_name="constants.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any(
            "Method 'create_name' found in Constants class" in v
            for v in result.value.violations
        )

    def test_rule1_magic_number_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule1_magic_number.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any("Loose collection constant" in v for v in result.value.violations)

    def test_rule2_valid_types_passes(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        module_source = (
            "from __future__ import annotations\n"
            "from typing import TypeVar\n"
            "\n"
            'T = TypeVar("T")\n'
            "\n"
            "class FlextTestTypes(Types):\n"
            "    pass\n"
        )
        root = _make_project_with_module(
            tmp_path,
            module_source=module_source,
            module_name="typings.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert result.value.passed

    def test_rule2_typevar_in_class_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule2_typevar_in_class.py"),
            module_name="typings.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any(
            "must inherit from a Types base" in v for v in result.value.violations
        )

    def test_rule2_typevar_wrong_module_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule2_typevar_wrong_module.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any(
            "TypeVar 'T' belongs in typings.py" in v for v in result.value.violations
        )

    def test_rule2_composite_type_loose_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule2_composite_type_loose.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any(
            "TypeAlias 'JsonValue' belongs in typings.py" in v
            for v in result.value.violations
        )

    def test_rule2_protocol_in_types_detected(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule2_protocol_in_types.py"),
            module_name="typings.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert not result.value.passed
        assert any("Inner class 'Serializable'" in v for v in result.value.violations)

    def test_exempt_files_skipped(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        project_root = tmp_path / "project"
        package_dir = project_root / "src" / "flext_test"
        package_dir.mkdir(parents=True)
        _ = (package_dir / "__init__.py").write_text(
            _read_fixture("rule0_no_class.py"), encoding="utf-8"
        )
        _ = (package_dir / "test_rule.py").write_text(
            _read_fixture("rule0_no_class.py"), encoding="utf-8"
        )
        _ = (package_dir / "_private.py").write_text(
            _read_fixture("rule0_no_class.py"), encoding="utf-8"
        )

        result = validator.validate(project_root)

        assert result.is_success
        assert result.value.passed
        assert result.value.violations == []
        assert "0 files checked" in result.value.summary

    def test_validate_returns_report(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule0_valid.py"),
            module_name="constants.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert isinstance(result.value, m.ValidationReport)
        assert "files checked" in result.value.summary

    def test_violation_message_format(self, tmp_path: Path) -> None:
        validator = FlextInfraNamespaceValidator()
        root = _make_project_with_module(
            tmp_path,
            module_source=_read_fixture("rule0_no_class.py"),
            module_name="models.py",
        )

        result = validator.validate(root)

        assert result.is_success
        assert result.value.violations
        first = result.value.violations[0]
        assert re.search(r"^\[NS-\d{3}-\d{3}\] .+\.py:\d+ — .+$", first) is not None
