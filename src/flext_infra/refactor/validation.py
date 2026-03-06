from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from flext_infra.refactor.result import FlextInfraRefactorResult


class PostCheckGate:
    class _PostCheckExpected(TypedDict, total=False):
        source_symbol: str
        expected_base_chain: list[str]

    def validate(
        self,
        result: FlextInfraRefactorResult,
        expected: _PostCheckExpected,
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if not result.success:
            if result.error:
                return False, [result.error]
            return False, ["transform_failed"]
        if not result.modified:
            return True, []

        file_path = result.file_path
        errors.extend(self._validate_imports(file_path))

        source_symbol = expected.get("source_symbol", "")
        expected_chain = expected.get("expected_base_chain", [])
        if source_symbol:
            errors.extend(self._validate_mro(file_path, source_symbol, expected_chain))

        errors.extend(self._validate_types(file_path))
        return len(errors) == 0, errors

    def _validate_imports(self, file_path: Path) -> list[str]:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            return [f"parse_error:{exc}"]

        unresolved: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is None
                and node.level == 0
            ):
                unresolved.append(f"line_{node.lineno}:invalid_import_from")
        return unresolved

    def _validate_mro(
        self,
        file_path: Path,
        class_name: str,
        expected_bases: Sequence[str],
    ) -> list[str]:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            return [f"mro_parse_error:{exc}"]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                actual = [self._base_name(base) for base in node.bases]
                actual_clean = [name for name in actual if name]
                expected_prefix = list(expected_bases)[: len(actual_clean)]
                if actual_clean != expected_prefix:
                    return [
                        f"mro_mismatch:{class_name}:expected={expected_prefix}:actual={actual_clean}"
                    ]
                return []
        return [f"class_not_found:{class_name}"]

    def _validate_types(self, file_path: Path) -> list[str]:
        try:
            source = file_path.read_text(encoding="utf-8")
            ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            return [f"type_parse_error:{exc}"]
        return []

    def _base_name(self, base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        if isinstance(base, ast.Subscript):
            return self._base_name(base.value)
        return ""


__all__ = ["PostCheckGate"]
