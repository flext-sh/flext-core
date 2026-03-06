"""Post-transformation validation gates for refactor results."""

from __future__ import annotations

import ast
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from flext_infra import FlextInfraCommandRunner, c, m, p, t


def _string_list(value: t.ContainerValue | None) -> list[str]:
    try:
        return TypeAdapter(list[str]).validate_python(value)
    except ValidationError:
        return []


class PostCheckGate:
    """Validate refactor results against policy expectations."""

    def __init__(self) -> None:
        """Initialize gate with a command runner."""
        self._runner: p.Infra.CommandRunner = FlextInfraCommandRunner()

    def validate(
        self,
        result: m.Infra.Refactor.Result,
        expected: Mapping[str, t.ContainerValue],
    ) -> tuple[bool, list[str]]:
        """Validate a refactor result against expected post-checks and gates."""
        errors: list[str] = []
        if not result.success:
            if result.error:
                return False, [result.error]
            return False, ["transform_failed"]
        if not result.modified:
            return True, []

        file_path = result.file_path
        post_checks = _string_list(expected.get(c.Infra.ReportKeys.POST_CHECKS))
        quality_gates = _string_list(expected.get("quality_gates"))

        if self._check_enabled("imports_resolve", post_checks):
            errors.extend(self._validate_imports(file_path))

        source_symbol_raw = expected.get(c.Infra.ReportKeys.SOURCE_SYMBOL, "")
        source_symbol = source_symbol_raw if isinstance(source_symbol_raw, str) else ""
        expected_chain = _string_list(expected.get("expected_base_chain"))
        if (
            source_symbol
            and expected_chain
            and self._check_enabled("mro_valid", post_checks)
        ):
            errors.extend(self._validate_mro(file_path, source_symbol, expected_chain))

        if self._check_enabled("lsp_diagnostics_clean", quality_gates):
            errors.extend(self._validate_types(file_path))

        return len(errors) == 0, errors

    def _check_enabled(self, check_name: str, checks: list[str]) -> bool:
        return check_name in checks

    def _validate_imports(self, file_path: Path) -> list[str]:
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            return [f"parse_error:{exc}"]

        unresolved: list[str] = [
            f"line_{node.lineno}:invalid_import_from"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is None
            and node.level == 0
        ]
        return unresolved

    def _validate_mro(
        self,
        file_path: Path,
        class_name: str,
        expected_bases: Sequence[str],
    ) -> list[str]:
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
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
        """Check that the file compiles without syntax errors."""
        result = self._runner.capture(
            [sys.executable, "-m", "py_compile", str(file_path)],
        )
        if result.is_failure:
            output = (result.error or "").strip()
            return [f"lsp_diagnostics_clean_failed:{output}"]
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
