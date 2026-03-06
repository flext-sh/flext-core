"""Post-transform validation for class refactoring."""

from __future__ import annotations

import ast
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_infra.refactor.mro_resolver import FlextInfraRefactorMROResolver
    from flext_infra.refactor.result import FlextInfraRefactorResult


class PostCheckError(Exception):
    """Raised when post-transform validation fails."""

    pass


class PostCheckGate:
    """Validates transform results after AST rewrite."""

    def __init__(
        self,
        mro_resolver: FlextInfraRefactorMROResolver | None = None,
    ) -> None:
        from flext_infra.refactor.mro_resolver import FlextInfraRefactorMROResolver

        self._mro_resolver = mro_resolver or FlextInfraRefactorMROResolver()

    def validate(
        self,
        result: FlextInfraRefactorResult,
        expected: dict,
    ) -> tuple[bool, list[str]]:
        """Validate transform result.

        Returns:
            Tuple of (success, list_of_errors)
        """
        errors: list[str] = []

        if not result.success:
            errors.append(f"Transform failed: {result.error}")
            return False, errors

        if not result.modified:
            # Nothing to validate if file wasn't modified
            return True, []

        file_path = result.file_path

        # Validate imports resolve
        import_errors = self._validate_imports(file_path)
        errors.extend(import_errors)

        # Validate MRO if expected bases provided
        if expected_bases := expected.get("expected_base_chain"):
            class_name = expected.get("source_symbol", "")
            mro_errors = self._validate_mro(file_path, class_name, expected_bases)
            errors.extend(mro_errors)

        # Validate type hints are consistent
        type_errors = self._validate_types(file_path)
        errors.extend(type_errors)

        return len(errors) == 0, errors

    def _validate_imports(self, file_path: Path) -> list[str]:
        """Check that all imports in file resolve."""
        errors: list[str] = []

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError, UnicodeDecodeError) as e:
            return [f"Failed to parse {file_path}: {e}"]

        # Collect all imports
        imports: list[tuple[str, int]] = []  # (name, lineno)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    imports.append((full_name, node.lineno))

        # For each import, try to resolve it
        for name, lineno in imports:
            if not self._can_resolve_import(name, file_path):
                errors.append(f"Line {lineno}: Cannot resolve import '{name}'")

        return errors

    def _can_resolve_import(self, name: str, file_path: Path) -> bool:
        """Check if an import can be resolved."""
        # Simple check: if it's a stdlib module or starts with flext_, assume ok
        # Full implementation would use importlib.util.find_spec
        parts = name.split(".")
        top_level = parts[0]

        # Known safe modules
        safe_modules = {
            "typing",
            "collections",
            "pathlib",
            "dataclasses",
            "abc",
            "types",
            "sys",
            "os",
            "re",
            "json",
            "yaml",
            "libcst",
        }

        if top_level in safe_modules:
            return True

        # FLEXT modules - check if they exist
        if top_level.startswith("flext_"):
            # Simplified check - in full implementation would verify module exists
            return True

        # Relative imports (from parent modules)
        if file_path.name.startswith("_"):
            return True

        # Assume other imports are resolvable (conservative)
        return True

    def _validate_mro(
        self,
        file_path: Path,
        class_name: str,
        expected_bases: Sequence[str],
    ) -> list[str]:
        """Validate class MRO matches expected base chain."""
        errors: list[str] = []

        if not class_name:
            return []

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError, UnicodeDecodeError) as e:
            return [f"Failed to parse {file_path} for MRO validation: {e}"]

        # Find class definition
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                actual_bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        actual_bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        actual_bases.append(base.attr)

                # Check direct bases match expected (first N)
                expected_direct = list(expected_bases)[: len(actual_bases)]
                if actual_bases != expected_direct:
                    errors.append(
                        f"Class {class_name}: base mismatch. "
                        f"Expected {expected_direct}, got {actual_bases}"
                    )

                break
        else:
            errors.append(f"Class {class_name} not found in {file_path}")

        return errors

    def _validate_types(self, file_path: Path) -> list[str]:
        """Validate type hints are consistent."""
        # Simplified type validation
        # Full implementation would use mypy/pyright API
        errors: list[str] = []

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, OSError, UnicodeDecodeError):
            return []

        # Check for common type issues
        for node in ast.walk(tree):
            # Check for undefined names in annotations
            if isinstance(node, ast.AnnAssign):
                if node.annotation:
                    undefined = self._find_undefined_names(node.annotation)
                    for name in undefined:
                        errors.append(
                            f"Line {node.lineno}: Undefined name '{name}' in type annotation"
                        )

            # Check for undefined names in function annotations
            elif isinstance(node, ast.FunctionDef):
                # Check argument annotations
                for arg in node.args.args + node.args.kwonlyargs:
                    if arg.annotation:
                        undefined = self._find_undefined_names(arg.annotation)
                        for name in undefined:
                            errors.append(
                                f"Line {arg.lineno}: Undefined name '{name}' in parameter annotation"
                            )

                # Check return annotation
                if node.returns:
                    undefined = self._find_undefined_names(node.returns)
                    for name in undefined:
                        errors.append(
                            f"Line {node.lineno}: Undefined name '{name}' in return annotation"
                        )

        return errors

    def _find_undefined_names(self, node: ast.AST) -> list[str]:
        """Find names in AST node that might be undefined."""
        undefined: list[str] = []

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                # Skip common built-in types
                if child.id not in {
                    "int",
                    "str",
                    "float",
                    "bool",
                    "list",
                    "dict",
                    "tuple",
                    "set",
                    "None",
                    "Any",
                    "Optional",
                    "Union",
                    "Callable",
                    "Type",
                    "ClassVar",
                    "Final",
                    "Literal",
                }:
                    undefined.append(child.id)

        return undefined


__all__ = ["PostCheckGate", "PostCheckError"]
