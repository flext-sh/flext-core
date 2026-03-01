"""Namespace validation service.

AST-based validator enforcing namespace rules 0-2 for flext projects.
Detection-only — does not auto-fix any files.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from flext_core import r

from flext_infra import m

__all__ = ["FlextInfraNamespaceValidator"]


class FlextInfraNamespaceValidator:
    """AST-based namespace validator for flext projects (Rules 0-2).

    Validates that each module follows the one-namespace-class-per-file
    convention, constants are centralized in ``constants.py``, and type
    definitions are centralized in ``typings.py``.
    """

    _EXEMPT_FILENAMES: Final[frozenset[str]] = frozenset({
        "__init__.py",
        "conftest.py",
        "__main__.py",
    })
    _EXEMPT_PREFIXES: Final[frozenset[str]] = frozenset({"test_", "_"})
    _ALIAS_NAMES: Final[frozenset[str]] = frozenset({
        "c",
        "t",
        "m",
        "p",
        "u",
        "r",
        "d",
        "e",
        "h",
        "s",
        "x",
        "tc",
    })
    _DUNDER_ALLOWED: Final[frozenset[str]] = frozenset({"__all__", "__version__"})
    _TYPEVAR_CALLABLES: Final[frozenset[str]] = frozenset({
        "TypeVar",
        "ParamSpec",
        "TypeVarTuple",
    })
    _ENUM_BASES: Final[frozenset[str]] = frozenset({"StrEnum", "Enum", "IntEnum"})
    _COLLECTION_CALLS: Final[frozenset[str]] = frozenset({
        "frozenset",
        "tuple",
        "dict",
        "list",
    })

    def validate(
        self, project_root: Path, *, scan_tests: bool = False
    ) -> r[m.ValidationReport]:
        """Validate namespace rules 0-2 for all discovered Python files.

        Args:
            project_root: Root directory of the project to validate.
            scan_tests: Whether to also scan the ``tests/`` directory.

        Returns:
            r with ValidationReport indicating namespace compliance.

        """
        try:
            files = self._discover_files(project_root, scan_tests=scan_tests)
            prefix = self._derive_prefix(project_root)
            violations: list[str] = []

            for filepath in files:
                tree = self._parse_file(filepath)
                if tree is None:
                    continue
                rel = filepath.relative_to(project_root)
                violations.extend(self._check_rule_0(tree, rel, prefix))
                violations.extend(self._check_rule_1(tree, rel))
                violations.extend(self._check_rule_2(tree, rel))

            passed = len(violations) == 0
            summary = (
                f"namespace validation passed ({len(files)} files checked)"
                if passed
                else f"{len(violations)} namespace violation(s) found ({len(files)} files checked)"
            )

            return r[m.ValidationReport].ok(
                m.ValidationReport(
                    passed=passed, violations=violations, summary=summary
                ),
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return r[m.ValidationReport].fail(f"Namespace validation failed: {exc}")

    # -- file discovery -------------------------------------------------------

    def _discover_files(self, root: Path, *, scan_tests: bool) -> list[Path]:
        """Walk ``src/`` (and optionally ``tests/``) for non-exempt .py files."""
        result: list[Path] = []
        dirs_to_scan = [root / "src"]
        if scan_tests:
            dirs_to_scan.append(root / "tests")

        for base_dir in dirs_to_scan:
            if not base_dir.is_dir():
                continue
            result.extend(
                py_file
                for py_file in sorted(base_dir.rglob("*.py"))
                if not self._is_exempt_file(py_file)
            )

        return sorted(result)

    def _is_exempt_file(self, filepath: Path) -> bool:
        """Check whether a file should be skipped from validation."""
        name = filepath.name
        if name in self._EXEMPT_FILENAMES:
            return True
        return any(name.startswith(pfx) for pfx in self._EXEMPT_PREFIXES)

    @staticmethod
    def _derive_prefix(project_root: Path) -> str:
        """Derive the expected class name prefix from the package directory.

        Finds the first package under ``src/`` and converts its name to
        PascalCase: ``flext_infra`` → ``FlextInfra``.
        """
        src_dir = project_root / "src"
        if not src_dir.is_dir():
            return ""
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                return "".join(part.title() for part in child.name.split("_"))
        return ""

    @staticmethod
    def derive_prefix(project_root: Path) -> str:
        """Public wrapper for deriving the class name prefix from a project.

        Args:
            project_root: Root directory of the project.

        Returns:
            The PascalCase prefix derived from the package name.

        """
        return FlextInfraNamespaceValidator._derive_prefix(project_root)

    # -- AST helpers ----------------------------------------------------------

    def _parse_file(self, path: Path) -> ast.Module | None:
        """Parse a Python file into an AST, returning None on failure."""
        try:
            source = path.read_text(encoding="utf-8")
            return ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            return None

    def _is_allowed_module_level(self, node: ast.stmt, filepath: Path) -> bool:
        """Check whether a non-class top-level statement is allowed."""
        # Imports are always allowed
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True

        # Module docstring (string expression)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return True

        # __all__ / __version__ assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in self._DUNDER_ALLOWED:
                    return True

        # Single/two-letter alias assignments (c = FlextConstants, tc = ...)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in self._ALIAS_NAMES:
                return True

        # TypeVar / ParamSpec / TypeVarTuple calls — only in typings.py
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            func_name = self._get_call_name(func)
            if func_name in self._TYPEVAR_CALLABLES:
                return filepath.name == "typings.py"

        # PEP 695 TypeAlias — only in typings.py
        if isinstance(node, ast.TypeAlias):
            return filepath.name == "typings.py"

        # TypeAlias annotated assignments — only in typings.py
        if isinstance(node, ast.AnnAssign) and self._annotation_contains(
            node.annotation, "TypeAlias"
        ):
            return filepath.name == "typings.py"

        # _private Final constants — only in constants.py
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id.startswith("_")
            and self._annotation_contains(node.annotation, "Final")
        ):
            return filepath.name == "constants.py"

        return False

    # -- Rule implementations -------------------------------------------------

    def _check_rule_0(self, tree: ast.Module, filepath: Path, prefix: str) -> list[str]:
        """Rule 0 — One namespace class per module.

        Checks that each module has exactly one top-level class whose name
        starts with the project prefix, and that all other top-level
        statements are on the allowlist.
        """
        violations: list[str] = []
        seq = 0

        outer_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        class_count = len(outer_classes)

        if class_count != 1:
            seq += 1
            violations.append(
                f"[NS-000-{seq:03d}] {filepath}:{outer_classes[0].lineno if outer_classes else 1}"
                f" — Multiple outer classes found (expected 1, got {class_count})"
                if class_count > 1
                else f"[NS-000-{seq:03d}] {filepath}:1 — No outer class found (expected 1, got 0)",
            )

        # Check prefix on outer class(es)
        for cls in outer_classes:
            if prefix and not cls.name.startswith(prefix):
                seq += 1
                violations.append(
                    f"[NS-000-{seq:03d}] {filepath}:{cls.lineno}"
                    f" — Class '{cls.name}' does not start with prefix '{prefix}'",
                )

        # Check non-class top-level statements
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                continue
            if not self._is_allowed_module_level(node, filepath):
                seq += 1
                lineno = getattr(node, "lineno", 0)
                violations.append(
                    f"[NS-000-{seq:03d}] {filepath}:{lineno}"
                    f" — Disallowed top-level statement: {type(node).__name__}",
                )

        return violations

    def _check_rule_1(self, tree: ast.Module, filepath: Path) -> list[str]:
        """Rule 1 — Constants centralization.

        In ``constants.py``: outer class must inherit from a Constants base,
        and inner classes must not contain methods.

        In other modules: no loose ``Final`` constants, no loose Enum classes,
        no loose collection constant assignments.
        """
        violations: list[str] = []
        seq = 0
        is_constants = filepath.name == "constants.py"

        if is_constants:
            outer_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            for cls in outer_classes:
                if not any(self._base_contains(b, "Constants") for b in cls.bases):
                    seq += 1
                    violations.append(
                        f"[NS-001-{seq:03d}] {filepath}:{cls.lineno}"
                        f" — Constants class '{cls.name}' must inherit from a Constants base",
                    )
                # Check for methods in inner class bodies
                for inner_node in ast.walk(cls):
                    if (
                        isinstance(inner_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and inner_node is not cls
                    ):
                        seq += 1
                        violations.append(
                            f"[NS-001-{seq:03d}] {filepath}:{inner_node.lineno}"
                            f" — Method '{inner_node.name}' found in Constants class",
                        )
        else:
            # Non-constants modules
            for node in tree.body:
                # Loose Final constants
                if isinstance(node, ast.AnnAssign) and self._annotation_contains(
                    node.annotation, "Final"
                ):
                    target_name = self._get_target_name(node.target)
                    # Allow _private Finals (they are checked by Rule 0 for constants.py only)
                    if target_name and not target_name.startswith("_"):
                        seq += 1
                        violations.append(
                            f"[NS-001-{seq:03d}] {filepath}:{node.lineno}"
                            f" — Loose Final constant '{target_name}' belongs in constants.py",
                        )

                # Loose collection constants
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    func_name = self._get_call_name(node.value.func)
                    if func_name in self._COLLECTION_CALLS:
                        target_name = self._get_assign_target_name(node)
                        if (
                            target_name
                            and target_name not in self._DUNDER_ALLOWED
                            and target_name not in self._ALIAS_NAMES
                        ):
                            seq += 1
                            violations.append(
                                f"[NS-001-{seq:03d}] {filepath}:{node.lineno}"
                                f" — Loose collection constant '{target_name}' belongs in constants.py",
                            )

            # Loose Enum classes (inside outer class body or top-level)
            outer_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            for cls in outer_classes:
                for inner in cls.body:
                    if isinstance(inner, ast.ClassDef) and any(
                        self._base_contains(b, base)
                        for b in inner.bases
                        for base in self._ENUM_BASES
                    ):
                        seq += 1
                        violations.append(
                            f"[NS-001-{seq:03d}] {filepath}:{inner.lineno}"
                            f" — Loose Enum '{inner.name}' belongs in constants.py",
                        )

        return violations

    def _check_rule_2(self, tree: ast.Module, filepath: Path) -> list[str]:
        """Rule 2 — Types centralization.

        In ``typings.py``: outer class must inherit from a Types base,
        and inner classes must not inherit from BaseModel or Protocol.

        In other modules: no TypeVar/ParamSpec/TypeVarTuple calls,
        no TypeAlias annotations, no PEP 695 TypeAlias statements.
        """
        violations: list[str] = []
        seq = 0
        is_typings = filepath.name == "typings.py"

        if is_typings:
            outer_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            for cls in outer_classes:
                if not any(self._base_contains(b, "Types") for b in cls.bases):
                    seq += 1
                    violations.append(
                        f"[NS-002-{seq:03d}] {filepath}:{cls.lineno}"
                        f" — Types class '{cls.name}' must inherit from a Types base",
                    )
                # Inner classes must not inherit from BaseModel or Protocol
                for inner in cls.body:
                    if isinstance(inner, ast.ClassDef):
                        for base in inner.bases:
                            if self._base_contains(
                                base, "BaseModel"
                            ) or self._base_contains(base, "Protocol"):
                                seq += 1
                                violations.append(
                                    f"[NS-002-{seq:03d}] {filepath}:{inner.lineno}"
                                    f" — Inner class '{inner.name}' in Types must not inherit from BaseModel/Protocol",
                                )
        else:
            # Non-typings modules
            for node in tree.body:
                # TypeVar / ParamSpec / TypeVarTuple calls
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    func_name = self._get_call_name(node.value.func)
                    if func_name in self._TYPEVAR_CALLABLES:
                        target_name = self._get_assign_target_name(node)
                        seq += 1
                        violations.append(
                            f"[NS-002-{seq:03d}] {filepath}:{node.lineno}"
                            f" — TypeVar '{target_name}' belongs in typings.py",
                        )

                # TypeAlias annotated assignments
                if isinstance(node, ast.AnnAssign) and self._annotation_contains(
                    node.annotation, "TypeAlias"
                ):
                    target_name = self._get_target_name(node.target)
                    seq += 1
                    violations.append(
                        f"[NS-002-{seq:03d}] {filepath}:{node.lineno}"
                        f" — TypeAlias '{target_name}' belongs in typings.py",
                    )

                # PEP 695 TypeAlias
                if isinstance(node, ast.TypeAlias):
                    name = getattr(node, "name", None)
                    name_str = getattr(name, "id", str(name)) if name else "unknown"
                    seq += 1
                    violations.append(
                        f"[NS-002-{seq:03d}] {filepath}:{node.lineno}"
                        f" — PEP 695 TypeAlias '{name_str}' belongs in typings.py",
                    )

        return violations

    # -- annotation / base helpers --------------------------------------------

    @staticmethod
    def _get_call_name(func: ast.expr) -> str:
        """Extract the function name from a Call node's func attribute."""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""

    @staticmethod
    def _get_target_name(target: ast.expr) -> str:
        """Extract the name from an assignment target."""
        if isinstance(target, ast.Name):
            return target.id
        return ""

    @staticmethod
    def _get_assign_target_name(node: ast.Assign) -> str:
        """Extract the first target name from an Assign node."""
        if node.targets and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return ""

    @staticmethod
    def _annotation_contains(annotation: ast.expr | None, name: str) -> bool:
        """Check whether an annotation AST node references a given name."""
        if annotation is None:
            return False
        if isinstance(annotation, ast.Name) and annotation.id == name:
            return True
        if isinstance(annotation, ast.Attribute) and annotation.attr == name:
            return True
        # Handle Subscript like Final[int]
        if isinstance(annotation, ast.Subscript):
            return FlextInfraNamespaceValidator._annotation_contains(
                annotation.value, name
            )
        return False

    @staticmethod
    def _base_contains(base: ast.expr, name: str) -> bool:
        """Check whether a class base AST node references a given name."""
        if isinstance(base, ast.Name) and base.id == name:
            return True
        return isinstance(base, ast.Attribute) and base.attr == name
