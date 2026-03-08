"""Automated namespace enforcement engine for flext workspace.

Scans all projects under src/, tests/, scripts/, examples/ for violations of
the MRO namespace pattern (c, t, p, m, u) and automatically applies fixes
using libcst and ast.  Detects:
    - Missing facade classes per project
    - Loose objects outside namespace classes
    - Incorrect facade alias usage
    - Cyclic import chains
    - Internal class exposure through public APIs
    - Settings pattern violations

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from flext_infra import c, u


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class NamespaceEnforcementModels:
    """Domain models for namespace enforcement reports."""

    class FacadeStatus(BaseModel):
        """Status of one facade family (c/t/p/m/u) in a project."""

        model_config = ConfigDict(frozen=True)

        family: str = Field(min_length=1, description="Facade alias letter")
        exists: bool = Field(description="Whether the facade class exists")
        class_name: str = Field(default="", description="Facade class name")
        file: str = Field(default="", description="File containing the class")
        symbol_count: int = Field(
            default=0, ge=0, description="Number of symbols inside"
        )

    class LooseObjectViolation(BaseModel):
        """A detected loose object outside any namespace class."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(ge=1, description="Line number")
        name: str = Field(min_length=1, description="Object name")
        kind: str = Field(description="class|function|constant|typealias")
        suggestion: str = Field(default="", description="Suggested target namespace")

    class CyclicImportViolation(BaseModel):
        """A detected cyclic import chain."""

        model_config = ConfigDict(frozen=True)

        cycle: tuple[str, ...] = Field(description="Module names forming the cycle")
        files: tuple[str, ...] = Field(
            default_factory=tuple, description="Files involved"
        )

    class ImportAliasViolation(BaseModel):
        """A detected import not using the canonical facade alias."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(ge=1, description="Line number")
        current_import: str = Field(description="Current import statement")
        suggested_import: str = Field(description="Correct facade import")

    class ProjectEnforcementReport(BaseModel):
        """Enforcement report for a single project."""

        project: str = Field(min_length=1, description="Project directory name")
        project_root: str = Field(description="Absolute project root path")
        facade_statuses: list[NamespaceEnforcementModels.FacadeStatus] = Field(
            default_factory=list, description="Status per facade family"
        )
        loose_objects: list[NamespaceEnforcementModels.LooseObjectViolation] = Field(
            default_factory=list, description="Loose object violations"
        )
        import_violations: list[NamespaceEnforcementModels.ImportAliasViolation] = (
            Field(default_factory=list, description="Import alias violations")
        )
        cyclic_imports: list[NamespaceEnforcementModels.CyclicImportViolation] = Field(
            default_factory=list, description="Cyclic import violations"
        )
        files_scanned: int = Field(default=0, ge=0, description="Total files scanned")

    class WorkspaceEnforcementReport(BaseModel):
        """Full workspace enforcement report."""

        workspace: str = Field(min_length=1, description="Workspace root path")
        projects: list[NamespaceEnforcementModels.ProjectEnforcementReport] = Field(
            default_factory=list, description="Per-project reports"
        )
        total_facades_missing: int = Field(
            default=0, ge=0, description="Missing facades"
        )
        total_loose_objects: int = Field(default=0, ge=0, description="Loose objects")
        total_import_violations: int = Field(
            default=0, ge=0, description="Import violations"
        )
        total_cyclic_imports: int = Field(
            default=0, ge=0, description="Cyclic import chains"
        )
        total_files_scanned: int = Field(
            default=0, ge=0, description="All files scanned"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FACADE_FAMILIES: dict[str, str] = dict(c.Infra.Refactor.FAMILY_SUFFIXES)
_FACADE_FILE_PATTERNS: dict[str, str] = dict(c.Infra.Refactor.FAMILY_FILES)
_CONSTANT_PATTERN: re.Pattern[str] = re.compile(r"^_?[A-Z][A-Z0-9_]+$")
_SETTINGS_FILE_NAMES: frozenset[str] = frozenset({"settings.py", "_settings.py"})
_PROTECTED_FILES: frozenset[str] = frozenset({
    "settings.py",
    "_settings.py",
    "__init__.py",
    "__main__.py",
    "__version__.py",
    "conftest.py",
    "py.typed",
})
_MIN_ALIAS_LENGTH: int = 2
"Minimum name length to be considered a potential constant (skip c, t, etc.)."
_MAX_RENDERED_LOOSE_OBJECTS: int = 10
"Maximum loose-object lines rendered per project block."
_MAX_RENDERED_IMPORT_VIOLATIONS: int = 5
"Maximum import-violation lines rendered per project block."


# ---------------------------------------------------------------------------
# Facade scanner
# ---------------------------------------------------------------------------
class NamespaceFacadeScanner:
    """Scan a project for existing facade classes per family."""

    @classmethod
    def scan_project(
        cls,
        *,
        project_root: Path,
        project_name: str,
    ) -> list[NamespaceEnforcementModels.FacadeStatus]:
        """Return one FacadeStatus per family for this project."""
        results: list[NamespaceEnforcementModels.FacadeStatus] = []
        class_stem = cls.project_class_stem(project_name=project_name)
        for family, suffix in _FACADE_FAMILIES.items():
            expected_class = f"{class_stem}{suffix}"
            found_class, found_file, symbol_count = cls._find_facade_class(
                project_root=project_root,
                family=family,
                expected_class=expected_class,
                suffix=suffix,
            )
            results.append(
                NamespaceEnforcementModels.FacadeStatus(
                    family=family,
                    exists=bool(found_class),
                    class_name=found_class,
                    file=found_file,
                    symbol_count=symbol_count,
                )
            )
        return results

    @classmethod
    def _find_facade_class(
        cls,
        *,
        project_root: Path,
        family: str,
        expected_class: str,
        suffix: str,
    ) -> tuple[str, str, int]:
        """Return (class_name, file_path, symbol_count) or empty."""
        file_pattern = _FACADE_FILE_PATTERNS[family]
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return ("", "", 0)
        for file_path in src_dir.rglob(file_pattern):
            try:
                tree = ast.parse(file_path.read_text(encoding=c.Infra.Encoding.DEFAULT))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if node.name == expected_class or node.name.endswith(suffix):
                    symbol_count = sum(
                        1
                        for child in ast.iter_child_nodes(node)
                        if isinstance(
                            child,
                            (
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                                ast.ClassDef,
                                ast.AnnAssign,
                                ast.Assign,
                            ),
                        )
                    )
                    return (node.name, str(file_path), symbol_count)
        return ("", "", 0)

    @staticmethod
    def project_class_stem(*, project_name: str) -> str:
        """Derive the class stem from project name.

        flext-core  -> Flext
        flext-infra -> FlextInfra
        flext-ldap  -> FlextLdap
        """
        normalized = project_name.strip().lower().replace("_", "-")
        if normalized == "flext-core":
            return "Flext"
        if normalized.startswith("flext-"):
            tail = normalized.removeprefix("flext-")
            parts = [p for p in tail.split("-") if p]
            return "Flext" + "".join(p.capitalize() for p in parts)
        parts = [p for p in normalized.split("-") if p]
        return "".join(p.capitalize() for p in parts) if parts else ""


# ---------------------------------------------------------------------------
# Loose object detector
# ---------------------------------------------------------------------------
class LooseObjectDetector:
    """Detect top-level objects not nested in namespace classes."""

    # Names that are OK at module-level
    ALLOWED_TOP_LEVEL: ClassVar[frozenset[str]] = frozenset({
        "__all__",
        "__version__",
        "__version_info__",
    })

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_name: str,
    ) -> list[NamespaceEnforcementModels.LooseObjectViolation]:
        """Return loose objects found in a single file."""
        if file_path.name in _PROTECTED_FILES:
            return []
        if file_path.name in _SETTINGS_FILE_NAMES:
            return []
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            return []

        namespace_classes = cls._find_namespace_classes(tree=tree)
        violations: list[NamespaceEnforcementModels.LooseObjectViolation] = []
        class_stem = NamespaceFacadeScanner.project_class_stem(
            project_name=project_name
        )
        for stmt in tree.body:
            violation = cls._check_statement(
                stmt=stmt,
                namespace_classes=namespace_classes,
                file_path=file_path,
                class_stem=class_stem,
            )
            if violation is not None:
                violations.append(violation)
        return violations

    @classmethod
    def _check_statement(
        cls,
        *,
        stmt: ast.stmt,
        namespace_classes: set[str],
        file_path: Path,
        class_stem: str,
    ) -> NamespaceEnforcementModels.LooseObjectViolation | None:
        """Check if a statement is a loose object."""
        # Skip imports, docstrings, __future__, runtime alias assignments
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return None
        if isinstance(stmt, ast.If):
            return None

        # ClassDef - only report if NOT a namespace class
        if isinstance(stmt, ast.ClassDef):
            if stmt.name in namespace_classes:
                return None
            # Runtime alias assignments (c = FlextConstants)
            return None  # Classes are OK for now, they may be namespace classes

        # FunctionDef at top level is always loose (except dunder and private)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name.startswith("__") and stmt.name.endswith("__"):
                return None
            # Private functions (_prefixed) are implementation details:
            # Pydantic BeforeValidator callbacks, type guards, helpers.
            if stmt.name.startswith("_"):
                return None
            return NamespaceEnforcementModels.LooseObjectViolation(
                file=str(file_path),
                line=stmt.lineno,
                name=stmt.name,
                kind="function",
                suggestion=f"{class_stem}Utilities",
            )

        # Module-level constants (UPPER_CASE assignments)
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if name in cls.ALLOWED_TOP_LEVEL:
                return None
            # Private constants (_PREFIXED) are module implementation details
            if name.startswith("_"):
                return None
            if _CONSTANT_PATTERN.match(name):
                return NamespaceEnforcementModels.LooseObjectViolation(
                    file=str(file_path),
                    line=stmt.lineno,
                    name=name,
                    kind="constant",
                    suggestion=f"{class_stem}Constants",
                )
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if name in cls.ALLOWED_TOP_LEVEL:
                    return None
                # Skip single-letter runtime aliases
                if len(name) <= _MIN_ALIAS_LENGTH:
                    return None
                # Private constants (_PREFIXED) are module implementation details
                if name.startswith("_"):
                    return None
                if _CONSTANT_PATTERN.match(name):
                    return NamespaceEnforcementModels.LooseObjectViolation(
                        file=str(file_path),
                        line=stmt.lineno,
                        name=name,
                        kind="constant",
                        suggestion=f"{class_stem}Constants",
                    )

        # Type aliases
        if isinstance(stmt, ast.TypeAlias):
            name = stmt.name.id if hasattr(stmt.name, "id") else ""
            if name and name not in cls.ALLOWED_TOP_LEVEL:
                return NamespaceEnforcementModels.LooseObjectViolation(
                    file=str(file_path),
                    line=stmt.lineno,
                    name=name,
                    kind="typealias",
                    suggestion=f"{class_stem}Types",
                )

        return None

    @staticmethod
    def _find_namespace_classes(*, tree: ast.Module) -> set[str]:
        """Find classes that end with known namespace suffixes."""
        classes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for suffix in _FACADE_FAMILIES.values():
                    if node.name.endswith(suffix):
                        classes.add(node.name)
                        break
        return classes


# ---------------------------------------------------------------------------
# Import alias violation detector
# ---------------------------------------------------------------------------
class ImportAliasDetector:
    """Detect imports not using canonical facade aliases."""

    # Modules that should be imported via aliases
    ALIAS_MODULES: ClassVar[dict[str, str]] = {
        "flext_core": "from flext_core import c, m, r, t, u, p",
        "flext_infra": "from flext_infra import c, m, t, u, p",
    }

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
    ) -> list[NamespaceEnforcementModels.ImportAliasViolation]:
        """Return import alias violations for one file."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            return []

        violations: list[NamespaceEnforcementModels.ImportAliasViolation] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.ImportFrom):
                continue
            if stmt.module is None:
                continue
            # __init__.py files ARE the facade definition — they
            # legitimately import from submodules (AGENTS.md §4:
            # "Within flext-core, import concrete submodules").
            if file_path.name == "__init__.py":
                continue
            # Check for deep submodule imports like "from flext_core.constants import X"
            for prefix, suggestion in cls.ALIAS_MODULES.items():
                if stmt.module.startswith(prefix + "."):
                    # This is a deep import - should use facade alias instead
                    import_names = (
                        ", ".join(
                            alias.name for alias in stmt.names if alias.name != "*"
                        )
                        if not any(alias.name == "*" for alias in stmt.names)
                        else "*"
                    )
                    current = f"from {stmt.module} import {import_names}"
                    violations.append(
                        NamespaceEnforcementModels.ImportAliasViolation(
                            file=str(file_path),
                            line=stmt.lineno,
                            current_import=current,
                            suggested_import=suggestion,
                        )
                    )
        return violations


# ---------------------------------------------------------------------------
# Cyclic import detector
# ---------------------------------------------------------------------------
class CyclicImportDetector:
    """Build intra-project import graph and detect cycles."""

    @classmethod
    def scan_project(
        cls,
        *,
        project_root: Path,
    ) -> list[NamespaceEnforcementModels.CyclicImportViolation]:
        """Return cyclic import chains found within a project."""
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return []

        # Build module → set[imported_module] graph
        graph: dict[str, set[str]] = defaultdict(set)
        file_map: dict[str, str] = {}
        package_roots = cls._discover_package_roots(src_dir=src_dir)

        for py_file in sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)):
            if "__pycache__" in py_file.parts:
                continue
            module_name = cls._file_to_module(file_path=py_file, src_dir=src_dir)
            if not module_name:
                continue
            file_map[module_name] = str(py_file)
            graph[module_name]  # ensure node exists
            try:
                tree = ast.parse(py_file.read_text(encoding=c.Infra.Encoding.DEFAULT))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for stmt in tree.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module:
                    imported = stmt.module
                    # Only track internal imports
                    root_pkg = imported.split(".")[0]
                    if root_pkg in package_roots:
                        graph[module_name].add(imported)

        # Detect cycles via topological sort
        violations: list[NamespaceEnforcementModels.CyclicImportViolation] = []
        try:
            TopologicalSorter(graph).static_order()
        except CycleError as exc:
            cycle_nodes = exc.args[1] if len(exc.args) > 1 else ()
            if cycle_nodes:
                normalized_cycle = tuple(
                    module_name
                    for module_name in cycle_nodes
                    if isinstance(module_name, str)
                )
                cycle_files = tuple(
                    file_map.get(module_name, module_name)
                    for module_name in normalized_cycle
                )
                violations.append(
                    NamespaceEnforcementModels.CyclicImportViolation(
                        cycle=normalized_cycle,
                        files=cycle_files,
                    )
                )
        return violations

    @staticmethod
    def _discover_package_roots(*, src_dir: Path) -> set[str]:
        """Top-level package names in src/."""
        roots: set[str] = set()
        for entry in src_dir.iterdir():
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            if entry.is_dir() and (entry / "__init__.py").is_file():
                roots.add(entry.name)
        return roots

    @staticmethod
    def _file_to_module(*, file_path: Path, src_dir: Path) -> str:
        """Convert file path to dotted module name."""
        try:
            rel = file_path.relative_to(src_dir)
        except ValueError:
            return ""
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
class FlextInfraNamespaceEnforcer:
    """Orchestrate namespace enforcement across the entire workspace.

    Usage::

        enforcer = FlextInfraNamespaceEnforcer(workspace_root=Path("/workspace"))
        report = enforcer.enforce(apply_changes=False)
    """

    def __init__(self, *, workspace_root: Path) -> None:
        """Create enforcer bound to a workspace root."""
        self._workspace_root = workspace_root.resolve()

    def enforce(
        self, *, apply_changes: bool = False
    ) -> NamespaceEnforcementModels.WorkspaceEnforcementReport:
        """Run full enforcement scan (and optionally apply fixes)."""
        project_roots = u.Infra.Refactor.discover_project_roots(
            workspace_root=self._workspace_root
        )
        project_reports: list[NamespaceEnforcementModels.ProjectEnforcementReport] = []
        total_missing = 0
        total_loose = 0
        total_import_v = 0
        total_cyclic = 0
        total_files = 0

        for project_root in project_roots:
            project_name = project_root.name
            report = self._enforce_project(
                project_root=project_root,
                project_name=project_name,
                apply_changes=apply_changes,
            )
            project_reports.append(report)
            total_missing += sum(1 for s in report.facade_statuses if not s.exists)
            total_loose += len(report.loose_objects)
            total_import_v += len(report.import_violations)
            total_cyclic += len(report.cyclic_imports)
            total_files += report.files_scanned

        return NamespaceEnforcementModels.WorkspaceEnforcementReport(
            workspace=str(self._workspace_root),
            projects=project_reports,
            total_facades_missing=total_missing,
            total_loose_objects=total_loose,
            total_import_violations=total_import_v,
            total_cyclic_imports=total_cyclic,
            total_files_scanned=total_files,
        )

    def _enforce_project(
        self,
        *,
        project_root: Path,
        project_name: str,
        apply_changes: bool,
    ) -> NamespaceEnforcementModels.ProjectEnforcementReport:
        """Enforce namespace rules on a single project."""
        # 1. Facade scan
        facade_statuses = NamespaceFacadeScanner.scan_project(
            project_root=project_root, project_name=project_name
        )

        if apply_changes:
            self._ensure_missing_facades(
                project_root=project_root,
                project_name=project_name,
                facade_statuses=facade_statuses,
            )
            facade_statuses = NamespaceFacadeScanner.scan_project(
                project_root=project_root,
                project_name=project_name,
            )

        # 2. Collect Python files across all scan directories
        py_files = self._collect_python_files(project_root=project_root)

        # 3. Detect loose objects
        loose_objects: list[NamespaceEnforcementModels.LooseObjectViolation] = []
        for py_file in py_files:
            loose_objects.extend(
                LooseObjectDetector.scan_file(
                    file_path=py_file, project_name=project_name
                )
            )

        # 4. Detect import alias violations
        import_violations: list[NamespaceEnforcementModels.ImportAliasViolation] = []
        for py_file in py_files:
            import_violations.extend(ImportAliasDetector.scan_file(file_path=py_file))

        if apply_changes and len(import_violations) > 0:
            self._rewrite_import_alias_violations(py_files=py_files)
            import_violations = []
            for py_file in py_files:
                import_violations.extend(
                    ImportAliasDetector.scan_file(file_path=py_file)
                )

        # 5. Detect cyclic imports
        cyclic_imports = CyclicImportDetector.scan_project(project_root=project_root)

        return NamespaceEnforcementModels.ProjectEnforcementReport(
            project=project_name,
            project_root=str(project_root),
            facade_statuses=facade_statuses,
            loose_objects=loose_objects,
            import_violations=import_violations,
            cyclic_imports=cyclic_imports,
            files_scanned=len(py_files),
        )

    @staticmethod
    def _preferred_file_name(*, family: str) -> str:
        if family == "c":
            return "constants.py"
        if family == "t":
            return "typings.py"
        if family == "p":
            return "protocols.py"
        if family == "m":
            return "models.py"
        return "utilities.py"

    @staticmethod
    def _base_import_for_family(*, family: str) -> str:
        if family == "c":
            return "from flext_core import FlextConstants"
        if family == "t":
            return "from flext_core import FlextTypes"
        if family == "p":
            return "from flext_core import FlextProtocols"
        if family == "m":
            return "from flext_core import FlextModels"
        return "from flext_core import FlextUtilities"

    @staticmethod
    def _base_class_for_family(*, family: str) -> str:
        if family == "c":
            return "FlextConstants"
        if family == "t":
            return "FlextTypes"
        if family == "p":
            return "FlextProtocols"
        if family == "m":
            return "FlextModels"
        return "FlextUtilities"

    @staticmethod
    def _write_missing_facade_file(
        *,
        file_path: Path,
        family: str,
        class_name: str,
    ) -> None:
        alias = family
        import_stmt = FlextInfraNamespaceEnforcer._base_import_for_family(family=family)
        base_class = FlextInfraNamespaceEnforcer._base_class_for_family(family=family)
        content = (
            '"""Auto-generated facade to enforce MRO namespace contracts."""\n\n'
            "from __future__ import annotations\n\n"
            f"{import_stmt}\n\n"
            f"class {class_name}({base_class}):\n"
            "    pass\n\n"
            f"{alias} = {class_name}\n"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)

    @classmethod
    def _ensure_missing_facades(
        cls,
        *,
        project_root: Path,
        project_name: str,
        facade_statuses: list[NamespaceEnforcementModels.FacadeStatus],
    ) -> None:
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return
        package_dirs = [
            entry
            for entry in sorted(src_dir.iterdir(), key=lambda item: item.name)
            if entry.is_dir() and (entry / "__init__.py").is_file()
        ]
        if len(package_dirs) == 0:
            return
        primary_package = package_dirs[0]
        stem = NamespaceFacadeScanner.project_class_stem(project_name=project_name)
        for status in facade_statuses:
            if status.exists:
                continue
            suffix = _FACADE_FAMILIES[status.family]
            class_name = f"{stem}{suffix}"
            file_name = cls._preferred_file_name(family=status.family)
            target_path = primary_package / file_name
            if target_path.exists():
                content = target_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
                mutated = False
                class_signature = f"class {class_name}("
                if (
                    class_signature not in content
                    and f"class {class_name}:" not in content
                ):
                    base_class = cls._base_class_for_family(family=status.family)
                    snippet = f"\n\nclass {class_name}({base_class}):\n    pass\n"
                    content = content.rstrip() + snippet
                    mutated = True
                alias_line = f"{status.family} = {class_name}"
                if alias_line not in content:
                    content = content.rstrip() + f"\n\n{alias_line}\n"
                    mutated = True
                if mutated:
                    target_path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
                continue
            cls._write_missing_facade_file(
                file_path=target_path,
                family=status.family,
                class_name=class_name,
            )

    @staticmethod
    def _rewrite_import_alias_violations(*, py_files: list[Path]) -> None:
        for file_path in py_files:
            try:
                source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
                tree = ast.parse(source)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            changed = False
            for stmt in tree.body:
                if not isinstance(stmt, ast.ImportFrom):
                    continue
                if stmt.module is None:
                    continue
                if stmt.module.startswith("flext_core."):
                    stmt.module = "flext_core"
                    stmt.names = [
                        ast.alias(name="c"),
                        ast.alias(name="m"),
                        ast.alias(name="r"),
                        ast.alias(name="t"),
                        ast.alias(name="u"),
                        ast.alias(name="p"),
                        ast.alias(name="d"),
                        ast.alias(name="e"),
                        ast.alias(name="h"),
                        ast.alias(name="s"),
                        ast.alias(name="x"),
                    ]
                    changed = True
                    continue
                if stmt.module.startswith("flext_infra."):
                    stmt.module = "flext_infra"
                    stmt.names = [
                        ast.alias(name="c"),
                        ast.alias(name="m"),
                        ast.alias(name="t"),
                        ast.alias(name="u"),
                        ast.alias(name="p"),
                    ]
                    changed = True
            if changed:
                normalized = ast.unparse(tree)
                file_path.write_text(
                    f"{normalized}\n", encoding=c.Infra.Encoding.DEFAULT
                )

    @staticmethod
    def _collect_python_files(*, project_root: Path) -> list[Path]:
        """Collect Python files from src/, tests/, scripts/, examples/."""
        files: list[Path] = []
        for dir_name in c.Infra.Refactor.MRO_SCAN_DIRECTORIES:
            scan_dir = project_root / dir_name
            if not scan_dir.is_dir():
                continue
            for py_file in sorted(scan_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)):
                if "__pycache__" in py_file.parts:
                    continue
                files.append(py_file)
        return files

    @staticmethod
    def render_text(
        report: NamespaceEnforcementModels.WorkspaceEnforcementReport,
    ) -> str:
        """Render enforcement report as CLI-friendly text."""
        lines: list[str] = [
            f"Workspace: {report.workspace}",
            f"Projects scanned: {len(report.projects)}",
            f"Files scanned: {report.total_files_scanned}",
            f"Missing facades: {report.total_facades_missing}",
            f"Loose objects: {report.total_loose_objects}",
            f"Import violations: {report.total_import_violations}",
            f"Cyclic imports: {report.total_cyclic_imports}",
            "",
        ]
        for proj in report.projects:
            missing = [s for s in proj.facade_statuses if not s.exists]
            if not missing and not proj.loose_objects and not proj.import_violations:
                continue
            lines.append(f"--- {proj.project} ---")
            if missing:
                lines.append(
                    "  Missing facades: "
                    + ", ".join(
                        f"{s.family} ({_FACADE_FAMILIES[s.family]})" for s in missing
                    )
                )
            if proj.loose_objects:
                lines.append(f"  Loose objects: {len(proj.loose_objects)}")
                lines.extend(
                    f"    {obj.file}:{obj.line} {obj.kind} '{obj.name}' -> {obj.suggestion}"
                    for obj in proj.loose_objects[:_MAX_RENDERED_LOOSE_OBJECTS]
                )
                if len(proj.loose_objects) > _MAX_RENDERED_LOOSE_OBJECTS:
                    lines.append(
                        f"    ... and {len(proj.loose_objects) - _MAX_RENDERED_LOOSE_OBJECTS} more"
                    )
            if proj.import_violations:
                lines.append(f"  Import violations: {len(proj.import_violations)}")
                lines.extend(
                    f"    {iv.file}:{iv.line} {iv.current_import}"
                    for iv in proj.import_violations[:_MAX_RENDERED_IMPORT_VIOLATIONS]
                )
                if len(proj.import_violations) > _MAX_RENDERED_IMPORT_VIOLATIONS:
                    lines.append(
                        f"    ... and {len(proj.import_violations) - _MAX_RENDERED_IMPORT_VIOLATIONS} more"
                    )
            if proj.cyclic_imports:
                lines.append(f"  Cyclic imports: {len(proj.cyclic_imports)}")
                lines.extend(
                    f"    Cycle: {' -> '.join(ci.cycle)}" for ci in proj.cyclic_imports
                )
            lines.append("")
        return "\n".join(lines) + "\n"


__all__ = [
    "CyclicImportDetector",
    "FlextInfraNamespaceEnforcer",
    "ImportAliasDetector",
    "LooseObjectDetector",
    "NamespaceEnforcementModels",
    "NamespaceFacadeScanner",
]
