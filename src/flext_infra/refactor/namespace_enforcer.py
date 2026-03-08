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
from io import StringIO
import token
import tokenize
from collections import defaultdict
from dataclasses import dataclass
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

    class InternalImportViolation(BaseModel):
        """Import violation record for private module/symbol usage."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(ge=1, description="Line number")
        current_import: str = Field(description="Current import statement")
        detail: str = Field(description="Violation details")

    class ManualProtocolViolation(BaseModel):
        """Protocol class defined outside canonical protocols module paths."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(ge=1, description="Line number")
        name: str = Field(min_length=1, description="Protocol class name")
        suggestion: str = Field(
            default="Move to protocols.py/protocols/*.py/_protocols.py",
            description="Canonical destination",
        )

    class RuntimeAliasViolation(BaseModel):
        """A module missing or misusing a runtime alias (c = FlextConstants)."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(default=0, ge=0, description="Line number (0 if missing)")
        kind: str = Field(description="missing|duplicate|wrong_class")
        alias: str = Field(description="Expected alias letter")
        detail: str = Field(default="", description="Explanation")

    class MissingFutureAnnotationsViolation(BaseModel):
        """A Python module missing `from __future__ import annotations`."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")

    class ManualTypingAliasViolation(BaseModel):
        """Typing alias definition outside canonical `typings*` modules."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(ge=1, description="Line number")
        name: str = Field(min_length=1, description="Type alias name")
        detail: str = Field(default="", description="Violation details")

    class CompatibilityAliasViolation(BaseModel):
        """Compatibility alias pattern (`Old = New`) flagged for cleanup."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(ge=1, description="Line number")
        alias_name: str = Field(min_length=1, description="Compatibility alias symbol")
        target_name: str = Field(min_length=1, description="Target symbol")

    class ParseFailureViolation(BaseModel):
        """Structured parse/read failure captured with typed origin."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        stage: str = Field(min_length=1, description="Enforcement stage")
        error_type: str = Field(min_length=1, description="Exception class name")
        detail: str = Field(default="", description="Error detail")

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
        internal_import_violations: list[
            NamespaceEnforcementModels.InternalImportViolation
        ] = Field(
            default_factory=list, description="Private/internal import violations"
        )
        manual_protocol_violations: list[
            NamespaceEnforcementModels.ManualProtocolViolation
        ] = Field(default_factory=list, description="Manual protocol violations")
        cyclic_imports: list[NamespaceEnforcementModels.CyclicImportViolation] = Field(
            default_factory=list, description="Cyclic import violations"
        )
        runtime_alias_violations: list[
            NamespaceEnforcementModels.RuntimeAliasViolation
        ] = Field(default_factory=list, description="Runtime alias violations")
        future_violations: list[
            NamespaceEnforcementModels.MissingFutureAnnotationsViolation
        ] = Field(default_factory=list, description="Missing __future__ violations")
        manual_typing_violations: list[
            NamespaceEnforcementModels.ManualTypingAliasViolation
        ] = Field(default_factory=list, description="Type alias outside typings scope")
        compatibility_alias_violations: list[
            NamespaceEnforcementModels.CompatibilityAliasViolation
        ] = Field(default_factory=list, description="Compatibility alias violations")
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation] = Field(
            default_factory=list,
            description="Read/parse failures classified by type",
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
        total_internal_import_violations: int = Field(
            default=0, ge=0, description="Private/internal import violations"
        )
        total_manual_protocol_violations: int = Field(
            default=0,
            ge=0,
            description="Manual protocol violations",
        )
        total_cyclic_imports: int = Field(
            default=0, ge=0, description="Cyclic import chains"
        )
        total_runtime_alias_violations: int = Field(
            default=0, ge=0, description="Runtime alias violations"
        )
        total_future_violations: int = Field(
            default=0, ge=0, description="Missing __future__ violations"
        )
        total_manual_typing_violations: int = Field(
            default=0, ge=0, description="Type aliases outside typings scope"
        )
        total_compatibility_alias_violations: int = Field(
            default=0, ge=0, description="Compatibility alias violations"
        )
        total_parse_failures: int = Field(
            default=0,
            ge=0,
            description="Read/parse failures across workspace",
        )
        total_files_scanned: int = Field(
            default=0, ge=0, description="All files scanned"
        )


@dataclass(frozen=True, slots=True)
class _ParsedPythonModule:
    source: str
    tree: ast.Module


def _load_python_module(
    file_path: Path,
    *,
    stage: str = "scan",
    parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
    | None = None,
) -> _ParsedPythonModule | None:
    try:
        source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
    except UnicodeDecodeError as exc:
        if parse_failures is not None:
            parse_failures.append(
                _new_parse_failure_violation(
                    file=str(file_path),
                    stage=stage,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
            )
        return None
    except OSError as exc:
        if parse_failures is not None:
            parse_failures.append(
                _new_parse_failure_violation(
                    file=str(file_path),
                    stage=stage,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
            )
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        if parse_failures is not None:
            parse_failures.append(
                _new_parse_failure_violation(
                    file=str(file_path),
                    stage=stage,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
            )
        return None
    return _ParsedPythonModule(source=source, tree=tree)


def _new_facade_status(
    *, family: str, exists: bool, class_name: str, file: str, symbol_count: int
) -> NamespaceEnforcementModels.FacadeStatus:
    return NamespaceEnforcementModels.FacadeStatus.model_validate({
        "family": family,
        "exists": exists,
        "class_name": class_name,
        "file": file,
        "symbol_count": symbol_count,
    })


def _new_loose_object_violation(
    *, file: str, line: int, name: str, kind: str, suggestion: str
) -> NamespaceEnforcementModels.LooseObjectViolation:
    return NamespaceEnforcementModels.LooseObjectViolation.model_validate({
        "file": file,
        "line": line,
        "name": name,
        "kind": kind,
        "suggestion": suggestion,
    })


def _new_import_alias_violation(
    *, file: str, line: int, current_import: str, suggested_import: str
) -> NamespaceEnforcementModels.ImportAliasViolation:
    return NamespaceEnforcementModels.ImportAliasViolation.model_validate({
        "file": file,
        "line": line,
        "current_import": current_import,
        "suggested_import": suggested_import,
    })


def _new_internal_import_violation(
    *, file: str, line: int, current_import: str, detail: str
) -> NamespaceEnforcementModels.InternalImportViolation:
    return NamespaceEnforcementModels.InternalImportViolation.model_validate({
        "file": file,
        "line": line,
        "current_import": current_import,
        "detail": detail,
    })


def _new_manual_protocol_violation(
    *, file: str, line: int, name: str, suggestion: str = ""
) -> NamespaceEnforcementModels.ManualProtocolViolation:
    payload = {
        "file": file,
        "line": line,
        "name": name,
    }
    if len(suggestion) > 0:
        payload["suggestion"] = suggestion
    return NamespaceEnforcementModels.ManualProtocolViolation.model_validate(payload)


def _new_cyclic_import_violation(
    *, cycle: tuple[str, ...], files: tuple[str, ...]
) -> NamespaceEnforcementModels.CyclicImportViolation:
    return NamespaceEnforcementModels.CyclicImportViolation.model_validate({
        "cycle": cycle,
        "files": files,
    })


def _new_runtime_alias_violation(
    *, file: str, kind: str, alias: str, detail: str, line: int = 0
) -> NamespaceEnforcementModels.RuntimeAliasViolation:
    return NamespaceEnforcementModels.RuntimeAliasViolation.model_validate({
        "file": file,
        "line": line,
        "kind": kind,
        "alias": alias,
        "detail": detail,
    })


def _new_future_annotations_violation(
    *, file: str
) -> NamespaceEnforcementModels.MissingFutureAnnotationsViolation:
    return NamespaceEnforcementModels.MissingFutureAnnotationsViolation.model_validate({
        "file": file
    })


def _new_manual_typing_alias_violation(
    *, file: str, line: int, name: str, detail: str
) -> NamespaceEnforcementModels.ManualTypingAliasViolation:
    return NamespaceEnforcementModels.ManualTypingAliasViolation.model_validate({
        "file": file,
        "line": line,
        "name": name,
        "detail": detail,
    })


def _new_compatibility_alias_violation(
    *, file: str, line: int, alias_name: str, target_name: str
) -> NamespaceEnforcementModels.CompatibilityAliasViolation:
    return NamespaceEnforcementModels.CompatibilityAliasViolation.model_validate({
        "file": file,
        "line": line,
        "alias_name": alias_name,
        "target_name": target_name,
    })


def _new_parse_failure_violation(
    *, file: str, stage: str, error_type: str, detail: str
) -> NamespaceEnforcementModels.ParseFailureViolation:
    return NamespaceEnforcementModels.ParseFailureViolation.model_validate({
        "file": file,
        "stage": stage,
        "error_type": error_type,
        "detail": detail,
    })


def _new_workspace_enforcement_report(
    *,
    workspace: str,
    projects: list[NamespaceEnforcementModels.ProjectEnforcementReport],
    total_facades_missing: int,
    total_loose_objects: int,
    total_import_violations: int,
    total_internal_import_violations: int,
    total_manual_protocol_violations: int,
    total_cyclic_imports: int,
    total_runtime_alias_violations: int,
    total_future_violations: int,
    total_manual_typing_violations: int,
    total_compatibility_alias_violations: int,
    total_parse_failures: int,
    total_files_scanned: int,
) -> NamespaceEnforcementModels.WorkspaceEnforcementReport:
    return NamespaceEnforcementModels.WorkspaceEnforcementReport.model_validate({
        "workspace": workspace,
        "projects": projects,
        "total_facades_missing": total_facades_missing,
        "total_loose_objects": total_loose_objects,
        "total_import_violations": total_import_violations,
        "total_internal_import_violations": total_internal_import_violations,
        "total_manual_protocol_violations": total_manual_protocol_violations,
        "total_cyclic_imports": total_cyclic_imports,
        "total_runtime_alias_violations": total_runtime_alias_violations,
        "total_future_violations": total_future_violations,
        "total_manual_typing_violations": total_manual_typing_violations,
        "total_compatibility_alias_violations": total_compatibility_alias_violations,
        "total_parse_failures": total_parse_failures,
        "total_files_scanned": total_files_scanned,
    })


def _new_project_enforcement_report(
    *,
    project: str,
    project_root: str,
    facade_statuses: list[NamespaceEnforcementModels.FacadeStatus],
    loose_objects: list[NamespaceEnforcementModels.LooseObjectViolation],
    import_violations: list[NamespaceEnforcementModels.ImportAliasViolation],
    internal_import_violations: list[
        NamespaceEnforcementModels.InternalImportViolation
    ],
    manual_protocol_violations: list[
        NamespaceEnforcementModels.ManualProtocolViolation
    ],
    cyclic_imports: list[NamespaceEnforcementModels.CyclicImportViolation],
    runtime_alias_violations: list[NamespaceEnforcementModels.RuntimeAliasViolation],
    future_violations: list[
        NamespaceEnforcementModels.MissingFutureAnnotationsViolation
    ],
    manual_typing_violations: list[
        NamespaceEnforcementModels.ManualTypingAliasViolation
    ],
    compatibility_alias_violations: list[
        NamespaceEnforcementModels.CompatibilityAliasViolation
    ],
    parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation],
    files_scanned: int,
) -> NamespaceEnforcementModels.ProjectEnforcementReport:
    return NamespaceEnforcementModels.ProjectEnforcementReport.model_validate({
        "project": project,
        "project_root": project_root,
        "facade_statuses": facade_statuses,
        "loose_objects": loose_objects,
        "import_violations": import_violations,
        "internal_import_violations": internal_import_violations,
        "manual_protocol_violations": manual_protocol_violations,
        "cyclic_imports": cyclic_imports,
        "runtime_alias_violations": runtime_alias_violations,
        "future_violations": future_violations,
        "manual_typing_violations": manual_typing_violations,
        "compatibility_alias_violations": compatibility_alias_violations,
        "parse_failures": parse_failures,
        "files_scanned": files_scanned,
    })


# ---------------------------------------------------------------------------
# Constants — all derived from c.Infra.Refactor.* (single source of truth)
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

# Derived lookups — computed once from FAMILY_SUFFIXES (SSOT)
_FILE_TO_FAMILY: dict[str, str] = {
    f"{suffix.lower()}.py": alias
    for alias, suffix in c.Infra.Refactor.FAMILY_SUFFIXES.items()
}
"""Map filename → family alias letter, derived from FAMILY_SUFFIXES."""

_FAMILY_EXPECTED_ALIAS: dict[str, tuple[str, str]] = {
    f"{suffix.lower()}.py": (alias, suffix)
    for alias, suffix in c.Infra.Refactor.FAMILY_SUFFIXES.items()
}
"""Map filename → (alias_letter, class_suffix) for rewrite operations."""

_CANONICAL_PROTOCOL_FILES: frozenset[str] = c.Infra.Refactor.MRO_PROTOCOLS_FILE_NAMES
"""Canonical protocol file names — from c.Infra.Refactor."""

_CANONICAL_PROTOCOL_DIR: str = c.Infra.Refactor.MRO_PROTOCOLS_DIRECTORY
"""Canonical protocol directory — from c.Infra.Refactor."""

_CANONICAL_TYPINGS_FILES: frozenset[str] = c.Infra.Refactor.MRO_TYPINGS_FILE_NAMES
"""Canonical typings file names — from c.Infra.Refactor."""

_CANONICAL_TYPINGS_DIR: str = c.Infra.Refactor.MRO_TYPINGS_DIRECTORY
"""Canonical typings directory — from c.Infra.Refactor."""


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
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
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
                parse_failures=parse_failures,
            )
            results.append(
                _new_facade_status(
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
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation] | None,
    ) -> tuple[str, str, int]:
        """Return (class_name, file_path, symbol_count) or empty."""
        file_pattern = _FACADE_FILE_PATTERNS[family]
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return ("", "", 0)
        for file_path in src_dir.rglob(file_pattern):
            parsed = _load_python_module(
                file_path,
                stage="facade-scan",
                parse_failures=parse_failures,
            )
            if parsed is None:
                continue
            tree = parsed.tree
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
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.LooseObjectViolation]:
        """Return loose objects found in a single file."""
        if file_path.name in _PROTECTED_FILES:
            return []
        if file_path.name in _SETTINGS_FILE_NAMES:
            return []
        parsed = _load_python_module(
            file_path,
            stage="loose-object-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree

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
            return _new_loose_object_violation(
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
                return _new_loose_object_violation(
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
                    return _new_loose_object_violation(
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
                return _new_loose_object_violation(
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
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.ImportAliasViolation]:
        """Return import alias violations for one file."""
        parsed = _load_python_module(
            file_path,
            stage="import-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree

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
                        _new_import_alias_violation(
                            file=str(file_path),
                            line=stmt.lineno,
                            current_import=current,
                            suggested_import=suggestion,
                        )
                    )
        return violations


class InternalImportDetector:
    """Detect direct private imports that bypass public facade imports."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.InternalImportViolation]:
        """Return internal/private import violations for one Python file."""
        parsed = _load_python_module(
            file_path,
            stage="internal-import-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[NamespaceEnforcementModels.InternalImportViolation] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.ImportFrom):
                continue
            if stmt.module is None:
                continue
            if file_path.name == "__init__.py":
                continue
            imported_names = [alias.name for alias in stmt.names if alias.name != "*"]
            import_list = ", ".join(imported_names) if imported_names else "*"
            current_import = f"from {stmt.module} import {import_list}"
            has_private_module = "._" in stmt.module
            has_private_symbol = any(name.startswith("_") for name in imported_names)
            if not (has_private_module or has_private_symbol):
                continue
            detail = (
                "private module import"
                if has_private_module
                else "private symbol import"
            )
            violations.append(
                _new_internal_import_violation(
                    file=str(file_path),
                    line=stmt.lineno,
                    current_import=current_import,
                    detail=detail,
                )
            )
        return violations


class ManualProtocolDetector:
    """Detect Protocol-like classes outside canonical `protocols*` locations."""

    # Derived from c.Infra.Refactor.MRO_PROTOCOLS_* (SSOT)
    CANONICAL_FILE_NAMES: ClassVar[frozenset[str]] = _CANONICAL_PROTOCOL_FILES
    CANONICAL_DIR_NAME: ClassVar[str] = _CANONICAL_PROTOCOL_DIR

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.ManualProtocolViolation]:
        """Return protocol violations for one file outside canonical paths."""
        in_canonical_file = file_path.name in cls.CANONICAL_FILE_NAMES
        in_canonical_dir = cls.CANONICAL_DIR_NAME in file_path.parts
        if in_canonical_file or in_canonical_dir:
            return []
        if file_path.name in _PROTECTED_FILES:
            return []
        parsed = _load_python_module(
            file_path,
            stage="manual-protocol-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[NamespaceEnforcementModels.ManualProtocolViolation] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            if cls.is_protocol_class(stmt):
                violations.append(
                    _new_manual_protocol_violation(
                        file=str(file_path),
                        line=stmt.lineno,
                        name=stmt.name,
                    )
                )
        return violations

    @staticmethod
    def is_protocol_class(node: ast.ClassDef) -> bool:
        """Return True when class bases indicate typing.Protocol lineage."""
        for base_expr in node.bases:
            if isinstance(base_expr, ast.Name) and base_expr.id == "Protocol":
                return True
            if isinstance(base_expr, ast.Attribute) and base_expr.attr == "Protocol":
                return True
            if isinstance(base_expr, ast.Subscript):
                root_expr = base_expr.value
                if isinstance(root_expr, ast.Name) and root_expr.id == "Protocol":
                    return True
                if (
                    isinstance(root_expr, ast.Attribute)
                    and root_expr.attr == "Protocol"
                ):
                    return True
        return False


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
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
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
            parsed = _load_python_module(
                py_file,
                stage="cyclic-import-scan",
                parse_failures=parse_failures,
            )
            if parsed is None:
                continue
            tree = parsed.tree
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
            _ = TopologicalSorter(graph).static_order()
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
                    _new_cyclic_import_violation(
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
# Runtime alias detector
# ---------------------------------------------------------------------------
class RuntimeAliasDetector:
    """Detect missing or misused runtime alias assignments (c = Flext*Constants).

    Per AGENTS.md §2: 'Namespace aliases are canonical public API surfaces'.
    Per AGENTS.md §4: 'No double-assignment of facade aliases'.
    Each facade module MUST have exactly one `alias = ClassName` at module bottom.
    """

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_name: str,
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.RuntimeAliasViolation]:
        """Return runtime alias violations for one file."""
        # Only check facade files inside src/ (not tests/constants.py etc.)
        # Dynamic check: file must match a known facade filename (from FAMILY_SUFFIXES)
        if file_path.name not in _FILE_TO_FAMILY:
            return []
        if file_path.name in _PROTECTED_FILES:
            return []
        # Only facade files live under src/ — skip tests/scripts/examples
        if "src" not in file_path.parts:
            return []
        parsed = _load_python_module(
            file_path,
            stage="runtime-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree

        violations: list[NamespaceEnforcementModels.RuntimeAliasViolation] = []
        _ = project_name
        # Determine which alias letter this file should define
        family = cls._family_for_file(file_name=file_path.name)
        if not family:
            return []

        # Find all top-level `x = SomeName` where x is a single letter
        alias_assignments: list[tuple[int, str, str]] = []  # (line, alias, value)
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    continue
                if len(target.id) == 1 and isinstance(stmt.value, ast.Name):
                    alias_assignments.append((stmt.lineno, target.id, stmt.value.id))

        expected_alias = family
        matches = [a for a in alias_assignments if a[1] == expected_alias]
        if len(matches) == 0:
            violations.append(
                _new_runtime_alias_violation(
                    file=str(file_path),
                    kind="missing",
                    alias=expected_alias,
                    detail=f"No '{expected_alias} = ...' assignment found",
                )
            )
        elif len(matches) > 1:
            violations.append(
                _new_runtime_alias_violation(
                    file=str(file_path),
                    line=matches[1][0],
                    kind="duplicate",
                    alias=expected_alias,
                    detail=f"Duplicate alias assignment at lines {', '.join(str(m[0]) for m in matches)}",
                )
            )

        return violations

    @staticmethod
    def _family_for_file(*, file_name: str) -> str:
        """Map file name to expected facade alias letter.

        Uses _FILE_TO_FAMILY derived from c.Infra.Refactor.FAMILY_SUFFIXES.
        """
        return _FILE_TO_FAMILY.get(file_name, "")


# ---------------------------------------------------------------------------
# Future annotations detector
# ---------------------------------------------------------------------------
class FutureAnnotationsDetector:
    """Detect Python modules missing `from __future__ import annotations`.

    Per AGENTS.md §3: 'from __future__ import annotations is mandatory in Python modules.'
    """

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.MissingFutureAnnotationsViolation]:
        """Return violations if __future__ annotations is missing."""
        if file_path.name == "py.typed":
            return []
        parsed = _load_python_module(
            file_path,
            stage="future-annotations-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []

        # Empty files are OK
        if len(parsed.tree.body) == 0:
            return []
        # Files with only a docstring are OK
        if (
            len(parsed.tree.body) == 1
            and isinstance(parsed.tree.body[0], ast.Expr)
            and isinstance(parsed.tree.body[0].value, ast.Constant)
        ):
            return []

        for stmt in parsed.tree.body:
            if (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module == "__future__"
                and any(alias.name == "annotations" for alias in stmt.names)
            ):
                return []
            # Only look at the first few statements
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                break

        return [_new_future_annotations_violation(file=str(file_path))]


class ManualTypingAliasDetector:
    """Detect type-alias declarations outside canonical `typings*` scope."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.ManualTypingAliasViolation]:
        """Return manual typing-alias violations for one file."""
        if file_path.suffix != ".py":
            return []
        # Canonical typings scope — derived from c.Infra.Refactor.MRO_TYPINGS_*
        if file_path.name in _CANONICAL_TYPINGS_FILES:
            return []
        if _CANONICAL_TYPINGS_DIR in file_path.parts:
            return []
        parsed = _load_python_module(
            file_path,
            stage="manual-typing-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        source, tree = parsed.source, parsed.tree
        violations: list[NamespaceEnforcementModels.ManualTypingAliasViolation] = []
        for stmt in tree.body:
            if isinstance(stmt, ast.TypeAlias):
                alias_name = stmt.name.id
                violations.append(
                    _new_manual_typing_alias_violation(
                        file=str(file_path),
                        line=stmt.lineno,
                        name=alias_name,
                        detail="PEP695 alias must be centralized under typings scope",
                    )
                )
                continue
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                annotation_src = ast.get_source_segment(source, stmt.annotation) or ""
                if "TypeAlias" in annotation_src:
                    violations.append(
                        _new_manual_typing_alias_violation(
                            file=str(file_path),
                            line=stmt.lineno,
                            name=stmt.target.id,
                            detail="TypeAlias assignment must be centralized under typings scope",
                        )
                    )
        return violations


class CompatibilityAliasDetector:
    """Detect compatibility alias assignments (`OldName = NewName`)."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation]
        | None = None,
    ) -> list[NamespaceEnforcementModels.CompatibilityAliasViolation]:
        """Return compatibility alias violations for one file."""
        if file_path.suffix != ".py":
            return []
        parsed = _load_python_module(
            file_path,
            stage="compatibility-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[NamespaceEnforcementModels.CompatibilityAliasViolation] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not isinstance(stmt.value, ast.Name):
                continue
            alias_name = target.id
            target_name = stmt.value.id
            if len(alias_name) == 1:
                continue
            if alias_name in {"__all__", "__version__", "__version_info__"}:
                continue
            if alias_name == target_name:
                continue
            if alias_name.isupper() and target_name.isupper():
                continue
            if alias_name[0].isupper() and target_name[0].isupper():
                violations.append(
                    _new_compatibility_alias_violation(
                        file=str(file_path),
                        line=stmt.lineno,
                        alias_name=alias_name,
                        target_name=target_name,
                    )
                )
        return violations


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
        super().__init__()
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
        total_internal_import_v = 0
        total_cyclic = 0
        total_alias_v = 0
        total_future_v = 0
        total_manual_protocol_v = 0
        total_manual_typing_v = 0
        total_compat_alias_v = 0
        total_parse_failures = 0
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
            total_internal_import_v += len(report.internal_import_violations)
            total_cyclic += len(report.cyclic_imports)
            total_alias_v += len(report.runtime_alias_violations)
            total_future_v += len(report.future_violations)
            total_manual_protocol_v += len(report.manual_protocol_violations)
            total_manual_typing_v += len(report.manual_typing_violations)
            total_compat_alias_v += len(report.compatibility_alias_violations)
            total_parse_failures += len(report.parse_failures)
            total_files += report.files_scanned

        return _new_workspace_enforcement_report(
            workspace=str(self._workspace_root),
            projects=project_reports,
            total_facades_missing=total_missing,
            total_loose_objects=total_loose,
            total_import_violations=total_import_v,
            total_internal_import_violations=total_internal_import_v,
            total_cyclic_imports=total_cyclic,
            total_runtime_alias_violations=total_alias_v,
            total_future_violations=total_future_v,
            total_manual_protocol_violations=total_manual_protocol_v,
            total_manual_typing_violations=total_manual_typing_v,
            total_compatibility_alias_violations=total_compat_alias_v,
            total_parse_failures=total_parse_failures,
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
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation] = []
        # 1. Facade scan
        facade_statuses = NamespaceFacadeScanner.scan_project(
            project_root=project_root,
            project_name=project_name,
            parse_failures=parse_failures,
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
                parse_failures=parse_failures,
            )

        # 2. Collect Python files across all scan directories
        py_files = self._collect_python_files(project_root=project_root)

        # 3. Detect loose objects
        loose_objects: list[NamespaceEnforcementModels.LooseObjectViolation] = []
        for py_file in py_files:
            loose_objects.extend(
                LooseObjectDetector.scan_file(
                    file_path=py_file,
                    project_name=project_name,
                    parse_failures=parse_failures,
                )
            )

        # 4. Detect import alias violations
        import_violations: list[NamespaceEnforcementModels.ImportAliasViolation] = []
        for py_file in py_files:
            import_violations.extend(
                ImportAliasDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )

        if apply_changes and len(import_violations) > 0:
            self._rewrite_import_alias_violations(py_files=py_files)
            import_violations = []
            for py_file in py_files:
                import_violations.extend(
                    ImportAliasDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )

        # 5. Detect cyclic imports
        cyclic_imports = CyclicImportDetector.scan_project(
            project_root=project_root,
            parse_failures=parse_failures,
        )

        internal_import_violations: list[
            NamespaceEnforcementModels.InternalImportViolation
        ] = []
        for py_file in py_files:
            internal_import_violations.extend(
                InternalImportDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )

        # 6. Detect runtime alias violations
        runtime_alias_violations: list[
            NamespaceEnforcementModels.RuntimeAliasViolation
        ] = []
        for py_file in py_files:
            runtime_alias_violations.extend(
                RuntimeAliasDetector.scan_file(
                    file_path=py_file,
                    project_name=project_name,
                    parse_failures=parse_failures,
                )
            )

        if apply_changes and len(runtime_alias_violations) > 0:
            self._rewrite_runtime_alias_violations(py_files=py_files)
            runtime_alias_violations = []
            for py_file in py_files:
                runtime_alias_violations.extend(
                    RuntimeAliasDetector.scan_file(
                        file_path=py_file,
                        project_name=project_name,
                        parse_failures=parse_failures,
                    )
                )

        # 7. Detect missing __future__ annotations
        future_violations: list[
            NamespaceEnforcementModels.MissingFutureAnnotationsViolation
        ] = []
        for py_file in py_files:
            future_violations.extend(
                FutureAnnotationsDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )

        if apply_changes and len(future_violations) > 0:
            self._rewrite_missing_future_annotations(py_files=py_files)
            future_violations = []
            for py_file in py_files:
                future_violations.extend(
                    FutureAnnotationsDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )

        manual_protocol_violations: list[
            NamespaceEnforcementModels.ManualProtocolViolation
        ] = []
        for py_file in py_files:
            manual_protocol_violations.extend(
                ManualProtocolDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )

        if apply_changes and len(manual_protocol_violations) > 0:
            self._rewrite_manual_protocol_violations(
                project_root=project_root,
                py_files=py_files,
                violations=manual_protocol_violations,
            )
            manual_protocol_violations = []
            for py_file in py_files:
                manual_protocol_violations.extend(
                    ManualProtocolDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )

        manual_typing_violations: list[
            NamespaceEnforcementModels.ManualTypingAliasViolation
        ] = []
        for py_file in py_files:
            manual_typing_violations.extend(
                ManualTypingAliasDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )

        if apply_changes and len(manual_typing_violations) > 0:
            self._rewrite_manual_typing_alias_violations(
                project_root=project_root,
                violations=manual_typing_violations,
                parse_failures=parse_failures,
            )
            manual_typing_violations = []
            for py_file in py_files:
                manual_typing_violations.extend(
                    ManualTypingAliasDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )

        compatibility_alias_violations: list[
            NamespaceEnforcementModels.CompatibilityAliasViolation
        ] = []
        for py_file in py_files:
            compatibility_alias_violations.extend(
                CompatibilityAliasDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )

        if apply_changes and len(compatibility_alias_violations) > 0:
            self._rewrite_compatibility_alias_violations(
                violations=compatibility_alias_violations,
                parse_failures=parse_failures,
            )
            compatibility_alias_violations = []
            for py_file in py_files:
                compatibility_alias_violations.extend(
                    CompatibilityAliasDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )

        return _new_project_enforcement_report(
            project=project_name,
            project_root=str(project_root),
            facade_statuses=facade_statuses,
            loose_objects=loose_objects,
            import_violations=import_violations,
            internal_import_violations=internal_import_violations,
            manual_protocol_violations=manual_protocol_violations,
            cyclic_imports=cyclic_imports,
            runtime_alias_violations=runtime_alias_violations,
            future_violations=future_violations,
            manual_typing_violations=manual_typing_violations,
            compatibility_alias_violations=compatibility_alias_violations,
            parse_failures=parse_failures,
            files_scanned=len(py_files),
        )

    @staticmethod
    def _preferred_file_name(*, family: str) -> str:
        """Get preferred file name for a family, derived from FAMILY_FILES."""
        pattern = _FACADE_FILE_PATTERNS.get(family, "utilities.py")
        return pattern.lstrip("*")

    @staticmethod
    def _base_import_for_family(*, family: str) -> str:
        """Get base import for a family, derived from FAMILY_SUFFIXES."""
        class_name = f"Flext{_FACADE_FAMILIES.get(family, 'Utilities')}"
        return f"from flext_core import {class_name}"

    @staticmethod
    def _base_class_for_family(*, family: str) -> str:
        """Get base class name for a family, derived from FAMILY_SUFFIXES."""
        return f"Flext{_FACADE_FAMILIES.get(family, 'Utilities')}"

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
        _ = file_path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)

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
                    _ = target_path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
                continue
            cls._write_missing_facade_file(
                file_path=target_path,
                family=status.family,
                class_name=class_name,
            )

    @staticmethod
    def _rewrite_import_alias_violations(*, py_files: list[Path]) -> None:
        """Rewrite deep imports to facade aliases using safe line-by-line regex.

        CRITICAL: Does NOT use ast.unparse() which destroys formatting/comments.
        Instead replaces only the matching import line, preserving all other content.
        Also skips __init__.py files per AGENTS.md §4.
        """
        deep_import_re = re.compile(
            r"^(\s*)from\s+(flext_core|flext_infra)\.\S+\s+import\s+.+$"
        )
        alias_map: dict[str, str] = {
            "flext_core": "from flext_core import c, m, r, t, u, p",
            "flext_infra": "from flext_infra import c, m, t, u, p",
        }
        for file_path in py_files:
            # __init__.py files ARE the facade — they must import submodules
            if file_path.name == "__init__.py":
                continue
            try:
                source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            except (OSError, UnicodeDecodeError):
                continue
            lines = source.splitlines(keepends=True)
            changed = False
            seen_replacements: set[str] = set()
            new_lines: list[str] = []
            for line in lines:
                match = deep_import_re.match(line.rstrip())
                if match:
                    indent = match.group(1)
                    pkg = match.group(2)
                    replacement = alias_map.get(pkg)
                    if replacement and replacement not in seen_replacements:
                        new_lines.append(f"{indent}{replacement}\n")
                        seen_replacements.add(replacement)
                        changed = True
                        continue
                    if replacement and replacement in seen_replacements:
                        # Duplicate deep import — drop the line
                        changed = True
                        continue
                new_lines.append(line)
            if changed:
                _ = file_path.write_text(
                    "".join(new_lines), encoding=c.Infra.Encoding.DEFAULT
                )

    @staticmethod
    def _rewrite_runtime_alias_violations(*, py_files: list[Path]) -> None:
        for file_path in py_files:
            expected = _FAMILY_EXPECTED_ALIAS.get(file_path.name)
            if expected is None:
                continue
            # Only rewrite facade files under src/ — skip tests/scripts/examples
            if "src" not in file_path.parts:
                continue
            alias_name, expected_suffix = expected
            parsed = _load_python_module(file_path)
            if parsed is None:
                continue
            source, tree = parsed.source, parsed.tree
            class_candidates = [
                node.name
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name.endswith(expected_suffix)
            ]
            if len(class_candidates) == 0:
                continue
            target_class = class_candidates[0]
            lines = source.splitlines()
            kept_lines = [
                line
                for line in lines
                if not line.strip().startswith(f"{alias_name} = ")
            ]
            kept_source = "\n".join(kept_lines).rstrip()
            rewritten = f"{kept_source}\n\n{alias_name} = {target_class}\n"
            _ = file_path.write_text(rewritten, encoding=c.Infra.Encoding.DEFAULT)

    @staticmethod
    def _rewrite_missing_future_annotations(*, py_files: list[Path]) -> None:
        for file_path in py_files:
            if file_path.name == "py.typed":
                continue
            try:
                source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            except (OSError, UnicodeDecodeError):
                continue
            if "from __future__ import annotations" in source:
                continue
            lines = source.splitlines()
            if len(lines) == 0:
                continue
            insert_idx = 0
            if lines[0].startswith('"""') or lines[0].startswith("'''"):
                quote = lines[0][:3]
                for idx in range(1, len(lines)):
                    if quote in lines[idx]:
                        insert_idx = idx + 1
                        break
            new_lines = (
                lines[:insert_idx]
                + ["", "from __future__ import annotations", ""]
                + lines[insert_idx:]
            )
            _ = file_path.write_text(
                "\n".join(new_lines).rstrip() + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )

    @classmethod
    def _rewrite_manual_protocol_violations(
        cls,
        *,
        project_root: Path,
        py_files: list[Path],
        violations: list[NamespaceEnforcementModels.ManualProtocolViolation],
    ) -> None:
        grouped_names: dict[Path, set[str]] = defaultdict(set)
        for violation in violations:
            grouped_names[Path(violation.file)].add(violation.name)
        protocol_moves: list[tuple[Path, Path, tuple[str, ...]]] = []
        for source_file, protocol_names in grouped_names.items():
            move_result = cls._move_protocol_classes_to_canonical_file(
                project_root=project_root,
                source_file=source_file,
                protocol_names=protocol_names,
            )
            if move_result is None:
                continue
            protocol_moves.append(move_result)
        if len(protocol_moves) > 0:
            cls._rewrite_moved_protocol_imports(
                project_root=project_root,
                py_files=py_files,
                protocol_moves=protocol_moves,
            )

    @classmethod
    def _move_protocol_classes_to_canonical_file(
        cls,
        *,
        project_root: Path,
        source_file: Path,
        protocol_names: set[str],
    ) -> tuple[Path, Path, tuple[str, ...]] | None:
        parsed = _load_python_module(source_file)
        if parsed is None:
            return None
        source, tree = parsed.source, parsed.tree

        class_nodes: list[ast.ClassDef] = []
        remove_ranges: list[tuple[int, int]] = []
        blocks: list[str] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            if stmt.name not in protocol_names:
                continue
            if not ManualProtocolDetector.is_protocol_class(stmt):
                continue
            block = ast.get_source_segment(source, stmt)
            if block is None:
                lines = source.splitlines()
                block = "\n".join(lines[stmt.lineno - 1 : stmt.end_lineno])
            if stmt.end_lineno is None:
                continue
            class_nodes.append(stmt)
            remove_ranges.append((stmt.lineno, stmt.end_lineno))
            blocks.append(block.strip("\n"))
        if len(class_nodes) == 0:
            return None

        target_file = cls._manual_protocol_target_file(
            project_root=project_root,
            source_file=source_file,
        )
        cls._append_protocol_blocks(
            project_root=project_root,
            target_file=target_file,
            blocks=blocks,
        )

        source_lines = source.splitlines()
        filtered_lines: list[str] = []
        for line_number, line_content in enumerate(source_lines, start=1):
            should_skip = any(
                start <= line_number <= end for start, end in remove_ranges
            )
            if should_skip:
                continue
            filtered_lines.append(line_content)
        rewritten = "\n".join(filtered_lines).rstrip()
        normalized = re.sub(r"\n{3,}", "\n\n", rewritten)
        _ = source_file.write_text(normalized + "\n", encoding=c.Infra.Encoding.DEFAULT)
        moved_names = tuple(sorted({node.name for node in class_nodes}))
        return (source_file, target_file, moved_names)

    @classmethod
    def _rewrite_manual_typing_alias_violations(
        cls,
        *,
        project_root: Path,
        violations: list[NamespaceEnforcementModels.ManualTypingAliasViolation],
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation],
    ) -> None:
        grouped_names: dict[Path, set[str]] = defaultdict(set)
        for violation in violations:
            grouped_names[Path(violation.file)].add(violation.name)
        for source_file, alias_names in grouped_names.items():
            cls._move_typing_aliases_to_canonical_file(
                project_root=project_root,
                source_file=source_file,
                alias_names=alias_names,
                parse_failures=parse_failures,
            )

    @classmethod
    def _move_typing_aliases_to_canonical_file(
        cls,
        *,
        project_root: Path,
        source_file: Path,
        alias_names: set[str],
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation],
    ) -> None:
        parsed = _load_python_module(
            source_file,
            stage="manual-typing-rewrite",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return
        source, tree = parsed.source, parsed.tree

        remove_ranges: list[tuple[int, int]] = []
        blocks: list[str] = []
        for stmt in tree.body:
            if isinstance(stmt, ast.TypeAlias):
                alias_name = stmt.name.id
                if alias_name not in alias_names:
                    continue
                if stmt.end_lineno is None:
                    continue
                block = ast.get_source_segment(source, stmt)
                if block is None:
                    lines = source.splitlines()
                    block = "\n".join(lines[stmt.lineno - 1 : stmt.end_lineno])
                remove_ranges.append((stmt.lineno, stmt.end_lineno))
                blocks.append(block.strip("\n"))
                continue
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.target.id not in alias_names:
                    continue
                if stmt.end_lineno is None:
                    continue
                annotation_src = ast.get_source_segment(source, stmt.annotation) or ""
                if "TypeAlias" not in annotation_src:
                    continue
                block = ast.get_source_segment(source, stmt)
                if block is None:
                    lines = source.splitlines()
                    block = "\n".join(lines[stmt.lineno - 1 : stmt.end_lineno])
                remove_ranges.append((stmt.lineno, stmt.end_lineno))
                blocks.append(block.strip("\n"))
        if len(blocks) == 0:
            return

        target_file = cls._manual_typings_target_file(
            project_root=project_root,
            source_file=source_file,
        )
        cls._append_typing_alias_blocks(target_file=target_file, blocks=blocks)

        source_lines = source.splitlines()
        filtered_lines: list[str] = []
        for line_number, line_content in enumerate(source_lines, start=1):
            should_skip = any(
                start <= line_number <= end for start, end in remove_ranges
            )
            if not should_skip:
                filtered_lines.append(line_content)
        rewritten = "\n".join(filtered_lines).rstrip()
        normalized = re.sub(r"\n{3,}", "\n\n", rewritten)
        _ = source_file.write_text(normalized + "\n", encoding=c.Infra.Encoding.DEFAULT)

    @staticmethod
    def _manual_typings_target_file(*, project_root: Path, source_file: Path) -> Path:
        parts = source_file.parts
        if "src" in parts:
            src_index = parts.index("src")
            if src_index + 1 < len(parts):
                package_name = parts[src_index + 1]
                return (
                    project_root
                    / c.Infra.Paths.DEFAULT_SRC_DIR
                    / package_name
                    / "typings.py"
                )
        return source_file.parent / "typings.py"

    @staticmethod
    def _append_typing_alias_blocks(*, target_file: Path, blocks: list[str]) -> None:
        if len(blocks) == 0:
            return
        target_source = (
            target_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
            if target_file.exists()
            else ""
        )
        updated = target_source
        if "from __future__ import annotations" not in updated:
            updated = "from __future__ import annotations\n\n" + updated.lstrip("\n")
        merged_blocks = "\n\n".join(blocks)
        if (
            "TypeAlias" in merged_blocks
            and "from typing import TypeAlias" not in updated
        ):
            updated = updated.rstrip() + "\n\nfrom typing import TypeAlias\n"
        updated = updated.rstrip() + "\n\n" + merged_blocks + "\n"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        _ = target_file.write_text(updated, encoding=c.Infra.Encoding.DEFAULT)

    @staticmethod
    def _rewrite_compatibility_alias_violations(
        *,
        violations: list[NamespaceEnforcementModels.CompatibilityAliasViolation],
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation],
    ) -> None:
        grouped: dict[Path, dict[str, str]] = defaultdict(dict)
        for violation in violations:
            grouped[Path(violation.file)][violation.alias_name] = violation.target_name
        for file_path, alias_map in grouped.items():
            parsed = _load_python_module(
                file_path,
                stage="compatibility-alias-rewrite",
                parse_failures=parse_failures,
            )
            if parsed is None:
                continue
            source = parsed.source
            assignment_lines: set[int] = set()
            for stmt in parsed.tree.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if not isinstance(stmt.value, ast.Name):
                    continue
                if target.id in alias_map and stmt.value.id == alias_map[target.id]:
                    assignment_lines.add(stmt.lineno)
            if len(assignment_lines) == 0:
                continue

            kept_lines = [
                line
                for idx, line in enumerate(source.splitlines(keepends=True), start=1)
                if idx not in assignment_lines
            ]
            kept_source = "".join(kept_lines)
            line_buffer = kept_source.splitlines(keepends=True)

            replacements_by_line: dict[int, list[tuple[int, int, str]]] = defaultdict(
                list
            )
            token_generator = tokenize.generate_tokens(StringIO(kept_source).readline)
            for tok in token_generator:
                if tok.type != token.NAME:
                    continue
                replacement = alias_map.get(tok.string)
                if replacement is None:
                    continue
                start_line, start_col = tok.start
                end_line, end_col = tok.end
                if start_line != end_line:
                    continue
                replacements_by_line[start_line - 1].append((
                    start_col,
                    end_col,
                    replacement,
                ))

            for line_idx, replacements in replacements_by_line.items():
                if line_idx < 0 or line_idx >= len(line_buffer):
                    continue
                line_text = line_buffer[line_idx]
                for start_col, end_col, replacement in sorted(
                    replacements,
                    key=lambda item: item[0],
                    reverse=True,
                ):
                    line_text = (
                        line_text[:start_col] + replacement + line_text[end_col:]
                    )
                line_buffer[line_idx] = line_text

            rewritten = "".join(line_buffer)
            if rewritten != source:
                _ = file_path.write_text(rewritten, encoding=c.Infra.Encoding.DEFAULT)

    @staticmethod
    def _manual_protocol_target_file(*, project_root: Path, source_file: Path) -> Path:
        parts = source_file.parts
        if "src" in parts:
            src_index = parts.index("src")
            if src_index + 1 < len(parts):
                package_name = parts[src_index + 1]
                return (
                    project_root
                    / c.Infra.Paths.DEFAULT_SRC_DIR
                    / package_name
                    / "protocols.py"
                )
        return source_file.parent / "protocols.py"

    @classmethod
    def _append_protocol_blocks(
        cls,
        *,
        project_root: Path,
        target_file: Path,
        blocks: list[str],
    ) -> None:
        if len(blocks) == 0:
            return
        project_name = project_root.name
        class_stem = NamespaceFacadeScanner.project_class_stem(
            project_name=project_name
        )
        protocols_class = f"{class_stem}Protocols"
        if target_file.exists():
            target_source = target_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
        else:
            target_source = ""

        updated = target_source
        if "from __future__ import annotations" not in updated:
            updated = "from __future__ import annotations\n\n" + updated.lstrip("\n")
        if "from typing import Protocol" not in updated:
            updated = updated.rstrip() + "\n\nfrom typing import Protocol\n"
        if (
            f"class {protocols_class}(" not in updated
            and f"class {protocols_class}:" not in updated
        ):
            updated = updated.rstrip() + f"\n\nclass {protocols_class}:\n    pass\n"
        alias_line = f"p = {protocols_class}"
        if alias_line not in updated:
            updated = updated.rstrip() + f"\n\n{alias_line}\n"

        for block in blocks:
            class_header = block.splitlines()[0].strip()
            if class_header in updated:
                continue
            updated = updated.rstrip() + "\n\n" + block + "\n"

        target_file.parent.mkdir(parents=True, exist_ok=True)
        _ = target_file.write_text(
            updated.rstrip() + "\n", encoding=c.Infra.Encoding.DEFAULT
        )

    @staticmethod
    def _rewrite_moved_protocol_imports(
        *,
        project_root: Path,
        py_files: list[Path],
        protocol_moves: list[tuple[Path, Path, tuple[str, ...]]],
    ) -> None:
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR

        def _module_path(file_path: Path) -> str:
            try:
                relative = file_path.relative_to(src_dir)
            except ValueError:
                return ""
            parts = list(relative.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)

        source_target_names: list[tuple[str, str, set[str]]] = []
        for source_file, target_file, moved_names in protocol_moves:
            source_module = _module_path(source_file)
            target_module = _module_path(target_file)
            if not source_module or not target_module or source_module == target_module:
                continue
            source_target_names.append((source_module, target_module, set(moved_names)))
        if len(source_target_names) == 0:
            return
        for py_file in py_files:
            parsed = _load_python_module(py_file)
            if parsed is None:
                continue
            source = parsed.source
            tree = parsed.tree
            new_lines = source.splitlines(keepends=True)
            changed = False
            for stmt in tree.body:
                if not isinstance(stmt, ast.ImportFrom):
                    continue
                if stmt.module is None:
                    continue
                for source_module, target_module, moved_names in source_target_names:
                    if stmt.module != source_module:
                        continue
                    imported = [alias.name for alias in stmt.names if alias.name != "*"]
                    if not any(name in moved_names for name in imported):
                        continue
                    if stmt.lineno <= 0 or stmt.lineno > len(new_lines):
                        continue
                    line_index = stmt.lineno - 1
                    line_text = new_lines[line_index]
                    new_lines[line_index] = line_text.replace(
                        source_module, target_module
                    )
                    changed = True
            if changed:
                _ = py_file.write_text(
                    "".join(new_lines), encoding=c.Infra.Encoding.DEFAULT
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
            f"Internal imports: {report.total_internal_import_violations}",
            f"Cyclic imports: {report.total_cyclic_imports}",
            f"Runtime alias violations: {report.total_runtime_alias_violations}",
            f"Missing __future__: {report.total_future_violations}",
            f"Manual protocols: {report.total_manual_protocol_violations}",
            f"Manual typing aliases: {report.total_manual_typing_violations}",
            f"Compatibility aliases: {report.total_compatibility_alias_violations}",
            f"Parse failures: {report.total_parse_failures}",
            "",
        ]
        for proj in report.projects:
            missing = [s for s in proj.facade_statuses if not s.exists]
            has_violations = (
                missing
                or proj.loose_objects
                or proj.import_violations
                or proj.internal_import_violations
                or proj.runtime_alias_violations
                or proj.future_violations
                or proj.manual_protocol_violations
                or proj.manual_typing_violations
                or proj.compatibility_alias_violations
                or proj.parse_failures
            )
            if not has_violations:
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
            if proj.internal_import_violations:
                lines.append(
                    f"  Internal imports: {len(proj.internal_import_violations)}"
                )
                lines.extend(
                    f"    {iv.file}:{iv.line} {iv.current_import} ({iv.detail})"
                    for iv in proj.internal_import_violations[
                        :_MAX_RENDERED_IMPORT_VIOLATIONS
                    ]
                )
                if (
                    len(proj.internal_import_violations)
                    > _MAX_RENDERED_IMPORT_VIOLATIONS
                ):
                    lines.append(
                        f"    ... and {len(proj.internal_import_violations) - _MAX_RENDERED_IMPORT_VIOLATIONS} more"
                    )
            if proj.cyclic_imports:
                lines.append(f"  Cyclic imports: {len(proj.cyclic_imports)}")
                lines.extend(
                    f"    Cycle: {' -> '.join(ci.cycle)}" for ci in proj.cyclic_imports
                )
            if proj.runtime_alias_violations:
                lines.append(
                    f"  Runtime alias violations: {len(proj.runtime_alias_violations)}"
                )
                lines.extend(
                    f"    {rv.file} [{rv.kind}] alias='{rv.alias}' {rv.detail}"
                    for rv in proj.runtime_alias_violations
                )
            if proj.future_violations:
                lines.append(
                    f"  Missing __future__ annotations: {len(proj.future_violations)}"
                )
                lines.extend(
                    f"    {fv.file}"
                    for fv in proj.future_violations[:_MAX_RENDERED_LOOSE_OBJECTS]
                )
                if len(proj.future_violations) > _MAX_RENDERED_LOOSE_OBJECTS:
                    lines.append(
                        f"    ... and {len(proj.future_violations) - _MAX_RENDERED_LOOSE_OBJECTS} more"
                    )
            if proj.manual_protocol_violations:
                lines.append(
                    f"  Manual protocols: {len(proj.manual_protocol_violations)}"
                )
                lines.extend(
                    f"    {pv.file}:{pv.line} {pv.name}"
                    for pv in proj.manual_protocol_violations[
                        :_MAX_RENDERED_LOOSE_OBJECTS
                    ]
                )
                if len(proj.manual_protocol_violations) > _MAX_RENDERED_LOOSE_OBJECTS:
                    lines.append(
                        f"    ... and {len(proj.manual_protocol_violations) - _MAX_RENDERED_LOOSE_OBJECTS} more"
                    )
            if proj.manual_typing_violations:
                lines.append(
                    f"  Manual typing aliases: {len(proj.manual_typing_violations)}"
                )
                lines.extend(
                    f"    {tv.file}:{tv.line} {tv.name}"
                    for tv in proj.manual_typing_violations[
                        :_MAX_RENDERED_LOOSE_OBJECTS
                    ]
                )
                if len(proj.manual_typing_violations) > _MAX_RENDERED_LOOSE_OBJECTS:
                    lines.append(
                        f"    ... and {len(proj.manual_typing_violations) - _MAX_RENDERED_LOOSE_OBJECTS} more"
                    )
            if proj.compatibility_alias_violations:
                lines.append(
                    f"  Compatibility aliases: {len(proj.compatibility_alias_violations)}"
                )
                lines.extend(
                    f"    {cv.file}:{cv.line} {cv.alias_name}={cv.target_name}"
                    for cv in proj.compatibility_alias_violations[
                        :_MAX_RENDERED_LOOSE_OBJECTS
                    ]
                )
                if (
                    len(proj.compatibility_alias_violations)
                    > _MAX_RENDERED_LOOSE_OBJECTS
                ):
                    lines.append(
                        f"    ... and {len(proj.compatibility_alias_violations) - _MAX_RENDERED_LOOSE_OBJECTS} more"
                    )
            if proj.parse_failures:
                lines.append(f"  Parse failures: {len(proj.parse_failures)}")
                lines.extend(
                    f"    {pf.file} [{pf.stage}] {pf.error_type}: {pf.detail}"
                    for pf in proj.parse_failures[:_MAX_RENDERED_LOOSE_OBJECTS]
                )
                if len(proj.parse_failures) > _MAX_RENDERED_LOOSE_OBJECTS:
                    lines.append(
                        f"    ... and {len(proj.parse_failures) - _MAX_RENDERED_LOOSE_OBJECTS} more"
                    )
            lines.append("")
        return "\n".join(lines) + "\n"


__all__ = [
    "CompatibilityAliasDetector",
    "CyclicImportDetector",
    "FlextInfraNamespaceEnforcer",
    "FutureAnnotationsDetector",
    "ImportAliasDetector",
    "InternalImportDetector",
    "LooseObjectDetector",
    "ManualProtocolDetector",
    "ManualTypingAliasDetector",
    "NamespaceEnforcementModels",
    "NamespaceFacadeScanner",
    "RuntimeAliasDetector",
]
