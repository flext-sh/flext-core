"""Cross-project dependency analysis using LibCST and ast-grep."""

from __future__ import annotations

import ast
import sys
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import ClassVar, override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, m, p, u
from flext_infra.refactor._models_namespace_enforcer import (
    FlextInfraNamespaceEnforcerModels as nem,
)


class ImportCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.imported_modules: set[str] = set()
        self.imported_symbols: set[str] = set()

    @override
    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            root = self._module_root(alias.name)
            if root:
                self.imported_modules.add(root)

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None or node.relative:
            return
        root = self._module_root(node.module)
        if root:
            self.imported_modules.add(root)
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            sym = self._imported_symbol(alias.name)
            if sym:
                self.imported_symbols.add(sym)

    def _module_root(self, node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parts: list[str] = []
            cur: cst.BaseExpression | None = node
            while isinstance(cur, cst.Attribute):
                parts.append(cur.attr.value)
                cur = cur.value
            if isinstance(cur, cst.Name):
                parts.append(cur.value)
                return parts[-1]
        return None

    def _imported_symbol(self, node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            return node.attr.value
        return None


class DependencyAnalyzer:
    """Build inter-project import graphs from workspace source trees."""

    def __init__(self, workspace_root: Path) -> None:
        """Initialize analyzer for the given workspace root."""
        super().__init__()
        self._workspace_root = workspace_root.resolve()
        self._stdlib_roots = set(sys.stdlib_module_names)
        self._projects = self._discover_projects()
        self._pkg_index = self._build_package_index(self._projects)
        self._graph_cache: dict[str, list[str]] | None = None

    def build_import_graph(self) -> r[dict[str, list[str]]]:
        """Build and cache the inter-project import graph."""
        if self._graph_cache is not None:
            return r[dict[str, list[str]]].ok(self._graph_cache)
        graph: dict[str, set[str]] = {p.name: set() for p in self._projects}
        for project in self._projects:
            files = self._find_import_candidate_files(project)
            for fp in files:
                parsed = self._parse_imports(fp)
                if parsed.is_failure:
                    continue
                file_data: m.Infra.Refactor.FileImportData = parsed.value
                for mod_root in file_data.imported_modules:
                    if mod_root in self._stdlib_roots:
                        continue
                    dep = self._pkg_index.get(mod_root)
                    if dep and dep != project.name:
                        graph[project.name].add(dep)
        ordered = {k: sorted(v) for k, v in sorted(graph.items())}
        self._graph_cache = ordered
        return r[dict[str, list[str]]].ok(ordered)

    def find_consumers(self, class_name: str) -> r[list[Path]]:
        """Find all files importing the given class name."""
        consumers: set[Path] = set()
        for project in self._projects:
            for fp in self._find_import_candidate_files(project):
                parsed = self._parse_imports(fp)
                if parsed.is_failure:
                    continue
                file_data = parsed.value
                if class_name in file_data.imported_symbols:
                    consumers.add(fp)
        return r[list[Path]].ok(sorted(consumers))

    def determine_transformation_order(self) -> r[list[str]]:
        """Return topologically sorted project order."""
        graph_result = self.build_import_graph()
        if graph_result.is_failure:
            return r[list[str]].fail(graph_result.error or "graph build failed")
        graph: dict[str, list[str]] = graph_result.value
        if not graph:
            return r[list[str]].ok([])
        try:
            sorter: TopologicalSorter[str] = TopologicalSorter(graph)
            return r[list[str]].ok(list(sorter.static_order()))
        except CycleError:
            return r[list[str]].ok(sorted(graph))

    def _discover_projects(self) -> list[m.Infra.Refactor.ProjectInfo]:
        projects: list[m.Infra.Refactor.ProjectInfo] = []
        for candidate in sorted(self._workspace_root.iterdir()):
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue
            src = candidate / c.Infra.Paths.DEFAULT_SRC_DIR
            if not src.is_dir():
                continue
            projects.append(
                m.Infra.Refactor.ProjectInfo(
                    name=candidate.name,
                    path=candidate,
                    src_path=src,
                    package_roots=self._discover_package_roots(src),
                ),
            )
        return projects

    def _discover_package_roots(self, src_path: Path) -> set[str]:
        roots: set[str] = set()
        for pkg_dir in src_path.iterdir():
            if pkg_dir.name.startswith("."):
                continue
            if pkg_dir.is_dir() and (pkg_dir / c.Infra.Files.INIT_PY).is_file():
                roots.add(pkg_dir.name)
            elif (
                pkg_dir.is_file()
                and pkg_dir.suffix == c.Infra.Extensions.PYTHON
                and (pkg_dir.stem != "__init__")
            ):
                roots.add(pkg_dir.stem)
        return roots

    def _build_package_index(
        self,
        projects: list[m.Infra.Refactor.ProjectInfo],
    ) -> dict[str, str]:
        idx: dict[str, str] = {}
        for proj in projects:
            for pkg in proj.package_roots:
                _ = idx.setdefault(pkg, proj.name)
        return idx

    def _find_import_candidate_files(
        self,
        project: m.Infra.Refactor.ProjectInfo,
    ) -> list[Path]:
        grep_files = self._scan_import_files_with_ast_grep(project.src_path)
        if grep_files.is_success and grep_files.value:
            path_set: set[Path] = grep_files.value
            return sorted(path_set)
        files_result = u.Infra.iter_python_files(
            workspace_root=self._workspace_root,
            project_roots=[project.path],
            include_tests=False,
            include_examples=False,
            include_scripts=False,
            src_dirs=frozenset({"src"}),
        )
        return files_result.fold(
            on_failure=lambda _: list[Path](),
            on_success=lambda v: v,
        )

    def _scan_import_files_with_ast_grep(self, src_path: Path) -> r[set[Path]]:
        files: set[Path] = set()
        for pattern in ("import $MODULE", "from $MODULE import $$$"):
            result = self._run_ast_grep(src_path, pattern)
            if result.is_failure:
                return r[set[Path]].fail(result.error or "ast-grep failed")
            entries: list[m.Infra.Refactor.AstGrepMatchEnvelope] = result.value
            for entry in entries:
                fp = Path(entry.file)
                if not fp.is_absolute():
                    fp = (src_path / fp).resolve()
                if fp.suffix == c.Infra.Extensions.PYTHON:
                    files.add(fp)
        return r[set[Path]].ok(files)

    def _run_ast_grep(
        self,
        src_path: Path,
        pattern: str,
    ) -> r[list[m.Infra.Refactor.AstGrepMatchEnvelope]]:
        cmd = [
            "sg",
            "--pattern",
            pattern,
            "--lang",
            c.Infra.Toml.PYTHON,
            "--json",
            str(src_path),
        ]
        capture = u.Infra.capture(cmd)
        if capture.is_failure:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].fail(
                capture.error or "capture failed",
            )
        if not capture.value:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].ok([])
        try:
            json_raw: str | bytes | bytearray = capture.value
            envelopes: list[m.Infra.Refactor.AstGrepMatchEnvelope] = TypeAdapter(
                list[m.Infra.Refactor.AstGrepMatchEnvelope]
            ).validate_json(
                json_raw,
            )
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].ok(envelopes)
        except ValidationError as exc:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].fail(str(exc))

    def _parse_imports(self, file_path: Path) -> r[m.Infra.Refactor.FileImportData]:
        tree = u.Infra.parse_module_cst(file_path)
        if tree is None:
            return r[m.Infra.Refactor.FileImportData].fail(f"{file_path}: parse_failed")
        col = ImportCollector()
        _ = tree.visit(col)
        return r[m.Infra.Refactor.FileImportData].ok(
            m.Infra.Refactor.FileImportData(
                imported_modules=col.imported_modules,
                imported_symbols=col.imported_symbols,
            ),
        )


class NamespaceFacadeScanner:
    """Scan projects for namespace facade class patterns."""

    @classmethod
    def scan_project(
        cls,
        *,
        project_root: Path,
        project_name: str,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.FacadeStatus]:
        """Scan a project for namespace facade classes and return their status."""
        results: list[nem.FacadeStatus] = []
        class_stem = cls.project_class_stem(project_name=project_name)
        for family, suffix in c.Infra.Refactor.NAMESPACE_FACADE_FAMILIES.items():
            expected_class = f"{class_stem}{suffix}"
            found_class, found_file, symbol_count = cls._find_facade_class(
                project_root=project_root,
                family=family,
                expected_class=expected_class,
                suffix=suffix,
                _parse_failures=parse_failures,
            )
            results.append(
                nem.FacadeStatus.create(
                    family=family,
                    exists=bool(found_class),
                    class_name=found_class,
                    file=found_file,
                    symbol_count=symbol_count,
                ),
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
        _parse_failures: list[nem.ParseFailureViolation] | None,
    ) -> tuple[str, str, int]:
        file_pattern = c.Infra.Refactor.NAMESPACE_FACADE_FILE_PATTERNS[family]
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return ("", "", 0)
        for file_path in src_dir.rglob(file_pattern):
            tree = u.Infra.parse_module_ast(file_path)
            if tree is None:
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
        """Derive the class name stem from a project name."""
        normalized = project_name.strip().lower().replace("_", "-")
        if normalized == "flext-core":
            return "Flext"
        if normalized.startswith("flext-"):
            tail = normalized.removeprefix("flext-")
            parts = [p for p in tail.split("-") if p]
            return "Flext" + "".join(p.capitalize() for p in parts)
        parts = [p for p in normalized.split("-") if p]
        return "".join(p.capitalize() for p in parts) if parts else ""


class LooseObjectDetector(p.Infra.Scanner):
    """Detect loose top-level objects that should be inside namespace classes."""

    ALLOWED_TOP_LEVEL = frozenset({"__all__", "__version__", "__version_info__"})

    def __init__(
        self,
        *,
        project_name: str,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._project_name = project_name
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            project_name=self._project_name,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=violation.line,
                    message=(
                        f"Loose {violation.kind} '{violation.name}' outside namespace"
                    ),
                    severity="error",
                    rule_id="namespace.loose_object",
                )
                for violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        project_name: str,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.LooseObjectViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            project_name=project_name,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        project_name: str,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.LooseObjectViolation]:
        """Scan a file for loose top-level objects outside namespace classes."""
        if file_path.name in c.Infra.Refactor.NAMESPACE_PROTECTED_FILES:
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_SETTINGS_FILE_NAMES:
            return []
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return []
        namespace_classes = cls._find_namespace_classes(tree=tree)
        violations: list[nem.LooseObjectViolation] = []
        class_stem = NamespaceFacadeScanner.project_class_stem(
            project_name=project_name,
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
    ) -> nem.LooseObjectViolation | None:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return None
        if isinstance(stmt, ast.If):
            return None
        if isinstance(stmt, ast.ClassDef):
            if stmt.name in namespace_classes:
                return None
            return None
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name.startswith("__") and stmt.name.endswith("__"):
                return None
            if stmt.name.startswith("_"):
                return None
            return nem.LooseObjectViolation.create(
                file=str(file_path),
                line=stmt.lineno,
                name=stmt.name,
                kind="function",
                suggestion=f"{class_stem}Utilities",
            )
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if name in cls.ALLOWED_TOP_LEVEL:
                return None
            if name.startswith("_"):
                return None
            if c.Infra.Refactor.NAMESPACE_CONSTANT_PATTERN.match(name):
                return nem.LooseObjectViolation.create(
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
                if len(name) <= c.Infra.Refactor.NAMESPACE_MIN_ALIAS_LENGTH:
                    return None
                if name.startswith("_"):
                    return None
                if c.Infra.Refactor.NAMESPACE_CONSTANT_PATTERN.match(name):
                    return nem.LooseObjectViolation.create(
                        file=str(file_path),
                        line=stmt.lineno,
                        name=name,
                        kind="constant",
                        suggestion=f"{class_stem}Constants",
                    )
        if isinstance(stmt, ast.TypeAlias):
            name = stmt.name.id if hasattr(stmt.name, "id") else ""
            if name and name not in cls.ALLOWED_TOP_LEVEL:
                return nem.LooseObjectViolation.create(
                    file=str(file_path),
                    line=stmt.lineno,
                    name=name,
                    kind="typealias",
                    suggestion=f"{class_stem}Types",
                )
        return None

    @staticmethod
    def _find_namespace_classes(*, tree: ast.Module) -> set[str]:
        classes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for suffix in c.Infra.Refactor.NAMESPACE_FACADE_FAMILIES.values():
                    if node.name.endswith(suffix):
                        classes.add(node.name)
                        break
        return classes


class ImportAliasDetector(p.Infra.Scanner):
    """Detect deep import paths that should use top-level aliases."""

    ALIAS_MODULES: ClassVar[dict[str, str]] = {
        "flext_core": "from flext_core import c, m, r, t, u, p",
        "flext_infra": "from flext_infra import c, m, t, u, p",
    }

    def __init__(
        self,
        *,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=violation.line,
                    message=(
                        f"Deep import '{violation.current_import}' should use "
                        f"'{violation.suggested_import}'"
                    ),
                    severity="error",
                    rule_id="namespace.import_alias",
                )
                for violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.ImportAliasViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.ImportAliasViolation]:
        """Scan a file for deep import paths that should use aliases."""
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return []
        violations: list[nem.ImportAliasViolation] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.ImportFrom):
                continue
            if stmt.module is None:
                continue
            if file_path.name == "__init__.py":
                continue
            for prefix, suggestion in cls.ALIAS_MODULES.items():
                if stmt.module.startswith(prefix + "."):
                    import_names = (
                        ", ".join(
                            alias.name for alias in stmt.names if alias.name != "*"
                        )
                        if not any(alias.name == "*" for alias in stmt.names)
                        else "*"
                    )
                    current = f"from {stmt.module} import {import_names}"
                    violations.append(
                        nem.ImportAliasViolation.create(
                            file=str(file_path),
                            line=stmt.lineno,
                            current_import=current,
                            suggested_import=suggestion,
                        ),
                    )
        return violations


class InternalImportDetector(p.Infra.Scanner):
    """Detect imports of private modules or symbols across boundaries."""

    def __init__(
        self,
        *,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=violation.line,
                    message=(
                        f"Internal import '{violation.current_import}': "
                        f"{violation.detail}"
                    ),
                    severity="error",
                    rule_id="namespace.internal_import",
                )
                for violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.InternalImportViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.InternalImportViolation]:
        """Scan a file for private module or symbol imports."""
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return []
        violations: list[nem.InternalImportViolation] = []
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
                nem.InternalImportViolation.create(
                    file=str(file_path),
                    line=stmt.lineno,
                    current_import=current_import,
                    detail=detail,
                ),
            )
        return violations


class ManualProtocolDetector(p.Infra.Scanner):
    """Detect Protocol classes defined outside canonical protocol files."""

    CANONICAL_FILE_NAMES = c.Infra.Refactor.NAMESPACE_CANONICAL_PROTOCOL_FILES
    CANONICAL_DIR_NAME = c.Infra.Refactor.NAMESPACE_CANONICAL_PROTOCOL_DIR

    def __init__(
        self,
        *,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=violation.line,
                    message=(
                        f"Protocol class '{violation.name}' must be centralized "
                        f"({violation.suggestion})"
                    ),
                    severity="error",
                    rule_id="namespace.manual_protocol",
                )
                for violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.ManualProtocolViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.ManualProtocolViolation]:
        """Scan a file for Protocol classes outside canonical locations."""
        in_canonical_file = file_path.name in cls.CANONICAL_FILE_NAMES
        in_canonical_dir = cls.CANONICAL_DIR_NAME in file_path.parts
        if in_canonical_file or in_canonical_dir:
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_PROTECTED_FILES:
            return []
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return []
        violations: list[nem.ManualProtocolViolation] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            if cls.is_protocol_class(stmt):
                violations.append(
                    nem.ManualProtocolViolation.create(
                        file=str(file_path),
                        line=stmt.lineno,
                        name=stmt.name,
                    ),
                )
        return violations

    @staticmethod
    def is_protocol_class(node: ast.ClassDef) -> bool:
        """Return whether the class definition inherits from Protocol."""
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


class CyclicImportDetector:
    """Detect cyclic import dependencies within a project."""

    # NOTE: CyclicImportDetector operates at project level, not file level — does not implement Scanner

    @classmethod
    def scan_project(
        cls,
        *,
        project_root: Path,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.CyclicImportViolation]:
        """Scan a project for cyclic import dependencies."""
        scan_dirs = [
            project_root / directory_name
            for directory_name in c.Infra.Refactor.MRO_SCAN_DIRECTORIES
            if (project_root / directory_name).is_dir()
        ]
        if len(scan_dirs) == 0:
            return []
        graph: dict[str, set[str]] = {}
        file_map: dict[str, str] = {}
        package_roots = cls._discover_package_roots(scan_dirs=scan_dirs)
        for scan_dir in scan_dirs:
            for py_file in u.Infra.iter_directory_python_files(scan_dir):
                base_module_name = cls._file_to_module(
                    file_path=py_file, src_dir=scan_dir
                )
                module_name = cls._module_name_for_scan_dir(
                    scan_dir=scan_dir,
                    base_module_name=base_module_name,
                )
                if not module_name:
                    continue
                if module_name not in file_map:
                    file_map[module_name] = str(py_file)
                graph.setdefault(module_name, set())
                tree = u.Infra.parse_module_ast(py_file)
                if tree is None:
                    continue
                for stmt in tree.body:
                    if isinstance(stmt, ast.Import):
                        for alias in stmt.names:
                            imported = alias.name
                            root_pkg = imported.split(".")[0]
                            if root_pkg in package_roots:
                                graph[module_name].add(imported)
                    if isinstance(stmt, ast.ImportFrom) and stmt.module:
                        imported = stmt.module
                        root_pkg = imported.split(".")[0]
                        if root_pkg in package_roots:
                            graph[module_name].add(imported)
        violations: list[nem.CyclicImportViolation] = []
        try:
            _ = list(TopologicalSorter(graph).static_order())
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
                    nem.CyclicImportViolation.create(
                        cycle=normalized_cycle,
                        files=cycle_files,
                    ),
                )
        return violations

    @staticmethod
    def _discover_package_roots(*, scan_dirs: list[Path]) -> set[str]:
        roots: set[str] = set()
        for scan_dir in scan_dirs:
            if (scan_dir / "__init__.py").is_file():
                roots.add(scan_dir.name)
            for entry in scan_dir.iterdir():
                if entry.name.startswith(".") or entry.name == "__pycache__":
                    continue
                if entry.is_dir() and (entry / "__init__.py").is_file():
                    roots.add(entry.name)
                elif (
                    entry.is_file()
                    and entry.suffix == c.Infra.Extensions.PYTHON
                    and entry.stem != "__init__"
                ):
                    roots.add(entry.stem)
        return roots

    @staticmethod
    def _module_name_for_scan_dir(*, scan_dir: Path, base_module_name: str) -> str:
        if not base_module_name:
            return ""
        if scan_dir.name == c.Infra.Paths.DEFAULT_SRC_DIR:
            return base_module_name
        if (scan_dir / "__init__.py").is_file():
            return f"{scan_dir.name}.{base_module_name}"
        return base_module_name

    @staticmethod
    def _file_to_module(*, file_path: Path, src_dir: Path) -> str:
        try:
            rel = file_path.relative_to(src_dir)
        except ValueError:
            return ""
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else ""


class RuntimeAliasDetector(p.Infra.Scanner):
    """Detect missing or duplicate runtime alias assignments."""

    def __init__(
        self,
        *,
        project_name: str,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._project_name = project_name
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            project_name=self._project_name,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=violation.line if violation.line > 0 else 1,
                    message=(
                        f"Runtime alias '{violation.alias}' {violation.kind}: "
                        f"{violation.detail}"
                    ),
                    severity="error",
                    rule_id="namespace.runtime_alias",
                )
                for violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        project_name: str,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.RuntimeAliasViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            project_name=project_name,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        project_name: str,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.RuntimeAliasViolation]:
        """Scan a file for missing or duplicate runtime alias assignments."""
        if file_path.name not in c.Infra.Refactor.NAMESPACE_FILE_TO_FAMILY:
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_PROTECTED_FILES:
            return []
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return []
        violations: list[nem.RuntimeAliasViolation] = []
        _ = project_name
        family = cls._family_for_file(file_name=file_path.name)
        if not family:
            return []
        alias_assignments: list[tuple[int, str, str]] = []
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
                nem.RuntimeAliasViolation.create(
                    file=str(file_path),
                    kind="missing",
                    alias=expected_alias,
                    detail=f"No '{expected_alias} = ...' assignment found",
                ),
            )
        elif len(matches) > 1:
            violations.append(
                nem.RuntimeAliasViolation.create(
                    file=str(file_path),
                    line=matches[1][0],
                    kind="duplicate",
                    alias=expected_alias,
                    detail=f"Duplicate alias assignment at lines {', '.join(str(mv[0]) for mv in matches)}",
                ),
            )
        return violations

    @staticmethod
    def _family_for_file(*, file_name: str) -> str:
        return c.Infra.Refactor.NAMESPACE_FILE_TO_FAMILY.get(file_name, "")


class FutureAnnotationsDetector(p.Infra.Scanner):
    """Detect Python files missing the future annotations import."""

    def __init__(
        self,
        *,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=1,
                    message="Missing 'from __future__ import annotations'",
                    severity="error",
                    rule_id="namespace.future_annotations",
                )
                for _violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.FutureAnnotationsViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.FutureAnnotationsViolation]:
        """Scan a file for missing future annotations import."""
        if file_path.name == "py.typed":
            return []
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return []
        if len(tree.body) == 0:
            return []
        if (
            len(tree.body) == 1
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
        ):
            return []
        for stmt in tree.body:
            if (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module == "__future__"
                and any(alias.name == "annotations" for alias in stmt.names)
            ):
                return []
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                break
        return [
            nem.FutureAnnotationsViolation.create(
                file=str(file_path),
            ),
        ]


class ManualTypingAliasDetector(p.Infra.Scanner):
    """Detect type aliases defined outside canonical typings files."""

    def __init__(
        self,
        *,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=violation.line,
                    message=f"Typing alias '{violation.name}': {violation.detail}",
                    severity="error",
                    rule_id="namespace.manual_typing_alias",
                )
                for violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.ManualTypingAliasViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.ManualTypingAliasViolation]:
        """Scan a file for type aliases outside canonical locations."""
        if file_path.suffix != ".py":
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_CANONICAL_TYPINGS_FILES:
            return []
        if c.Infra.Refactor.NAMESPACE_CANONICAL_TYPINGS_DIR in file_path.parts:
            return []
        parsed = FlextInfraRefactorDependencyAnalyzerFacade.load_python_module(
            file_path,
            stage="manual-typing-alias-scan",
            parse_failures=_parse_failures,
        )
        if parsed is None:
            return []
        source = parsed.source
        tree = parsed.tree
        violations: list[nem.ManualTypingAliasViolation] = []
        for stmt in tree.body:
            if isinstance(stmt, ast.TypeAlias):
                alias_name = stmt.name.id
                violations.append(
                    nem.ManualTypingAliasViolation.create(
                        file=str(file_path),
                        line=stmt.lineno,
                        name=alias_name,
                        detail="PEP695 alias must be centralized under typings scope",
                    ),
                )
                continue
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                annotation_src = ast.get_source_segment(source, stmt.annotation) or ""
                if "TypeAlias" in annotation_src:
                    violations.append(
                        nem.ManualTypingAliasViolation.create(
                            file=str(file_path),
                            line=stmt.lineno,
                            name=stmt.target.id,
                            detail="TypeAlias assignment must be centralized under typings scope",
                        ),
                    )
        return violations


class CompatibilityAliasDetector(p.Infra.Scanner):
    """Detect compatibility alias assignments that may be removable."""

    def __init__(
        self,
        *,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> None:
        """Initialize scanner with project configuration."""
        super().__init__()
        self._parse_failures = parse_failures

    @override
    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        """Scan a file and return protocol-standardized scan output."""
        violations = type(self).scan_file_impl(
            file_path=file_path,
            _parse_failures=self._parse_failures,
        )
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=violation.line,
                    message=(
                        f"Compatibility alias '{violation.alias_name}' -> "
                        f"'{violation.target_name}'"
                    ),
                    severity="error",
                    rule_id="namespace.compatibility_alias",
                )
                for violation in violations
            ],
            detector_name=self.__class__.__name__,
        )

    @classmethod
    def detect_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.CompatibilityAliasViolation]:
        """Scan a file and return typed namespace violations."""
        return cls.scan_file_impl(
            file_path=file_path,
            _parse_failures=parse_failures,
        )

    @classmethod
    def scan_file_impl(
        cls,
        *,
        file_path: Path,
        _parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> list[nem.CompatibilityAliasViolation]:
        """Scan a file for compatibility aliases that may be removable."""
        if file_path.suffix != ".py":
            return []
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return []
        violations: list[nem.CompatibilityAliasViolation] = []
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
                    nem.CompatibilityAliasViolation.create(
                        file=str(file_path),
                        line=stmt.lineno,
                        alias_name=alias_name,
                        target_name=target_name,
                    ),
                )
        return violations


class FlextInfraRefactorDependencyAnalyzerFacade:
    """Facade grouping all dependency analysis detectors and scanners."""

    @staticmethod
    def load_python_module(
        file_path: Path,
        *,
        stage: str = "scan",
        parse_failures: list[nem.ParseFailureViolation] | None = None,
    ) -> m.Infra.Refactor.ParsedPythonModule | None:
        """Load and parse a Python source file, recording failures if provided."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except UnicodeDecodeError as exc:
            if parse_failures is not None:
                parse_failures.append(
                    nem.ParseFailureViolation.create(
                        file=str(file_path),
                        stage=stage,
                        error_type=type(exc).__name__,
                        detail=str(exc),
                    ),
                )
            return None
        except OSError as exc:
            if parse_failures is not None:
                parse_failures.append(
                    nem.ParseFailureViolation.create(
                        file=str(file_path),
                        stage=stage,
                        error_type=type(exc).__name__,
                        detail=str(exc),
                    ),
                )
            return None
        tree = u.Infra.parse_ast_from_source(source)
        if tree is None:
            if parse_failures is not None:
                parse_failures.append(
                    nem.ParseFailureViolation.create(
                        file=str(file_path),
                        stage=stage,
                        error_type="SyntaxError",
                        detail="invalid python source",
                    ),
                )
            return None
        return m.Infra.Refactor.ParsedPythonModule(source=source, tree=tree)

    NamespaceFacadeScanner = NamespaceFacadeScanner
    LooseObjectDetector = LooseObjectDetector
    ImportAliasDetector = ImportAliasDetector
    InternalImportDetector = InternalImportDetector
    ManualProtocolDetector = ManualProtocolDetector
    CyclicImportDetector = CyclicImportDetector
    RuntimeAliasDetector = RuntimeAliasDetector
    FutureAnnotationsDetector = FutureAnnotationsDetector
    ManualTypingAliasDetector = ManualTypingAliasDetector
    CompatibilityAliasDetector = CompatibilityAliasDetector


__all__ = [
    "CompatibilityAliasDetector",
    "CyclicImportDetector",
    "DependencyAnalyzer",
    "FlextInfraRefactorDependencyAnalyzerFacade",
    "FutureAnnotationsDetector",
    "ImportAliasDetector",
    "InternalImportDetector",
    "LooseObjectDetector",
    "ManualProtocolDetector",
    "ManualTypingAliasDetector",
    "NamespaceFacadeScanner",
    "RuntimeAliasDetector",
]
