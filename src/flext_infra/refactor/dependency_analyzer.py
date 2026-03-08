"""Cross-project dependency analysis using LibCST and ast-grep."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import ClassVar, override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, m, u


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
                for mod_root in parsed.value.imported_modules:
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
                if class_name in parsed.value.imported_symbols:
                    consumers.add(fp)
        return r[list[Path]].ok(sorted(consumers))

    def determine_transformation_order(self) -> r[list[str]]:
        """Return topologically sorted project order."""
        graph_result = self.build_import_graph()
        if graph_result.is_failure:
            return r[list[str]].fail(graph_result.error or "graph build failed")
        graph = graph_result.value
        if not graph:
            return r[list[str]].ok([])
        try:
            return r[list[str]].ok(list(TopologicalSorter(graph).static_order()))
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
        for p in src_path.iterdir():
            if p.name.startswith("."):
                continue
            if p.is_dir() and (p / c.Infra.Files.INIT_PY).is_file():
                roots.add(p.name)
            elif (
                p.is_file()
                and p.suffix == c.Infra.Extensions.PYTHON
                and (p.stem != "__init__")
            ):
                roots.add(p.stem)
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
            return sorted(grep_files.value)
        return sorted(self._iter_python_files(project.src_path))

    def _iter_python_files(self, src_path: Path) -> list[Path]:
        return [
            fp
            for fp in src_path.rglob(c.Infra.Extensions.PYTHON_GLOB)
            if "__pycache__" not in fp.parts
        ]

    def _scan_import_files_with_ast_grep(self, src_path: Path) -> r[set[Path]]:
        files: set[Path] = set()
        for pattern in ("import $MODULE", "from $MODULE import $$$"):
            result = self._run_ast_grep(src_path, pattern)
            if result.is_failure:
                return r[set[Path]].fail(result.error or "ast-grep failed")
            for entry in result.value:
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
        capture = u.Infra.Refactor.capture_output(cmd)
        if capture.is_failure:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].fail(
                capture.error or "capture failed",
            )
        if not capture.value:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].ok([])
        try:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].ok(
                TypeAdapter(list[m.Infra.Refactor.AstGrepMatchEnvelope]).validate_json(
                    capture.value,
                ),
            )
        except ValidationError as exc:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].fail(str(exc))

    def _parse_imports(self, file_path: Path) -> r[m.Infra.Refactor.FileImportData]:
        try:
            src = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = cst.parse_module(src)
            col = ImportCollector()
            _ = tree.visit(col)
            return r[m.Infra.Refactor.FileImportData].ok(
                m.Infra.Refactor.FileImportData(
                    imported_modules=col.imported_modules,
                    imported_symbols=col.imported_symbols,
                ),
            )
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError) as exc:
            return r[m.Infra.Refactor.FileImportData].fail(f"{file_path}: {exc}")


def load_python_module(
    file_path: Path,
    *,
    stage: str = "scan",
    parse_failures: list[
        m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
    ]
    | None = None,
) -> ParsedPythonModule | None:
    """Load and parse a Python source file, recording failures if provided."""
    try:
        source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
    except UnicodeDecodeError as exc:
        if parse_failures is not None:
            parse_failures.append(
                m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation.create(
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
                m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation.create(
                    file=str(file_path),
                    stage=stage,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                ),
            )
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        if parse_failures is not None:
            parse_failures.append(
                m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation.create(
                    file=str(file_path),
                    stage=stage,
                    error_type=type(exc).__name__,
                    detail=str(exc),
                ),
            )
        return None
    return ParsedPythonModule(source=source, tree=tree)


@dataclass(frozen=True, slots=True)
class ParsedPythonModule:
    source: str
    tree: ast.Module


class NamespaceFacadeScanner:
    """Scan projects for namespace facade class patterns."""

    @classmethod
    def scan_project(
        cls,
        *,
        project_root: Path,
        project_name: str,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.FacadeStatus]:
        """Scan a project for namespace facade classes and return their status."""
        results: list[m.Infra.Refactor.NamespaceEnforcementModels.FacadeStatus] = []
        class_stem = cls.project_class_stem(project_name=project_name)
        for family, suffix in c.Infra.Refactor.NAMESPACE_FACADE_FAMILIES.items():
            expected_class = f"{class_stem}{suffix}"
            found_class, found_file, symbol_count = cls._find_facade_class(
                project_root=project_root,
                family=family,
                expected_class=expected_class,
                suffix=suffix,
                parse_failures=parse_failures,
            )
            results.append(
                m.Infra.Refactor.NamespaceEnforcementModels.FacadeStatus.create(
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
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None,
    ) -> tuple[str, str, int]:
        file_pattern = c.Infra.Refactor.NAMESPACE_FACADE_FILE_PATTERNS[family]
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return ("", "", 0)
        for file_path in src_dir.rglob(file_pattern):
            parsed = load_python_module(
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


class LooseObjectDetector:
    """Detect loose top-level objects that should be inside namespace classes."""

    ALLOWED_TOP_LEVEL = frozenset({"__all__", "__version__", "__version_info__"})

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_name: str,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.LooseObjectViolation]:
        """Scan a file for loose top-level objects outside namespace classes."""
        if file_path.name in c.Infra.Refactor.NAMESPACE_PROTECTED_FILES:
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_SETTINGS_FILE_NAMES:
            return []
        parsed = load_python_module(
            file_path,
            stage="loose-object-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        namespace_classes = cls._find_namespace_classes(tree=tree)
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.LooseObjectViolation
        ] = []
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
    ) -> m.Infra.Refactor.NamespaceEnforcementModels.LooseObjectViolation | None:
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
            return (
                m.Infra.Refactor.NamespaceEnforcementModels.LooseObjectViolation.create(
                    file=str(file_path),
                    line=stmt.lineno,
                    name=stmt.name,
                    kind="function",
                    suggestion=f"{class_stem}Utilities",
                )
            )
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if name in cls.ALLOWED_TOP_LEVEL:
                return None
            if name.startswith("_"):
                return None
            if c.Infra.Refactor.NAMESPACE_CONSTANT_PATTERN.match(name):
                return m.Infra.Refactor.NamespaceEnforcementModels.LooseObjectViolation.create(
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
                    return m.Infra.Refactor.NamespaceEnforcementModels.LooseObjectViolation.create(
                        file=str(file_path),
                        line=stmt.lineno,
                        name=name,
                        kind="constant",
                        suggestion=f"{class_stem}Constants",
                    )
        if isinstance(stmt, ast.TypeAlias):
            name = stmt.name.id if hasattr(stmt.name, "id") else ""
            if name and name not in cls.ALLOWED_TOP_LEVEL:
                return m.Infra.Refactor.NamespaceEnforcementModels.LooseObjectViolation.create(
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


class ImportAliasDetector:
    """Detect deep import paths that should use top-level aliases."""

    ALIAS_MODULES: ClassVar[dict[str, str]] = {
        "flext_core": "from flext_core import c, m, r, t, u, p",
        "flext_infra": "from flext_infra import c, m, t, u, p",
    }

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.ImportAliasViolation]:
        """Scan a file for deep import paths that should use aliases."""
        parsed = load_python_module(
            file_path,
            stage="import-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ImportAliasViolation
        ] = []
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
                        m.Infra.Refactor.NamespaceEnforcementModels.ImportAliasViolation.create(
                            file=str(file_path),
                            line=stmt.lineno,
                            current_import=current,
                            suggested_import=suggestion,
                        ),
                    )
        return violations


class InternalImportDetector:
    """Detect imports of private modules or symbols across boundaries."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.InternalImportViolation]:
        """Scan a file for private module or symbol imports."""
        parsed = load_python_module(
            file_path,
            stage="internal-import-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.InternalImportViolation
        ] = []
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
                m.Infra.Refactor.NamespaceEnforcementModels.InternalImportViolation.create(
                    file=str(file_path),
                    line=stmt.lineno,
                    current_import=current_import,
                    detail=detail,
                ),
            )
        return violations


class ManualProtocolDetector:
    """Detect Protocol classes defined outside canonical protocol files."""

    CANONICAL_FILE_NAMES = c.Infra.Refactor.NAMESPACE_CANONICAL_PROTOCOL_FILES
    CANONICAL_DIR_NAME = c.Infra.Refactor.NAMESPACE_CANONICAL_PROTOCOL_DIR

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.ManualProtocolViolation]:
        """Scan a file for Protocol classes outside canonical locations."""
        in_canonical_file = file_path.name in cls.CANONICAL_FILE_NAMES
        in_canonical_dir = cls.CANONICAL_DIR_NAME in file_path.parts
        if in_canonical_file or in_canonical_dir:
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_PROTECTED_FILES:
            return []
        parsed = load_python_module(
            file_path,
            stage="manual-protocol-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ManualProtocolViolation
        ] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            if cls.is_protocol_class(stmt):
                violations.append(
                    m.Infra.Refactor.NamespaceEnforcementModels.ManualProtocolViolation.create(
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

    @classmethod
    def scan_project(
        cls,
        *,
        project_root: Path,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.CyclicImportViolation]:
        """Scan a project for cyclic import dependencies."""
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return []
        graph: dict[str, set[str]] = {}
        file_map: dict[str, str] = {}
        package_roots = cls._discover_package_roots(src_dir=src_dir)
        for py_file in sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)):
            if "__pycache__" in py_file.parts:
                continue
            module_name = cls._file_to_module(file_path=py_file, src_dir=src_dir)
            if not module_name:
                continue
            file_map[module_name] = str(py_file)
            graph.setdefault(module_name, set())
            parsed = load_python_module(
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
                    root_pkg = imported.split(".")[0]
                    if root_pkg in package_roots:
                        graph[module_name].add(imported)
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.CyclicImportViolation
        ] = []
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
                    m.Infra.Refactor.NamespaceEnforcementModels.CyclicImportViolation.create(
                        cycle=normalized_cycle,
                        files=cycle_files,
                    ),
                )
        return violations

    @staticmethod
    def _discover_package_roots(*, src_dir: Path) -> set[str]:
        roots: set[str] = set()
        for entry in src_dir.iterdir():
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            if entry.is_dir() and (entry / "__init__.py").is_file():
                roots.add(entry.name)
        return roots

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


class RuntimeAliasDetector:
    """Detect missing or duplicate runtime alias assignments."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_name: str,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.RuntimeAliasViolation]:
        """Scan a file for missing or duplicate runtime alias assignments."""
        if file_path.name not in c.Infra.Refactor.NAMESPACE_FILE_TO_FAMILY:
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_PROTECTED_FILES:
            return []
        if "src" not in file_path.parts:
            return []
        parsed = load_python_module(
            file_path,
            stage="runtime-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.RuntimeAliasViolation
        ] = []
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
                m.Infra.Refactor.NamespaceEnforcementModels.RuntimeAliasViolation.create(
                    file=str(file_path),
                    kind="missing",
                    alias=expected_alias,
                    detail=f"No '{expected_alias} = ...' assignment found",
                ),
            )
        elif len(matches) > 1:
            violations.append(
                m.Infra.Refactor.NamespaceEnforcementModels.RuntimeAliasViolation.create(
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


class FutureAnnotationsDetector:
    """Detect Python files missing the future annotations import."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.FutureAnnotationsViolation]:
        """Scan a file for missing future annotations import."""
        if file_path.name == "py.typed":
            return []
        parsed = load_python_module(
            file_path,
            stage="future-annotations-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        if len(parsed.tree.body) == 0:
            return []
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
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                break
        return [
            m.Infra.Refactor.NamespaceEnforcementModels.FutureAnnotationsViolation.create(
                file=str(file_path),
            ),
        ]


class ManualTypingAliasDetector:
    """Detect type aliases defined outside canonical typings files."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.ManualTypingAliasViolation]:
        """Scan a file for type aliases outside canonical locations."""
        if file_path.suffix != ".py":
            return []
        if file_path.name in c.Infra.Refactor.NAMESPACE_CANONICAL_TYPINGS_FILES:
            return []
        if c.Infra.Refactor.NAMESPACE_CANONICAL_TYPINGS_DIR in file_path.parts:
            return []
        parsed = load_python_module(
            file_path,
            stage="manual-typing-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        source, tree = parsed.source, parsed.tree
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ManualTypingAliasViolation
        ] = []
        for stmt in tree.body:
            if isinstance(stmt, ast.TypeAlias):
                alias_name = stmt.name.id
                violations.append(
                    m.Infra.Refactor.NamespaceEnforcementModels.ManualTypingAliasViolation.create(
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
                        m.Infra.Refactor.NamespaceEnforcementModels.ManualTypingAliasViolation.create(
                            file=str(file_path),
                            line=stmt.lineno,
                            name=stmt.target.id,
                            detail="TypeAlias assignment must be centralized under typings scope",
                        ),
                    )
        return violations


class CompatibilityAliasDetector:
    """Detect compatibility alias assignments that may be removable."""

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        parse_failures: list[
            m.Infra.Refactor.NamespaceEnforcementModels.ParseFailureViolation
        ]
        | None = None,
    ) -> list[m.Infra.Refactor.NamespaceEnforcementModels.CompatibilityAliasViolation]:
        """Scan a file for compatibility aliases that may be removable."""
        if file_path.suffix != ".py":
            return []
        parsed = load_python_module(
            file_path,
            stage="compatibility-alias-scan",
            parse_failures=parse_failures,
        )
        if parsed is None:
            return []
        tree = parsed.tree
        violations: list[
            m.Infra.Refactor.NamespaceEnforcementModels.CompatibilityAliasViolation
        ] = []
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
                    m.Infra.Refactor.NamespaceEnforcementModels.CompatibilityAliasViolation.create(
                        file=str(file_path),
                        line=stmt.lineno,
                        alias_name=alias_name,
                        target_name=target_name,
                    ),
                )
        return violations


class FlextInfraRefactorDependencyAnalyzerFacade:
    """Facade grouping all dependency analysis detectors and scanners."""

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
    "load_python_module",
]
