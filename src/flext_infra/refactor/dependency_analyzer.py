"""Cross-project dependency analysis using LibCST and ast-grep."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_infra import FlextInfraCommandRunner, c


@dataclass(frozen=True)
class _ProjectInfo:
    name: str
    path: Path
    src_path: Path
    package_roots: set[str]


@dataclass(frozen=True)
class _FileImportData:
    imported_modules: set[str]
    imported_symbols: set[str]


class _ImportCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.imported_modules: set[str] = set()
        self.imported_symbols: set[str] = set()

    @override
    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            module_root = self._module_root(alias.name)
            if module_root:
                self.imported_modules.add(module_root)

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        if node.relative:
            return

        module_root = self._module_root(node.module)
        if module_root:
            self.imported_modules.add(module_root)

        if isinstance(node.names, cst.ImportStar):
            return

        for alias in node.names:
            symbol = self._imported_symbol(alias.name)
            if symbol:
                self.imported_symbols.add(symbol)

    def _module_root(self, node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parts: list[str] = []
            current: cst.BaseExpression | None = node
            while isinstance(current, cst.Attribute):
                parts.append(current.attr.value)
                current = current.value
            if isinstance(current, cst.Name):
                parts.append(current.value)
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
        self._workspace_root = workspace_root.resolve()
        self._stdlib_roots = set(sys.stdlib_module_names)
        self._projects = self._discover_projects()
        self._package_to_project = self._build_package_index(self._projects)
        self._graph_cache: dict[str, list[str]] | None = None

    def build_import_graph(self) -> dict[str, list[str]]:
        """Build and cache the inter-project import graph."""
        if self._graph_cache is not None:
            return self._graph_cache

        graph: dict[str, set[str]] = {project.name: set() for project in self._projects}

        for project in self._projects:
            files = self._find_import_candidate_files(project)
            for file_path in files:
                file_data = self._parse_imports(file_path)
                for module_root in file_data.imported_modules:
                    if module_root in self._stdlib_roots:
                        continue
                    imported_project = self._package_to_project.get(module_root)
                    if imported_project is None or imported_project == project.name:
                        continue
                    graph[project.name].add(imported_project)

        ordered_graph = {
            project_name: sorted(imports)
            for project_name, imports in sorted(graph.items())
        }
        self._graph_cache = ordered_graph
        return ordered_graph

    def find_consumers(self, class_name: str) -> list[Path]:
        """Find all files importing the given class name."""
        consumers: set[Path] = set()

        for project in self._projects:
            files = self._find_import_candidate_files(project)
            for file_path in files:
                file_data = self._parse_imports(file_path)
                if class_name in file_data.imported_symbols:
                    consumers.add(file_path)

        return sorted(consumers)

    def determine_transformation_order(self) -> list[str]:
        """Return topologically sorted project order for safe transformations."""
        graph = self.build_import_graph()
        if not graph:
            return []

        sorter = TopologicalSorter(graph)
        try:
            return list(sorter.static_order())
        except CycleError:
            return sorted(graph)

    def _discover_projects(self) -> list[_ProjectInfo]:
        projects: list[_ProjectInfo] = []

        for candidate in sorted(self._workspace_root.iterdir()):
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue

            src_path = candidate / c.Infra.Paths.DEFAULT_SRC_DIR
            if not src_path.is_dir():
                continue

            package_roots = self._discover_package_roots(src_path)
            projects.append(
                _ProjectInfo(
                    name=candidate.name,
                    path=candidate,
                    src_path=src_path,
                    package_roots=package_roots,
                )
            )

        return projects

    def _discover_package_roots(self, src_path: Path) -> set[str]:
        package_roots: set[str] = set()

        for path in src_path.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_dir() and (path / c.Infra.Files.INIT_PY).is_file():
                package_roots.add(path.name)
            elif (
                path.is_file()
                and path.suffix == c.Infra.Extensions.PYTHON
                and path.stem != "__init__"
            ):
                package_roots.add(path.stem)

        return package_roots

    def _build_package_index(
        self,
        projects: list[_ProjectInfo],
    ) -> dict[str, str]:
        package_to_project: dict[str, str] = {}

        for project in projects:
            for package_root in project.package_roots:
                package_to_project.setdefault(package_root, project.name)

        return package_to_project

    def _find_import_candidate_files(self, project: _ProjectInfo) -> list[Path]:
        ast_grep_files = self._scan_import_files_with_ast_grep(project.src_path)
        if ast_grep_files:
            return sorted(ast_grep_files)

        return sorted(self._iter_python_files(project.src_path))

    def _iter_python_files(self, src_path: Path) -> list[Path]:
        files: list[Path] = []
        for file_path in src_path.rglob(c.Infra.Extensions.PYTHON_GLOB):
            if "__pycache__" in file_path.parts:
                continue
            files.append(file_path)
        return files

    def _scan_import_files_with_ast_grep(self, src_path: Path) -> set[Path]:
        files: set[Path] = set()

        for pattern in ("import $MODULE", "from $MODULE import $$$"):
            payload = self._run_ast_grep(src_path, pattern)
            for entry in payload:
                file_raw = entry.get("file")
                if not isinstance(file_raw, str):
                    continue
                file_path = Path(file_raw)
                if not file_path.is_absolute():
                    file_path = (src_path / file_path).resolve()
                if file_path.suffix == c.Infra.Extensions.PYTHON:
                    files.add(file_path)

        return files

    def _run_ast_grep(self, src_path: Path, pattern: str) -> list[dict[str, object]]:
        cmd = [
            "sg",
            "--pattern",
            pattern,
            "--lang",
            c.Infra.Toml.PYTHON,
            "--json",
            str(src_path),
        ]
        runner = FlextInfraCommandRunner()
        result = runner.capture(cmd)
        if result.is_failure:
            return []

        raw_output = result.value.strip()
        if not raw_output:
            return []

        try:
            return TypeAdapter(list[dict[str, object]]).validate_json(raw_output)
        except ValidationError:
            return []

    def _parse_imports(self, file_path: Path) -> _FileImportData:
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            module = cst.parse_module(source)
            collector = _ImportCollector()
            module.visit(collector)
            return _FileImportData(
                imported_modules=collector.imported_modules,
                imported_symbols=collector.imported_symbols,
            )
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
            return _FileImportData(imported_modules=set(), imported_symbols=set())


__all__ = ["DependencyAnalyzer"]
