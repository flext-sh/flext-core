"""Cross-project dependency analysis using LibCST and ast-grep."""

from __future__ import annotations

import sys
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import override

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
                )
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
        self, projects: list[m.Infra.Refactor.ProjectInfo]
    ) -> dict[str, str]:
        idx: dict[str, str] = {}
        for proj in projects:
            for pkg in proj.package_roots:
                _ = idx.setdefault(pkg, proj.name)
        return idx

    def _find_import_candidate_files(
        self, project: m.Infra.Refactor.ProjectInfo
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
        self, src_path: Path, pattern: str
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
                capture.error or "capture failed"
            )
        if not capture.value:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].ok([])
        try:
            return r[list[m.Infra.Refactor.AstGrepMatchEnvelope]].ok(
                TypeAdapter(list[m.Infra.Refactor.AstGrepMatchEnvelope]).validate_json(
                    capture.value
                )
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
                )
            )
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError) as exc:
            return r[m.Infra.Refactor.FileImportData].fail(f"{file_path}: {exc}")


__all__ = ["DependencyAnalyzer"]
