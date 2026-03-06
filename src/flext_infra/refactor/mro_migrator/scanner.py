"""AST scanner for migrate-to-mro candidates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from flext_infra import FlextInfraDiscoveryService, c


@dataclass(frozen=True)
class MROCandidate:
    file: str
    module: str
    symbol: str
    line: int
    kind: str
    class_name: str


@dataclass(frozen=True)
class MROFileScan:
    file: str
    module: str
    constants_class: str
    types_class: str
    candidates: tuple[MROCandidate, ...]


class FlextInfraRefactorMROScanner:
    @classmethod
    def scan_workspace(
        cls,
        *,
        workspace_root: Path,
        target: str,
    ) -> tuple[list[MROFileScan], int]:
        results: list[MROFileScan] = []
        scanned = 0
        for project_root in cls._project_roots(workspace_root=workspace_root):
            for file_path in cls._iter_python_files(project_root=project_root):
                scanned += 1
                result = cls.scan_file(
                    file_path=file_path,
                    project_root=project_root,
                    target=target,
                )
                if result is None or len(result.candidates) == 0:
                    continue
                results.append(result)
        return results, scanned

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_root: Path,
        target: str,
    ) -> MROFileScan | None:
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return None

        constants_class = cls._first_matching_class(tree=tree, suffix="Constants")
        types_class = cls._first_matching_class(tree=tree, suffix="Types")
        module = cls._module_path(file_path=file_path, project_root=project_root)
        candidates: list[MROCandidate] = []

        for stmt in tree.body:
            if target in {"constants", "all"}:
                constant = cls._constant_candidate(
                    stmt=stmt,
                    file_path=file_path,
                    module=module,
                    class_name=constants_class or "FlextConstants",
                )
                if constant is not None:
                    candidates.append(constant)
            if target in {"typings", "all"}:
                typed = cls._typing_candidate(
                    stmt=stmt,
                    file_path=file_path,
                    module=module,
                    class_name=types_class or "FlextTypes",
                )
                if typed is not None:
                    candidates.append(typed)

        return MROFileScan(
            file=str(file_path),
            module=module,
            constants_class=constants_class,
            types_class=types_class,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _project_roots(*, workspace_root: Path) -> list[Path]:
        discovery = FlextInfraDiscoveryService()
        projects = discovery.discover_projects(workspace_root)
        roots: list[Path] = []
        if projects.is_success:
            roots = [project.path for project in projects.unwrap()]
        if (
            len(roots) == 0
            and (workspace_root / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir()
        ):
            roots = [workspace_root]
        return roots

    @staticmethod
    def _iter_python_files(*, project_root: Path) -> list[Path]:
        files: list[Path] = []
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        tests_dir = project_root / c.Infra.Directories.TESTS
        if src_dir.is_dir():
            files.extend(sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)))
        if tests_dir.is_dir():
            files.extend(sorted(tests_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)))
        return files

    @staticmethod
    def _module_path(*, file_path: Path, project_root: Path) -> str:
        rel = file_path.relative_to(project_root)
        parts = [part for part in rel.with_suffix("").parts if part != "src"]
        return ".".join(parts)

    @staticmethod
    def _first_matching_class(*, tree: ast.Module, suffix: str) -> str:
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name.endswith(suffix):
                return stmt.name
        return ""

    @staticmethod
    def _constant_candidate(
        *,
        stmt: ast.stmt,
        file_path: Path,
        module: str,
        class_name: str,
    ) -> MROCandidate | None:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            return None
        annotation = stmt.annotation
        is_final = False
        if isinstance(annotation, ast.Name):
            is_final = annotation.id == "Final"
        elif isinstance(annotation, ast.Attribute):
            is_final = annotation.attr == "Final"
        elif isinstance(annotation, ast.Subscript) and isinstance(
            annotation.value, ast.Name
        ):
            is_final = annotation.value.id == "Final"
        if not is_final:
            return None
        return MROCandidate(
            file=str(file_path),
            module=module,
            symbol=stmt.target.id,
            line=stmt.lineno,
            kind="constant",
            class_name=class_name,
        )

    @staticmethod
    def _typing_candidate(
        *,
        stmt: ast.stmt,
        file_path: Path,
        module: str,
        class_name: str,
    ) -> MROCandidate | None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 0 or not isinstance(stmt.targets[0], ast.Name):
                return None
            if not isinstance(stmt.value, ast.Call):
                return None
            func = stmt.value.func
            is_typevar = isinstance(func, ast.Name) and func.id == "TypeVar"
            if isinstance(func, ast.Attribute):
                is_typevar = func.attr == "TypeVar"
            if not is_typevar:
                return None
            return MROCandidate(
                file=str(file_path),
                module=module,
                symbol=stmt.targets[0].id,
                line=stmt.lineno,
                kind="typevar",
                class_name=class_name,
            )

        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name):
                return None
            annotation = stmt.annotation
            is_alias = isinstance(annotation, ast.Name) and annotation.id == "TypeAlias"
            if isinstance(annotation, ast.Attribute):
                is_alias = annotation.attr == "TypeAlias"
            if not is_alias:
                return None
            return MROCandidate(
                file=str(file_path),
                module=module,
                symbol=stmt.target.id,
                line=stmt.lineno,
                kind="typealias",
                class_name=class_name,
            )

        return None


__all__ = ["FlextInfraRefactorMROScanner", "MROCandidate", "MROFileScan"]
