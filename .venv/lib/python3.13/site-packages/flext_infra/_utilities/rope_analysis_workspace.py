"""Rope workspace indexing helpers."""

from __future__ import annotations

import operator
from pathlib import Path

from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t


class FlextInfraUtilitiesRopeAnalysisWorkspace:
    """Rope-backed workspace indexing helpers."""

    @staticmethod
    def _project_root_for_file(repository_root: Path, file_path: Path) -> Path | None:
        """Project root for file."""
        for parent in file_path.parents:
            if (parent / "pyproject.toml").is_file():
                return parent.resolve()
            if parent == repository_root:
                return repository_root
        return None

    @classmethod
    def _package_name_for_dir(cls, package_dir: Path, *, project_root: Path) -> str:
        """Package name for dir."""
        try:
            relative_parts = package_dir.relative_to(project_root).parts
        except ValueError:
            return ""
        if not relative_parts:
            return ""
        root_name = relative_parts[0]
        if root_name == c.Infra.DEFAULT_SRC_DIR:
            package_parts = relative_parts[1:]
        elif root_name in c.Infra.ROOT_WRAPPER_SEGMENTS:
            package_parts = relative_parts
        else:
            package_parts = ()
        return ".".join(package_parts)

    @classmethod
    def _module_name_for_file(cls, file_path: Path, *, project_root: Path) -> str:
        """Return the module name for a file."""
        if file_path.name in {c.Infra.INIT_PY, c.Infra.INIT_PYI}:
            return cls._package_name_for_dir(
                file_path.parent, project_root=project_root
            )
        package_name = cls._package_name_for_dir(
            file_path.parent, project_root=project_root
        )
        return f"{package_name}.{file_path.stem}" if package_name else ""

    @staticmethod
    def _is_generated_init_stub(file_path: Path) -> bool:
        """Return whether ``file_path`` is a codegen-owned package stub."""
        if file_path.name != c.Infra.INIT_PYI:
            return False
        return file_path.read_text(encoding=c.Cli.ENCODING_DEFAULT).startswith(
            c.Infra.AUTOGEN_HEADERS
        )

    @staticmethod
    def _python_and_stub_file_paths(
        rope_project: t.Infra.RopeProject, resolved_root: Path
    ) -> tuple[Path, ...]:
        """Return indexed sources, declared wrapper modules, and typing stubs."""
        python_paths = {
            path.resolve()
            for path in FlextInfraUtilitiesRopeCore.python_file_paths(rope_project)
            if not set(path.relative_to(resolved_root).parts)
            & c.Infra.ITERATION_EXCLUDED_PARTS
        }
        # flext-pulj (codex): Rope's source roots omit tests/examples/scripts;
        # index those declared wrapper surfaces so explicitly targeted codegen
        # can update their generated initializers without textual fallbacks.
        wrapper_paths = {
            path.resolve()
            for wrapper_name in c.Infra.ROOT_WRAPPER_SEGMENTS
            for wrapper_root in (resolved_root / wrapper_name,)
            if wrapper_root.is_dir()
            for path in wrapper_root.rglob("*.py")
            if path.is_file()
            and not set(path.relative_to(resolved_root).parts)
            & c.Infra.ITERATION_EXCLUDED_PARTS
        }
        stub_paths = {
            path.resolve()
            for path in resolved_root.rglob("*.pyi")
            if path.is_file()
            and not set(path.relative_to(resolved_root).parts)
            & c.Infra.ITERATION_EXCLUDED_PARTS
        }
        return tuple(
            sorted(python_paths | wrapper_paths | stub_paths, key=Path.as_posix)
        )

    @classmethod
    def _collect_modules(
        cls, rope_project: t.Infra.RopeProject, resolved_root: Path
    ) -> tuple[
        dict[str, m.Infra.RopeModuleIndexEntry],
        dict[Path, list[m.Infra.RopeModuleIndexEntry]],
        dict[str, Path],
        dict[str, str],
        set[Path],
    ]:
        """Collect modules."""
        modules_by_path: dict[str, m.Infra.RopeModuleIndexEntry] = {}
        modules_by_dir: dict[Path, list[m.Infra.RopeModuleIndexEntry]] = {}
        package_dir_by_name: dict[str, Path] = {}
        project_package_by_root: dict[str, str] = {}
        package_dirs: set[Path] = set()
        for file_path in cls._python_and_stub_file_paths(rope_project, resolved_root):
            resolved_file_path = file_path.resolve()
            if cls._is_generated_init_stub(resolved_file_path):
                continue
            try:
                resource_path = resolved_file_path.relative_to(resolved_root).as_posix()
            except ValueError:
                continue
            package_dir = resolved_file_path.parent
            is_package_init = resolved_file_path.name in {
                c.Infra.INIT_PY,
                c.Infra.INIT_PYI,
            }
            project_root = cls._project_root_for_file(resolved_root, resolved_file_path)
            module_name = (
                cls._module_name_for_file(resolved_file_path, project_root=project_root)
                if project_root is not None
                else ""
            )
            package_name = (
                cls._package_name_for_dir(package_dir, project_root=project_root)
                if project_root is not None
                else module_name
                if is_package_init
                else module_name.rsplit(".", maxsplit=1)[0]
                if "." in module_name
                else ""
            )
            entry = m.Infra.RopeModuleIndexEntry(
                file_path=resolved_file_path,
                resource_path=resource_path,
                module_name=module_name,
                package_name=package_name,
                package_dir=package_dir,
                project_root=project_root,
                is_package_init=is_package_init,
            )
            modules_by_path[str(resolved_file_path)] = entry
            modules_by_dir.setdefault(package_dir, []).append(entry)
            package_dirs.add(package_dir)
            if package_name:
                package_dir_by_name[package_name] = package_dir
                if (
                    project_root is not None
                    and "." not in package_name
                    and package_dir.parent.name == c.Infra.DEFAULT_SRC_DIR
                ):
                    project_package_by_root[str(project_root)] = package_name
        return (
            modules_by_path,
            modules_by_dir,
            package_dir_by_name,
            project_package_by_root,
            package_dirs,
        )

    @classmethod
    def index_rope_workspace(
        cls, rope_project: t.Infra.RopeProject, repository_root: Path
    ) -> m.Infra.RopeWorkspaceIndex:
        """Build a generic Rope workspace index for package-oriented planning."""
        resolved_root = repository_root.resolve()
        (
            modules_by_path,
            modules_by_dir,
            package_dir_by_name,
            project_package_by_root,
            package_dirs,
        ) = cls._collect_modules(rope_project, resolved_root)
        sorted_package_dirs = tuple(sorted(package_dirs))
        package_dir_set = frozenset(sorted_package_dirs)
        direct_children_by_dir: dict[Path, list[Path]] = {
            package_dir: [] for package_dir in sorted_package_dirs
        }
        descendants_by_dir: dict[Path, list[Path]] = {
            package_dir: [] for package_dir in sorted_package_dirs
        }
        for package_dir in sorted_package_dirs:
            parent_dir = package_dir.parent
            if parent_dir in package_dir_set:
                direct_children_by_dir[parent_dir].append(package_dir)
            for ancestor_dir in package_dir.parents:
                if ancestor_dir == package_dir:
                    continue
                if ancestor_dir in package_dir_set:
                    descendants_by_dir[ancestor_dir].append(package_dir)
        packages_by_dir: dict[str, m.Infra.RopePackageIndexEntry] = {}
        for package_dir in sorted_package_dirs:
            dir_modules = tuple(
                sorted(
                    modules_by_dir.get(package_dir, ()),
                    key=operator.attrgetter("file_path.name"),
                )
            )
            init_path = (package_dir / c.Infra.INIT_PY).resolve()
            init_entry = modules_by_path.get(str(init_path))
            project_root = (
                init_entry.project_root
                if init_entry is not None
                else next(
                    (
                        entry.project_root
                        for entry in dir_modules
                        if entry.project_root is not None
                    ),
                    None,
                )
            )
            package_name = (
                cls._package_name_for_dir(package_dir, project_root=project_root)
                if project_root is not None
                else init_entry.package_name
                if init_entry is not None
                else ""
            )
            if package_name and package_name not in package_dir_by_name:
                package_dir_by_name[package_name] = package_dir
            if (
                project_root is not None
                and "." not in package_name
                and package_dir.parent.name == c.Infra.DEFAULT_SRC_DIR
                and str(project_root) not in project_package_by_root
            ):
                project_package_by_root[str(project_root)] = package_name
            direct_child_dirs = tuple(direct_children_by_dir.get(package_dir, ()))
            descendant_child_dirs = tuple(descendants_by_dir.get(package_dir, ()))
            packages_by_dir[str(package_dir)] = m.Infra.RopePackageIndexEntry(
                package_dir=package_dir,
                init_path=init_path,
                package_name=package_name,
                project_root=project_root,
                modules=dir_modules,
                direct_child_dirs=direct_child_dirs,
                descendant_child_dirs=descendant_child_dirs,
            )
        return m.Infra.RopeWorkspaceIndex(
            repository_root=resolved_root,
            package_dirs=sorted_package_dirs,
            packages_by_dir=packages_by_dir,
            modules_by_path=modules_by_path,
            package_dir_by_name=package_dir_by_name,
            project_package_by_root=project_package_by_root,
        )


__all__: list[str] = ["FlextInfraUtilitiesRopeAnalysisWorkspace"]
