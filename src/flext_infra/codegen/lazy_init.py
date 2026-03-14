"""Lazy-init ``__init__.py`` generator (PEP 562).

Auto-discovers exports from sibling ``.py`` files and generates clean
lazy-loading ``__init__.py`` files using ``flext_core.lazy``.

The generator **never** reads existing ``__init__.py`` content for exports
or import mappings.  It discovers everything by scanning sibling ``.py``
files' ``__all__`` and AST definitions.  You can DELETE every
``__init__.py`` and regenerate them perfectly from scratch.

Processes directories bottom-up (deepest first) so child packages are
generated before parents, and parent packages can include child exports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import contextlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import u
from flext_infra._utilities.output import output
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra.constants import FlextInfraConstants as c

# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class FlextInfraCodegenLazyInit(s[int]):
    """Generates ``__init__.py`` with PEP 562 lazy imports.

    Scans sibling ``.py`` files in each package directory, discovers their
    exports via ``__all__`` or AST scanning, and generates clean lazy-loading
    ``__init__.py`` files.  Processes bottom-up so child packages are
    generated before parents.
    """

    def __init__(self, workspace_root: Path) -> None:
        """Initialize lazy init generator with workspace root."""
        super().__init__(
            config_type=None,
            config_overrides=None,
            initial_context=None,
            subproject=None,
            services=None,
            factories=None,
            resources=None,
            container_overrides=None,
            wire_modules=None,
            wire_packages=None,
            wire_classes=None,
        )
        self._root: Path = workspace_root

    @override
    def execute(self) -> r[int]:
        """Execute the lazy-init generation process."""
        return r[int].ok(self.run(check_only=False))

    def run(self, *, check_only: bool = False) -> int:
        """Process all package directories in the workspace.

        Discovers package directories under ``src/``, ``tests/``,
        ``examples/``, and ``scripts/``, processes them bottom-up
        (deepest first), and generates PEP 562 lazy-import
        ``__init__.py`` files.

        Args:
            check_only: If True, only report without writing.

        Returns the number of errors (0 = perfect).

        """
        pkg_dirs = self._find_package_dirs()
        total = ok = errors = 0
        # Bottom-up: child exports computed before parents consume them
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}

        for pkg_dir in pkg_dirs:
            total += 1
            result, exports = self._process_directory(
                pkg_dir,
                check_only=check_only,
                dir_exports=dir_exports,
            )
            if exports:
                dir_exports[str(pkg_dir)] = exports
            if result is None:
                continue
            if result < 0:
                errors += 1
            else:
                ok += 1

        output.info(
            f"Lazy-init summary: {ok} generated, {errors} errors"
            f" ({total} dirs scanned)",
        )
        return errors

    def _find_package_dirs(self) -> list[Path]:
        """Find all package directories that need ``__init__.py``.

        Scans ``src/``, ``tests/``, ``examples/``, ``scripts/`` for
        directories containing ``.py`` files besides ``__init__.py``.

        Returns:
            Sorted by depth (deepest first) for bottom-up processing.

        """
        root_dirs = ("src", "tests", "examples", "scripts")
        pkg_dirs: set[Path] = set()
        for root_name in root_dirs:
            root = self._root / root_name
            if not root.is_dir():
                continue
            files_result = u.Infra.iter_python_files(
                workspace_root=self._root,
                project_roots=[self._root],
                src_dirs=frozenset({root_name}),
            )
            if files_result.is_failure:
                continue
            for py_file in files_result.value:
                if any(
                    part.startswith(".") or part in {"vendor", "node_modules", ".venv"}
                    for part in py_file.parts
                ):
                    continue
                parent = py_file.parent
                if _dir_has_py_files(parent):
                    pkg_dirs.add(parent)
        return sorted(pkg_dirs, key=lambda p: len(p.parts), reverse=True)

    def _process_directory(
        self,
        pkg_dir: Path,
        *,
        check_only: bool,
        dir_exports: Mapping[str, dict[str, tuple[str, str]]],
    ) -> tuple[int | None, dict[str, tuple[str, str]]]:
        """Process a single directory to generate its ``__init__.py``.

        Args:
            pkg_dir: Directory to process.
            check_only: If True, count exports without writing.
            dir_exports: Pre-computed exports from child directories.

        Returns:
            ``(result_code, exports_dict)``.
            result_code is ``None`` if skipped, ``0`` if OK, ``<0`` on error.

        """
        init_path = pkg_dir / "__init__.py"
        current_pkg = _infer_package(init_path)
        if not current_pkg:
            return (None, {})

        # 1. Read ONLY docstring from existing __init__.py
        docstring = _read_existing_docstring(init_path)
        if not docstring:
            docstring = _default_docstring(pkg_dir.name)

        # 2. Build lazy map from sibling .py files
        lazy_map = _build_sibling_export_index(pkg_dir, current_pkg)

        # 3. Add exports from child subdirectories (already computed)
        _merge_child_exports(pkg_dir, lazy_map, dir_exports)

        # 4. Handle __version__.py
        inline_constants, version_lazy = _extract_version_exports(
            pkg_dir,
            current_pkg,
        )
        lazy_map.update(version_lazy)

        # 5. Handle single-letter aliases via ALIAS_TO_SUFFIX
        _resolve_aliases(lazy_map)

        # 6. Remove infrastructure names (eagerly imported, not lazy)
        for infra_name in ("cleanup_submodule_namespace", "lazy_getattr"):
            lazy_map.pop(infra_name, None)

        # 7. Remove inline constants from lazy map (inlined directly)
        for k in inline_constants:
            lazy_map.pop(k, None)

        # 8. Build final exports list
        exports = sorted(set(lazy_map) | set(inline_constants))
        if not exports:
            return (None, dict(lazy_map))

        if check_only:
            return (0, dict(lazy_map))

        # 9. Generate the __init__.py file
        return self._write_init(
            init_path,
            docstring,
            exports,
            lazy_map,
            inline_constants,
            current_pkg,
        )

    def _write_init(
        self,
        init_path: Path,
        docstring: str,
        exports: list[str],
        lazy_map: dict[str, tuple[str, str]],
        inline_constants: dict[str, str],
        current_pkg: str,
    ) -> tuple[int, dict[str, tuple[str, str]]]:
        """Write the generated ``__init__.py`` and run ruff fix."""
        try:
            generated = _generate_file(
                docstring,
                exports,
                lazy_map,
                inline_constants,
                current_pkg,
            )
            init_path.write_text(generated, encoding=c.Infra.Encoding.DEFAULT)
            _run_ruff_fix(init_path)
        except (OSError, ValueError) as exc:
            output.error(f"generating {init_path}: {exc}")
            return (-1, dict(lazy_map))

        rel_path = (
            init_path.relative_to(self._root)
            if self._root in init_path.parents
            else init_path
        )
        output.info(f"  OK: {rel_path} — {len(exports)} exports")
        return (0, dict(lazy_map))


# ---------------------------------------------------------------------------
# Directory / package helpers
# ---------------------------------------------------------------------------


def _dir_has_py_files(pkg_dir: Path) -> bool:
    """Return True if directory has ``.py`` files besides ``__init__.py``."""
    return any(
        f.name != "__init__.py" and f.suffix == ".py"
        for f in pkg_dir.iterdir()
        if f.is_file()
    )


def _infer_package(path: Path) -> str:
    """Infer dotted package name from a path under ``src/``, ``tests/``, etc."""
    abs_path = str(path.absolute())
    for root_dir in ("/src/", "/tests/", "/examples/", "/scripts/"):
        idx = abs_path.rfind(root_dir)
        if idx != -1:
            rel = abs_path[idx + len(root_dir) :]
            pkg_parts = rel.split("/")[:-1]
            if root_dir == "/src/":
                return ".".join(pkg_parts)
            root_name = root_dir.strip("/")
            return ".".join([root_name, *pkg_parts]) if pkg_parts else root_name
    return ""


# ---------------------------------------------------------------------------
# Source-file scanning (the core of auto-discovery)
# ---------------------------------------------------------------------------


def _default_docstring(dir_name: str) -> str:
    """Generate a default module docstring from directory name."""
    label = dir_name.replace("_", " ").replace("-", " ").strip()
    return f'"""{label.capitalize()} package."""'


def _read_existing_docstring(init_path: Path) -> str:
    """Read ONLY the docstring from an existing ``__init__.py``.

    This is the **only** thing we read from existing ``__init__.py`` files.
    Everything else is auto-discovered from sibling ``.py`` source files.
    """
    if not init_path.exists():
        return ""
    try:
        content = init_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    # NOTE: source text needed below - cannot delegate to u.Infra.parse_module_ast
    tree = u.Infra.parse_ast_from_source(content)
    if tree is None:
        return ""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        ds = tree.body[0]
        return "\n".join(content.splitlines()[ds.lineno - 1 : ds.end_lineno])
    return ""


def _build_sibling_export_index(
    pkg_dir: Path,
    current_pkg: str,
) -> dict[str, tuple[str, str]]:
    """Scan sibling ``.py`` files for exports.

    For each non-private, non-dunder sibling ``.py`` file:

    1. If it has ``__all__`` → use those names.
    2. If no ``__all__`` → scan AST for public classes/functions/assignments.

    Returns ``{export_name: (module_path, attr_name)}``.
    """
    index: dict[str, tuple[str, str]] = {}
    for py_file in sorted(pkg_dir.glob("*.py")):
        if py_file.name in {"__init__.py", "__main__.py", "__version__.py"}:
            continue
        if py_file.name.startswith("_"):
            continue
        if py_file.stem[0:1].isdigit():
            continue

        mod_stem = py_file.stem
        mod_path = f"{current_pkg}.{mod_stem}" if current_pkg else mod_stem
        sibling_tree = u.Infra.parse_module_ast(py_file)
        if sibling_tree is None:
            output.warning(f"skipping {py_file.name}: parse failed")
            continue

        # Prefer __all__ when available
        has_all, all_exports = _extract_exports(sibling_tree)
        if has_all and all_exports:
            for name in all_exports:
                index[name] = (mod_path, name)
        else:
            _scan_ast_public_defs(sibling_tree, mod_path, index)

    return index


def _extract_exports(tree: ast.Module) -> tuple[bool, list[str]]:
    """Extract ``__all__`` entries from the AST."""
    exports: list[str] = []
    has_all = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    has_all = True
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        exports.extend(
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                            and isinstance(elt.value, str)
                        )
    return (has_all, exports)


def _scan_ast_public_defs(
    tree: ast.Module,
    mod_path: str,
    index: dict[str, tuple[str, str]],
) -> None:
    """Scan AST for public classes, functions, and assignments."""
    for node in ast.iter_child_nodes(tree):
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        for name in names:
            if not name.startswith("_"):
                index[name] = (mod_path, name)


def _extract_inline_constants(tree: ast.Module) -> dict[str, str]:
    """Extract inline constant assignments like ``__version__ = '1.0.0'``."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                constants[target.id] = node.value.value
    return constants


# ---------------------------------------------------------------------------
# Child-directory export merging (bottom-up)
# ---------------------------------------------------------------------------


def _should_bubble_up(name: str) -> bool:
    """Check if an export should bubble up to the parent package.

    Filters out private names, entry points, and ALL_CAPS utility constants.
    """
    if name.startswith("_"):
        return False
    if name == "main":
        return False
    # Skip ALL_CAPS constants (e.g., BLUE, BOLD, SYM_ARROW)
    return not name.isupper()


def _merge_child_exports(
    pkg_dir: Path,
    lazy_map: dict[str, tuple[str, str]],
    dir_exports: Mapping[str, dict[str, tuple[str, str]]],
) -> None:
    """Merge child subdirectory exports into parent's lazy map.

    For each immediate subdirectory that has computed exports,
    add their exports to the parent.  Sibling file exports take
    precedence over child exports (already in ``lazy_map``).
    """
    for subdir in sorted(pkg_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        subdir_key = str(subdir)
        if subdir_key not in dir_exports:
            continue
        sub_exports = dir_exports[subdir_key]
        for name, (mod, attr) in sub_exports.items():
            if not _should_bubble_up(name):
                continue
            # Sibling file exports take precedence
            if name not in lazy_map:
                lazy_map[name] = (mod, attr)


# ---------------------------------------------------------------------------
# Version and alias resolution
# ---------------------------------------------------------------------------


def _extract_version_exports(
    pkg_dir: Path,
    current_pkg: str,
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Extract version-related exports from ``__version__.py``.

    Returns:
        ``(inline_constants, lazy_entries)``.
        ``inline_constants``: String constants to inline (``__version__``).
        ``lazy_entries``: Non-string dunder constants for lazy loading
        (``__version_info__``).

    """
    ver_file = pkg_dir / "__version__.py"
    if not ver_file.exists():
        return ({}, {})
    tree = u.Infra.parse_module_ast(ver_file)
    if tree is None:
        return ({}, {})

    inline = _extract_inline_constants(tree)
    ver_mod = f"{current_pkg}.__version__" if current_pkg else "__version__"

    lazy: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id.startswith("__")
                and target.id.endswith("__")
                and target.id not in inline
            ):
                lazy[target.id] = (ver_mod, target.id)

    return (inline, lazy)


def _resolve_aliases(lazy_map: dict[str, tuple[str, str]]) -> None:
    """Resolve single-letter aliases from ``ALIAS_TO_SUFFIX`` mapping.

    For each alias (``c``, ``m``, ``t``, ``u``, ``p``, ``r``, ``d``,
    ``e``, ``h``, ``s``, ``x``), find a class in the lazy_map whose name
    ends with the corresponding suffix and create a mapping.
    """
    for alias, suffix in c.Infra.Codegen.ALIAS_TO_SUFFIX.items():
        if alias in lazy_map:
            continue
        for name, (mod, _attr) in list(lazy_map.items()):
            if name.endswith(suffix) and len(name) > 1:
                lazy_map[alias] = (mod, name)
                break


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _generate_type_checking(groups: Mapping[str, list[tuple[str, str]]]) -> list[str]:
    """Generate the ``if TYPE_CHECKING`` import block.

    Groups imports by top-level package with blank lines between groups,
    following isort conventions.
    """
    lines: list[str] = ["if TYPE_CHECKING:"]
    if not groups:
        lines.append("    pass")
        return lines

    def _emit_module(mod: str) -> None:
        items = groups[mod]
        sorted_items = sorted(items, key=lambda x: (x[1], x[0] != x[1]))
        parts: list[str] = []
        for export_name, attr_name in sorted_items:
            if export_name == attr_name:
                parts.append(export_name)
            else:
                parts.append(f"{attr_name} as {export_name}")
        joined = ", ".join(parts)
        line = f"    from {mod} import {joined}"
        if len(line) > c.Infra.Codegen.MAX_LINE_LENGTH:
            lines.append(f"    from {mod} import (")
            lines.extend(f"        {part}," for part in parts)
            lines.append("    )")
        else:
            lines.append(line)

    sorted_mods = sorted(groups, key=str.lower)
    prev_top: str | None = None
    for mod in sorted_mods:
        top = mod.split(".")[0]
        if prev_top is not None and top != prev_top:
            lines.append("")
        _emit_module(mod)
        prev_top = top
    return lines


def _generate_file(
    docstring_source: str,
    exports: list[str],
    filtered: Mapping[str, tuple[str, str]],
    inline_constants: Mapping[str, str],
    current_pkg: str,
) -> str:
    """Generate the complete ``__init__.py`` content."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for export_name in sorted(filtered):
        mod, attr = filtered[export_name]
        groups[mod].append((export_name, attr))

    out: list[str] = [c.Infra.Codegen.AUTOGEN_HEADER]
    if docstring_source:
        out.extend([docstring_source, ""])

    if current_pkg == c.Infra.Packages.CORE_UNDERSCORE:
        lazy_import = (
            "from flext_core._utilities.lazy import"
            " cleanup_submodule_namespace, lazy_getattr"
        )
    else:
        lazy_import = (
            "from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr"
        )

    out.extend([
        "from __future__ import annotations",
        "",
        "from typing import TYPE_CHECKING, Any",
        "",
        lazy_import,
        "",
    ])
    out.extend(_generate_type_checking(groups))
    out.append("")
    for name, value in sorted(inline_constants.items()):
        out.append(f'{name} = "{value}"')
    if inline_constants:
        out.append("")
    out.extend([
        "# Lazy import mapping: export_name -> (module_path, attr_name)",
        "_LAZY_IMPORTS: dict[str, tuple[str, str]] = {",
    ])
    for exp in sorted(exports):
        if exp in filtered:
            mod, attr = filtered[exp]
            out.append(f'    "{exp}": ("{mod}", "{attr}"),')
    out.extend(["}", ""])
    out.append("__all__ = [")
    out.extend(f'    "{exp}",' for exp in sorted(exports))
    out.extend(["]", "", ""])
    out.extend([
        "def __getattr__(name: str) -> Any:",
        '    """Lazy-load module attributes on first access (PEP 562)."""',
        "    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)",
        "",
        "",
        "def __dir__() -> list[str]:",
        '    """Return list of available attributes for dir() and autocomplete."""',
        "    return sorted(__all__)",
        "",
        "",
        "cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)",
        "",
    ])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Post-generation cleanup
# ---------------------------------------------------------------------------


def _run_ruff_fix(path: Path) -> None:
    """Run ``ruff --fix`` on the given file to auto-fix lint issues."""
    with contextlib.suppress(FileNotFoundError):
        runner = FlextInfraUtilitiesSubprocess()
        runner.run_checked([
            c.Infra.Cli.RUFF,
            c.Infra.Cli.RuffCmd.CHECK,
            "--fix",
            "--quiet",
            str(path),
        ])
