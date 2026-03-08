"""Code generation helpers for infrastructure lazy-init processing.

Centralizes AST/codegen helpers previously defined as module-level
functions in ``flext_infra.codegen.lazy_init``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import contextlib
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from flext_infra.constants import c
from flext_infra.subprocess import FlextInfraCommandRunner


class FlextInfraUtilitiesCodegen:
    """Code generation helpers for lazy-init and AST operations.

    Usage via namespace::

        from flext_infra import u

        pkg = u.Infra.Codegen.infer_package(path)
    """

    @staticmethod
    def infer_package(path: Path) -> str:
        """Infer the package name from a ``src/<pkg>/__init__.py`` path.

        Args:
            path: Path to an ``__init__.py`` file.

        Returns:
            Dotted package name, or empty string if not under ``src/``.

        """
        abs_path = str(path.absolute())
        src_idx = abs_path.rfind("/src/")
        if src_idx != -1:
            rel = abs_path[src_idx + 5 :]
            pkg_parts = rel.split("/")[:-1]
            return ".".join(pkg_parts)
        return ""

    @staticmethod
    def resolve_module(raw_module: str, level: int, current_pkg: str) -> str:
        """Resolve a potentially relative import to an absolute module path.

        Args:
            raw_module: The raw module name from the import statement.
            level: Number of leading dots (relative import level).
            current_pkg: The current package dotted name.

        Returns:
            Absolute module path string.

        """
        if level == 0:
            return raw_module
        if not current_pkg:
            return raw_module
        parts = current_pkg.split(".")
        base = parts[: len(parts) - level + 1]
        if not base:
            return raw_module
        return ".".join(base) + ("." + raw_module if raw_module else "")

    @staticmethod
    def extract_docstring_source(tree: ast.Module, content: str) -> str:
        """Extract the raw docstring source preserving original formatting.

        Args:
            tree: Parsed AST module.
            content: Original file content string.

        Returns:
            Docstring source text, or empty string if no docstring.

        """
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            ds = tree.body[0]
            return "\n".join(content.splitlines()[ds.lineno - 1 : ds.end_lineno])
        return ""

    @staticmethod
    def extract_exports(tree: ast.Module) -> tuple[bool, list[str]]:
        """Extract ``__all__`` entries from the AST.

        Args:
            tree: Parsed AST module.

        Returns:
            Tuple of (has_all, list_of_export_names).

        """
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

    @staticmethod
    def extract_inline_constants(tree: ast.Module) -> dict[str, str]:
        """Extract inline constant assignments like ``__version__ = '1.0.0'``.

        Args:
            tree: Parsed AST module.

        Returns:
            Dictionary mapping constant names to their string values.

        """
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

    @staticmethod
    def parse_existing_lazy_imports(tree: ast.Module) -> dict[str, tuple[str, str]]:
        """Parse an existing ``_LAZY_IMPORTS`` dict literal from the AST.

        Handles both ``_LAZY_IMPORTS = {...}`` and
        ``_LAZY_IMPORTS: dict[str, tuple[str, str]] = {...}``.

        Args:
            tree: Parsed AST module.

        Returns:
            Dictionary mapping export names to (module_path, attr_name) tuples.

        """
        result: dict[str, tuple[str, str]] = {}
        lazy_import_pair_size = 2

        def _extract(d: ast.expr) -> None:
            if not isinstance(d, ast.Dict):
                return
            for key, val in zip(d.keys, d.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(val, ast.Tuple)
                    and (len(val.elts) == lazy_import_pair_size)
                    and isinstance(val.elts[0], ast.Constant)
                    and isinstance(val.elts[1], ast.Constant)
                ):
                    result[str(key.value)] = (
                        str(val.elts[0].value),
                        str(val.elts[1].value),
                    )

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_LAZY_IMPORTS":
                        _extract(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and (node.target.id == "_LAZY_IMPORTS")
                and (node.value is not None)
            ):
                _extract(node.value)
        return result

    @staticmethod
    def derive_lazy_map(
        tree: ast.Module, current_pkg: str
    ) -> dict[str, tuple[str, str]]:
        """Derive lazy import mappings from import statements in the AST.

        Args:
            tree: Parsed AST module.
            current_pkg: Current package dotted name.

        Returns:
            Dictionary mapping export names to (module_path, attr_name) tuples.

        """
        lazy_map: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                raw_module = node.module or ""
                if raw_module in c.Infra.Codegen.SKIP_MODULES:
                    continue
                module_path = FlextInfraUtilitiesCodegen.resolve_module(
                    raw_module, node.level, current_pkg
                )
                if module_path == current_pkg:
                    continue
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    lazy_map[asname] = (module_path, name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    if name in c.Infra.Codegen.SKIP_STDLIB:
                        continue
                    lazy_map[asname] = (name, "")
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                rhs = node.value.id
                if rhs in lazy_map:
                    mod, attr = lazy_map[rhs]
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            lazy_map[target.id] = (mod, attr)
        for a_name, suffix in c.Infra.Codegen.ALIAS_TO_SUFFIX.items():
            if a_name not in lazy_map:
                continue
            alias_mod, alias_attr = lazy_map[a_name]
            if alias_attr == a_name:
                for name, (mod, _) in lazy_map.items():
                    if mod == alias_mod and name.endswith(suffix) and (len(name) > 1):
                        lazy_map[a_name] = (mod, name)
                        break
        return lazy_map

    @staticmethod
    def resolve_unmapped(
        exports_set: set[str],
        filtered: dict[str, tuple[str, str]],
        current_pkg: str,
        pkg_dir: Path,
    ) -> None:
        """Resolve unmapped single-letter aliases and ``__version__``.

        Mutates ``filtered`` in place to add resolved mappings.

        Args:
            exports_set: Set of all export names.
            filtered: dict of already-mapped exports (modified in place).
            current_pkg: Current package dotted name.
            pkg_dir: Package directory path.

        """
        unmapped = exports_set - set(filtered)
        if not unmapped:
            return
        for alias in sorted(unmapped):
            if alias in c.Infra.Codegen.ALIAS_TO_SUFFIX:
                suffix = c.Infra.Codegen.ALIAS_TO_SUFFIX[alias]
                for name, (mod, _) in filtered.items():
                    if name.endswith(suffix) and len(name) > 1:
                        filtered[alias] = (mod, name)
                        break
            elif alias == "__version__" and current_pkg:
                ver_file = pkg_dir / "__version__.py"
                if ver_file.exists():
                    filtered["__version__"] = (
                        f"{current_pkg}.__version__",
                        "__version__",
                    )
            elif alias == "__version_info__" and current_pkg:
                ver_file = pkg_dir / "__version__.py"
                if ver_file.exists():
                    filtered["__version_info__"] = (
                        f"{current_pkg}.__version__",
                        "__version_info__",
                    )

    @staticmethod
    def generate_type_checking(
        groups: Mapping[str, list[tuple[str, str]]],
    ) -> list[str]:
        """Generate the ``if TYPE_CHECKING`` import block.

        Groups imports by top-level package with blank lines between groups,
        following isort conventions.

        Args:
            groups: Mapping of module names to list of (export_name, attr_name) tuples.

        Returns:
            List of lines for the TYPE_CHECKING block.

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

    @staticmethod
    def generate_file(
        docstring_source: str,
        exports: list[str],
        filtered: Mapping[str, tuple[str, str]],
        inline_constants: Mapping[str, str],
        current_pkg: str,
    ) -> str:
        """Generate the complete ``__init__.py`` content.

        Args:
            docstring_source: Raw docstring source from the original file.
            exports: List of export names.
            filtered: Mapping of export names to (module_path, attr_name) tuples.
            inline_constants: Mapping of constant names to values.
            current_pkg: Current package dotted name.

        Returns:
            Complete generated ``__init__.py`` file content.

        """
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for export_name in sorted(filtered):
            mod, attr = filtered[export_name]
            groups[mod].append((export_name, attr))
        out: list[str] = []
        if docstring_source:
            out.extend([docstring_source, ""])
        if current_pkg == c.Infra.Packages.CORE_UNDERSCORE:
            lazy_import = "from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr"
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
        out.extend(FlextInfraUtilitiesCodegen.generate_type_checking(groups))
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
            "def __getattr__(name: str) -> Any:  # noqa: ANN401  # JUSTIFIED: Ruff (any-type) with PEP 562 dynamic module exports — https://docs.astral.sh/ruff/rules/any-type/",
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

    @staticmethod
    def run_ruff_fix(path: Path) -> None:
        """Run ``ruff --fix`` on the given file to auto-fix lint issues.

        Args:
            path: Path to the file to fix.

        """
        with contextlib.suppress(FileNotFoundError):
            runner = FlextInfraCommandRunner()
            runner.run_checked([
                c.Infra.Cli.RUFF,
                c.Infra.Cli.RuffCmd.CHECK,
                "--fix",
                "--quiet",
                str(path),
            ])


__all__ = ["FlextInfraUtilitiesCodegen"]
