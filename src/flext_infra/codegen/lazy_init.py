"""Lazy-init __init__.py generator (PEP 562).

Reads existing ``__init__.py`` files, extracts ``__all__`` and import mappings,
and generates clean lazy-loading versions using ``flext_core.lazy``.

Handles two cases:

1. Files with existing ``_LAZY_IMPORTS``: parses the dict and regenerates cleanly.
2. Files without ``_LAZY_IMPORTS``: derives mappings from import statements.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import contextlib
import subprocess  # noqa: S404
from collections import defaultdict
from pathlib import Path
from typing import override

from flext_core import FlextService, r

from flext_infra.output import output

_ALIAS_TO_SUFFIX: dict[str, str] = {
    "c": "Constants",
    "d": "Decorators",
    "e": "Exceptions",
    "h": "Handlers",
    "m": "Models",
    "p": "Protocols",
    "r": "Result",
    "s": "Service",
    "t": "Types",
    "u": "Utilities",
    "x": "Mixins",
}

_SKIP_MODULES: frozenset[str] = frozenset({
    "__future__",
    "typing",
    "collections.abc",
    "abc",
})

_SKIP_STDLIB: frozenset[str] = frozenset({
    "sys",
    "importlib",
    "typing",
    "collections",
    "abc",
})

_MAX_LINE_LENGTH = 88


class FlextInfraLazyInitGenerator(FlextService[int]):
    """Generates ``__init__.py`` with PEP 562 lazy imports.

    This service scans ``__init__.py`` files under ``src/`` directories in a
    workspace, extracts ``__all__`` and import mappings, and rewrites them
    using ``flext_core.lazy`` utilities.
    """

    def __init__(self, workspace_root: Path) -> None:  # noqa: D107
        super().__init__()
        self._root: Path = workspace_root

    # -- public API ----------------------------------------------------------
    @override
    def execute(self) -> r[int]:
        """Execute the lazy-init generation process."""
        return r[int].ok(self.run(check_only=False))

    def run(self, *, check_only: bool = False) -> int:
        """Process all ``__init__.py`` files in the workspace.

        Returns the number of files that had unmapped exports (0 = perfect).
        """
        init_files = sorted(
            p
            for p in self._root.rglob("src/**/__init__.py")
            if not any(
                part.startswith(".") or part in {"vendor", "node_modules", ".venv"}
                for part in p.parts
            )
        )

        total = ok = errors = unmapped_count = 0
        for path in init_files:
            total += 1
            result = self._process_file(path, check_only=check_only)
            if result is None:
                continue  # skipped
            if result < 0:
                errors += 1
            elif result > 0:
                unmapped_count += 1
                ok += 1
            else:
                ok += 1

        output.info(
            f"Lazy-init summary: {ok} generated, {errors} errors, "
            f"{unmapped_count} with unmapped exports "
            f"({total} files scanned)",
        )
        return unmapped_count

    # -- private helpers -----------------------------------------------------

    def _process_file(
        self,
        path: Path,
        *,
        check_only: bool = False,
    ) -> int | None:
        """Process a single __init__.py.

        Returns:
            None if skipped, 0 if OK, >0 if unmapped exports, <0 on error.

        """
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            output.error(f"reading {path}: {exc}")
            return -1

        try:
            tree = ast.parse(content)
        except Exception as exc:
            output.error(f"parsing {path}: {exc}")
            return -1

        docstring_source = _extract_docstring_source(tree, content)
        has_all, exports = _extract_exports(tree)
        if not has_all or not exports:
            return None  # skip

        current_pkg = _infer_package(path)

        # Extract inline constants like __version__ = "1.0.0"
        inline_constants = _extract_inline_constants(tree)
        exports_set = set(exports)
        inline_constants = {
            k: v for k, v in inline_constants.items() if k in exports_set
        }

        existing_lazy = _parse_existing_lazy_imports(tree)
        lazy_map = existing_lazy or _derive_lazy_map(tree, current_pkg)

        filtered = {k: v for k, v in lazy_map.items() if k in exports_set}

        for k in inline_constants:
            filtered.pop(k, None)

        if not existing_lazy:
            pkg_dir = path.parent
            _resolve_unmapped(exports_set, filtered, current_pkg, pkg_dir)
            for k in inline_constants:
                filtered.pop(k, None)

        if check_only:
            n_mapped = len(filtered) + len(inline_constants)
            return len(exports_set) - n_mapped

        generated = _generate_file(
            docstring_source,
            exports,
            filtered,
            inline_constants,
            current_pkg,
        )
        path.write_text(generated, encoding="utf-8")

        _run_ruff_fix(path)

        n_mapped = len(filtered) + len(inline_constants)
        n_missing = len(exports_set) - n_mapped
        msg = f"  OK: {path.relative_to(self._root)} — {len(exports)} exports, {n_mapped} mapped"
        if n_missing > 0:
            still_unmapped = sorted(
                exports_set - set(filtered) - set(inline_constants),
            )
            msg += f", {n_missing} UNMAPPED: {still_unmapped}"
        output.info(msg)
        return n_missing


# ---------------------------------------------------------------------------
# Pure functions (no class dependency)
# ---------------------------------------------------------------------------


def _infer_package(path: Path) -> str:
    """Infer the package name from a ``src/<pkg>/__init__.py`` path."""
    abs_path = str(path.absolute())
    src_idx = abs_path.rfind("/src/")
    if src_idx != -1:
        rel = abs_path[src_idx + 5 :]
        pkg_parts = rel.split("/")[:-1]
        return ".".join(pkg_parts)
    return ""


def _resolve_module(raw_module: str, level: int, current_pkg: str) -> str:
    """Resolve a potentially relative import to an absolute module path."""
    if level == 0:
        return raw_module
    if not current_pkg:
        return raw_module
    parts = current_pkg.split(".")
    base = parts[: len(parts) - level + 1]
    if not base:
        return raw_module
    return ".".join(base) + ("." + raw_module if raw_module else "")


def _extract_docstring_source(tree: ast.Module, content: str) -> str:
    """Extract the raw docstring source preserving original formatting."""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        ds = tree.body[0]
        return "\n".join(content.splitlines()[ds.lineno - 1 : ds.end_lineno])
    return ""


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
    return has_all, exports


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


def _parse_existing_lazy_imports(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Parse an existing ``_LAZY_IMPORTS`` dict literal from the AST.

    Handles both ``_LAZY_IMPORTS = {...}`` and
    ``_LAZY_IMPORTS: dict[str, tuple[str, str]] = {...}``.
    """
    result: dict[str, tuple[str, str]] = {}

    def _extract(d: ast.expr) -> None:
        if not isinstance(d, ast.Dict):
            return
        for key, val in zip(d.keys, d.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and isinstance(val, ast.Tuple)
                and len(val.elts) == 2  # noqa: PLR2004
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
            and node.target.id == "_LAZY_IMPORTS"
            and node.value is not None
        ):
            _extract(node.value)

    return result


def _derive_lazy_map(
    tree: ast.Module,
    current_pkg: str,
) -> dict[str, tuple[str, str]]:
    """Derive lazy import mappings from import statements in the AST."""
    lazy_map: dict[str, tuple[str, str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            raw_module = node.module or ""
            if raw_module in _SKIP_MODULES:
                continue
            module_path = _resolve_module(raw_module, node.level, current_pkg)
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
                if name in _SKIP_STDLIB:
                    continue
                lazy_map[asname] = (name, "")

    # Capture assignment aliases: `c = FlextConstants`
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            rhs = node.value.id
            if rhs in lazy_map:
                mod, attr = lazy_map[rhs]
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        lazy_map[target.id] = (mod, attr)

    # Fix single-letter aliases imported alongside facade classes.
    for alias, suffix in _ALIAS_TO_SUFFIX.items():
        if alias not in lazy_map:
            continue
        alias_mod, alias_attr = lazy_map[alias]
        if alias_attr == alias:
            for name, (mod, _) in lazy_map.items():
                if mod == alias_mod and name.endswith(suffix) and len(name) > 1:
                    lazy_map[alias] = (mod, name)
                    break

    return lazy_map


def _resolve_unmapped(
    exports_set: set[str],
    filtered: dict[str, tuple[str, str]],
    current_pkg: str,
    pkg_dir: Path,
) -> None:
    """Resolve unmapped single-letter aliases and ``__version__``."""
    unmapped = exports_set - set(filtered)
    if not unmapped:
        return

    for alias in sorted(unmapped):
        if alias in _ALIAS_TO_SUFFIX:
            suffix = _ALIAS_TO_SUFFIX[alias]
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


def _generate_type_checking(
    groups: dict[str, list[tuple[str, str]]],
) -> list[str]:
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
        sorted_items = sorted(
            items,
            key=lambda x: (x[1], x[0] != x[1]),
        )
        parts: list[str] = []
        for export_name, attr_name in sorted_items:
            if export_name == attr_name:
                parts.append(export_name)
            else:
                parts.append(f"{attr_name} as {export_name}")

        joined = ", ".join(parts)
        line = f"    from {mod} import {joined}"
        if len(line) > _MAX_LINE_LENGTH:
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
    filtered: dict[str, tuple[str, str]],
    inline_constants: dict[str, str],
    current_pkg: str,
) -> str:
    """Generate the complete ``__init__.py`` content."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for export_name in sorted(filtered):
        mod, attr = filtered[export_name]
        groups[mod].append((export_name, attr))

    out: list[str] = []

    if docstring_source:
        out.extend([docstring_source, ""])

    # flext_core itself must use _utilities.lazy (avoid circular import)
    if current_pkg == "flext_core":
        lazy_import = (
            "from flext_core._utilities.lazy import "
            "cleanup_submodule_namespace, lazy_getattr"
        )
    else:
        lazy_import = "from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr"

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
        "def __getattr__(name: str) -> Any:  # noqa: ANN401",
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


def _run_ruff_fix(path: Path) -> None:
    """Run ``ruff --fix`` on the given file to auto-fix lint issues."""
    with contextlib.suppress(FileNotFoundError):
        subprocess.run(  # noqa: S603
            ["ruff", "check", "--fix", "--quiet", str(path)],  # noqa: S607
            check=False,
            capture_output=True,
        )
