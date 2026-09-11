"""Compatibility alias remover transformer.

Removes module-level ``Alias = Target`` compatibility assignments and rewrites
non-canonical facade imports to their canonical short aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, override

from flext_infra import c, u
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorCompatibilityAlias(FlextInfraRopeTransformer):
    """Remove compatibility aliases and rewrite references to canonical names."""

    _description = "remove compatibility aliases and rewrite references"

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply compatibility alias removal to source text."""
        self.changes.clear()
        updated = source
        updated = self._rewrite_compat_imports(updated)
        updated = self._rewrite_compat_assignments(updated)
        return updated, list(self.changes)

    def _collect_compat_assignments(self, source: str) -> dict[str, str]:
        """Detect ``Alias = Target`` compatibility assignments."""
        alias_map: dict[str, str] = {}
        for match in c.Infra.COMPAT_ALIAS_RE.finditer(source):
            alias_name, target_name = match.group(1), match.group(2)
            if alias_name in c.Infra.COMPAT_SKIP_NAMES or alias_name == target_name:
                continue
            if alias_name.isupper() and target_name.isupper():
                continue
            alias_map[alias_name] = target_name
        return alias_map

    def _rewrite_compat_assignments(self, source: str) -> str:
        """Remove compatibility assignments and rewrite references."""
        alias_map = self._collect_compat_assignments(source)
        if not alias_map:
            return source

        lines = source.splitlines(keepends=True)
        kept_lines: list[str] = []
        removed: set[str] = set()
        for line in lines:
            alias_name = u.Infra.compat_assignment_target(line, alias_map=alias_map)
            if alias_name is not None:
                removed.add(alias_name)
                continue
            kept_lines.append(line)

        updated = "".join(kept_lines)
        rewritten: str = u.Infra.apply_token_replacements(
            source=updated, alias_map=alias_map
        )
        updated = rewritten
        for alias_name in sorted(removed):
            self._record_change(
                f"Removed compatibility alias: {alias_name} = {alias_map[alias_name]}"
            )
        return updated

    def _rewrite_compat_imports(self, source: str) -> str:
        """Rewrite ``from <pkg> import LongFacadeName`` to canonical aliases."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        protected_targets = frozenset(self._collect_compat_assignments(source).values())
        alias_renames = c.ENFORCEMENT_COMPATIBILITY_ALIAS_RENAMES
        alias_map: dict[str, str] = {}
        existing_names = self._collect_existing_names(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                long_name = alias.name
                canonical = alias_renames.get(long_name)
                if canonical is None or alias.asname is not None:
                    continue
                if long_name in protected_targets:
                    # The long name is the target of a local compatibility alias;
                    # rewriting it would break that alias.
                    continue
                if canonical in existing_names and canonical != long_name:
                    # Avoid shadowing an existing name in the file.
                    continue
                alias_map[long_name] = canonical

        if not alias_map:
            return source

        updated = u.Infra.apply_token_replacements(source=source, alias_map=alias_map)
        updated = self._deduplicate_import_names(updated)
        for long_name in sorted(alias_map):
            self._record_change(
                f"Rewrote compatibility import: {long_name} -> {alias_map[long_name]}"
            )
        return updated

    @staticmethod
    def _collect_existing_names(tree: ast.AST) -> set[str]:
        """Collect names defined or referenced in the module."""
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                names.add(node.name)
        return names

    @staticmethod
    def _deduplicate_import_names(source: str) -> str:
        """Remove duplicate names within single-line ``from ... import`` lines."""
        lines = source.splitlines(keepends=True)
        result: list[str] = []
        expected_import_parts = 2
        for line in lines:
            stripped = line.lstrip()
            if not stripped.startswith("from ") or "(" in stripped:
                result.append(line)
                continue
            parts = stripped.removeprefix("from ").split(" import ", 1)
            if len(parts) != expected_import_parts:
                result.append(line)
                continue
            module, names_part = parts
            names = [name.strip() for name in names_part.split(",") if name.strip()]
            if not names:
                result.append(line)
                continue
            seen: set[str] = set()
            unique: list[str] = []
            for name in names:
                if name not in seen:
                    seen.add(name)
                    unique.append(name)
            if len(unique) == len(names):
                result.append(line)
                continue
            result.append(f"from {module} import {', '.join(unique)}\n")
        return "".join(result)


__all__: list[str] = ["FlextInfraRefactorCompatibilityAlias"]
