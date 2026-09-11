"""Import modernizer transformer -- rope-based implementation.

Removes forbidden imports, replaces symbol usages with runtime alias paths,
and adds missing runtime alias imports to the module header.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c, u
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorImportModernizer(FlextInfraRopeTransformer):
    """Rewrite forbidden imports and replace symbols with runtime alias paths."""

    def __init__(
        self,
        imports_to_remove: t.StrSequence,
        symbols_to_replace: t.StrMapping,
        runtime_aliases: t.Infra.StrSet,
        blocked_aliases: t.Infra.StrSet,
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize import rewrite configuration."""
        super().__init__(on_change=on_change)
        self._imports_to_remove = frozenset(imports_to_remove)
        self._symbols_to_replace = dict(symbols_to_replace)
        self._runtime_aliases = runtime_aliases
        self._blocked_aliases = blocked_aliases
        self.modified_imports = False
        self.aliases_needed: t.Infra.StrSet = set()
        self.aliases_present: t.Infra.StrSet = set()
        self.active_symbol_replacements: dict[str, str] = {}

    @override
    def transform(
        self, rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.TransformResult:
        """Apply import modernization via rope utilities."""
        source = resource.read()
        updated, changes = self.apply_to_source(source)
        if updated != source and changes:
            resource.write(updated)
        return updated, changes

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply import modernization to source text."""
        self.modified_imports = False
        self.aliases_needed.clear()
        self.aliases_present.clear()
        self.active_symbol_replacements.clear()
        self.changes.clear()
        updated = source
        self._scan_core_aliases(updated)
        updated = self._rewrite_forbidden_imports(updated)
        updated = self._replace_symbol_usages(updated)
        updated = self._inject_missing_aliases(updated)
        return updated, list(self.changes)

    def _scan_core_aliases(self, source: str) -> None:
        """Scan source for existing core alias imports."""
        core_pkg = c.Infra.PKG_CORE_UNDERSCORE
        self.aliases_present.update(
            u.Infra.collect_from_import_bound_names(
                source, module_name=core_pkg
            ).intersection(self._runtime_aliases)
        )

    def _rewrite_forbidden_imports(self, source: str) -> str:
        """Remove or trim forbidden import lines via regex."""
        lines = source.splitlines(keepends=True)
        result: t.MutableSequenceOf[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            next_i, rewritten, handled = self._consume_forbidden_multiline_import(
                lines, i, stripped
            )
            if handled:
                if rewritten is not None:
                    result.append(rewritten)
                i = next_i
                continue
            rewritten_single = self._rewrite_forbidden_single_line_import(
                line, stripped
            )
            if rewritten_single is not line:
                if rewritten_single is not None:
                    result.append(rewritten_single)
                i += 1
                continue
            result.append(line)
            i += 1
        return "".join(result)

    def _consume_forbidden_multiline_import(
        self, lines: t.StrSequence, start: int, stripped_line: str
    ) -> tuple[int, str | None, bool]:
        """Consume one forbidden parenthesized import statement when present."""
        from_match = c.Infra.FROM_IMPORT_CAPTURE_PAREN_OPEN_RE.match(stripped_line)
        if from_match is None:
            return start, None, False
        module = from_match.group(1)
        if module not in self._imports_to_remove:
            return start, None, False
        end = start
        import_lines = [lines[start]]
        while end + 1 < len(lines) and ")" not in lines[end]:
            end += 1
            import_lines.append(lines[end])
        full_text = "".join(import_lines)
        return end + 1, self._filter_import_names(module, full_text), True

    def _rewrite_forbidden_single_line_import(
        self, original_line: str, stripped_line: str
    ) -> str | None:
        """Rewrite one forbidden single-line import or return the original line."""
        from_single = c.Infra.FROM_IMPORT_LINE_TRIM_RE.match(stripped_line)
        if from_single is None:
            return original_line
        module = from_single.group(1)
        if module not in self._imports_to_remove:
            return original_line
        return self._filter_import_names(module, original_line)

    def _filter_import_names(self, module: str, import_text: str) -> str | None:
        """Filter names from an import statement. Returns None to remove entirely."""
        # Extract all names from the import text
        names_part = (
            import_text.split("import", 1)[1] if "import" in import_text else ""
        )
        names_part = names_part.strip().strip("()")
        mapped: t.MutableSequenceOf[str] = []
        unmapped: t.MutableSequenceOf[str] = []
        for bare_name, bound in u.Infra.parse_import_names(names_part):
            if bare_name not in self._symbols_to_replace:
                unmapped.append(
                    bare_name if bare_name == bound else f"{bare_name} as {bound}"
                )
                continue
            alias_path = self._symbols_to_replace[bare_name]
            alias_root = alias_path.split(".")[0]
            if alias_root in self._blocked_aliases:
                unmapped.append(
                    bare_name if bare_name == bound else f"{bare_name} as {bound}"
                )
                continue
            self.active_symbol_replacements[bound] = alias_path
            self.aliases_needed.add(alias_root)
            mapped.append(bare_name)
        if not mapped:
            return import_text
        self.modified_imports = True
        self._record_change(f"Removed import: from {module}")
        if unmapped:
            return f"from {module} import {', '.join(unmapped)}\n"
        return None

    def _replace_symbol_usages(self, source: str) -> str:
        """Replace migrated symbol references with runtime-alias paths."""
        for local_name, alias_path in self.active_symbol_replacements.items():
            new_source = c.Infra.compile_word(local_name).sub(alias_path, source)
            if new_source != source:
                self._record_change(f"Replaced: {local_name} -> {alias_path}")
                source = new_source
        return source

    def _inject_missing_aliases(self, source: str) -> str:
        """Insert any newly required runtime alias imports."""
        missing = sorted(self.aliases_needed - self.aliases_present)
        if not (self.modified_imports and missing):
            return source
        pkg = c.Infra.PKG_CORE_UNDERSCORE
        import_line = f"from {pkg} import {', '.join(missing)}\n"
        self._record_change(f"Added: from flext_core import {', '.join(missing)}")
        lines = source.splitlines(keepends=True)
        idx = u.Infra.find_import_insert_position(lines, past_existing=False)
        lines.insert(idx, import_line)
        return "".join(lines)


__all__: list[str] = ["FlextInfraRefactorImportModernizer"]
