"""Shared text/path helpers for namespace refactor utilities."""

from __future__ import annotations

import operator
import token
import tokenize
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra._utilities.rope_source import FlextInfraUtilitiesRopeSource
from flext_infra.constants import c
from flext_infra.iteration import FlextInfraUtilitiesIteration

if TYPE_CHECKING:
    from flext_infra.typings import t


class FlextInfraUtilitiesRefactorNamespaceCommon:
    """Shared text and path helpers for namespace refactor utilities."""

    @staticmethod
    def shared_repository_root(*, py_files: t.SequenceOf[Path]) -> Path:
        """Shared repository root."""
        existing_files = [path.resolve() for path in py_files if path.exists()]
        if not existing_files:
            return Path.cwd()
        project_root = FlextInfraUtilitiesIteration.resolve_project_root(
            existing_files[0]
        )
        return (
            project_root.parent
            if project_root is not None
            else existing_files[0].parent
        )

    @staticmethod
    def _parse_simple_from_import_line(line: str) -> t.Infra.TransformResult | None:
        """Parse simple from import line."""
        stripped = line.strip()
        if (
            not stripped.startswith("from ")
            or "(" in stripped
            or ")" in stripped
            or " as " in stripped
        ):
            return None
        module_name, separator, imported_names = stripped.removeprefix(
            "from "
        ).partition(" import ")
        if not separator or not module_name or not imported_names:
            return None
        names = [name.strip() for name in imported_names.split(",") if name.strip()]
        return (module_name, names)

    @staticmethod
    def insert_import_lines(
        *, lines: t.StrSequence, imports: t.StrSequence
    ) -> t.StrSequence:
        """Insert import lines."""
        if not imports:
            return list(lines)
        insert_idx = (
            FlextInfraUtilitiesRopeSource.index_after_docstring_and_future_imports(
                lines
            )
        )
        return [*lines[:insert_idx], *imports, *lines[insert_idx:]]

    @staticmethod
    def canonical_target_file(
        *, project_root: Path, source_file: Path, filename: str
    ) -> Path:
        """Canonical target file."""
        parts = source_file.parts
        src_dir: str = c.Infra.DEFAULT_SRC_DIR
        if src_dir in parts:
            src_index = parts.index(src_dir)
            if src_index + 1 < len(parts):
                package_name = parts[src_index + 1]
                return project_root / src_dir / package_name / filename
        return source_file.parent / filename

    @staticmethod
    def find_top_level_block(
        *, lines: t.StrSequence, header: str
    ) -> tuple[int, int] | None:
        """Find top level block."""
        start_idx = -1
        for idx, line in enumerate(lines):
            if line.startswith(header):
                start_idx = idx
                break
        if start_idx < 0:
            return None
        end_idx = len(lines)
        for idx in range(start_idx + 1, len(lines)):
            line = lines[idx]
            if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
                end_idx = idx
                break
        return (start_idx, end_idx)

    @staticmethod
    def compat_assignment_target(line: str, *, alias_map: t.StrMapping) -> str | None:
        """Compat assignment target."""
        stripped = line.strip()
        if "=" not in stripped or stripped.startswith("#"):
            return None
        left, right = [part.strip() for part in stripped.split("=", 1)]
        return left if alias_map.get(left) == right else None

    @staticmethod
    def apply_token_replacements(*, source: str, alias_map: t.StrMapping) -> str:
        """Apply token replacements."""
        line_buffer = source.splitlines(keepends=True)
        replacements_by_line: t.MappingKV[
            int, t.MutableSequenceOf[t.Triple[int, int, str]]
        ] = defaultdict(list)
        token_generator = tokenize.generate_tokens(StringIO(source).readline)
        for tok in token_generator:
            if tok.type != token.NAME:
                continue
            replacement = alias_map.get(tok.string)
            if replacement is None:
                continue
            start_line, start_col = tok.start
            end_line, end_col = tok.end
            if start_line != end_line:
                continue
            replacements_by_line[start_line - 1].append((
                start_col,
                end_col,
                replacement,
            ))
        for line_idx, replacements in replacements_by_line.items():
            if line_idx < 0 or line_idx >= len(line_buffer):
                continue
            line_text = line_buffer[line_idx]
            for start_col, end_col, replacement in sorted(
                replacements, key=operator.itemgetter(0), reverse=True
            ):
                line_text = line_text[:start_col] + replacement + line_text[end_col:]
            line_buffer[line_idx] = line_text
        return "".join(line_buffer)


__all__: list[str] = ["FlextInfraUtilitiesRefactorNamespaceCommon"]
