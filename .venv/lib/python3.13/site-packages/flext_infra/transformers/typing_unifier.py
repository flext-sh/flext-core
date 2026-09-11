"""Unify inline typing unions and TypeAlias declarations to canonical forms.

All transformations are syntactic and driven by ``c.Infra.*_RE`` regex
constants — no ``ast`` parsing or tree walking is required:

- **Inline-union canonicalization**: replaces permutations like
  ``int | str`` with the canonical ``t.<Alias>`` (configured via
  ``canonical_map``).
- **Built-in annotation canonicalization**: rewrites ``dict[K, V]`` →
  ``t.MappingKV[K, V]``, ``list[X]`` → ``t.SequenceOf[X]``, and bare
  ``Any``/``object`` / ``typing.Any`` → ``t.JsonValue``. The bracket
  forms guard against false positives by requiring an immediate ``[``.
- **PEP 695 TypeAlias modernization**: rewrites ``X: TypeAlias = expr``
  into ``type X = expr``.
- **Canonical ``t`` import injection**: adds ``from <pkg> import t``
  after the last import line when ``t.`` is used but the import is
  missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c
from flext_infra.transformers._canonical_t_import import (
    FlextInfraEnsureCanonicalTImportMixin,
)
from flext_infra.transformers._typing_rewrite import (
    FlextInfraRefactorTypingUnifierRewriteMixin,
)
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraRefactorTypingUnifier(
    FlextInfraEnsureCanonicalTImportMixin,
    FlextInfraRopeTransformer,
    FlextInfraRefactorTypingUnifierRewriteMixin,
):
    """Unify inline type unions and modernize TypeAlias to PEP 695."""

    _description = "canonicalize types and modernize TypeAlias"

    def __init__(
        self,
        *,
        canonical_map: t.MappingKV[frozenset[str], str],
        file_path: Path | None = None,
    ) -> None:
        """Initialize with canonical union map and optional file path for skip logic."""
        super().__init__()
        self._canonical_map = canonical_map
        self._file_path = file_path
        self._is_definition_file = self._is_typing_definition_file(file_path)

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply unions, built-in canonicalization, TypeAlias and t import."""
        if self._is_definition_file:
            return source, list(self.changes)

        def union_size(item: tuple[frozenset[str], str]) -> int:
            return len(item[0])

        for member_set, canonical in sorted(
            self._canonical_map.items(), key=union_size, reverse=True
        ):
            pattern = self._union_pattern(member_set)
            if pattern is None:
                continue

            def replacer(match: t.Infra.RegexMatch, canonical: str = canonical) -> str:
                """Replace one matched union with the canonical alias."""
                self._record_change(
                    f"Canonicalized inline union {match.group(0)} -> {canonical}"
                )
                return canonical

            source, _count = pattern.subn(replacer, source)
        source = self._canonicalize_annotation_builtins(source)
        source = self._modernize_typealias(source)
        added, did_add = self._ensure_t_import(
            source, self._canonical_import_module(self._file_path)
        )
        if did_add:
            self._record_change(
                "Added canonical t import from "
                f"{self._canonical_import_module(self._file_path)}"
            )
        source = added
        return source, list(self.changes)

    def _canonicalize_annotation_builtins(self, source: str) -> str:
        """Rewrite built-in generic annotations to canonical ``t.*`` aliases."""
        rewritten, changes = self._rewrite_annotation_text(source)
        for change in changes:
            self._record_change(change)
        return rewritten

    @staticmethod
    def _union_pattern(members: frozenset[str]) -> t.Infra.RegexPattern | None:
        """Build regex matching any permutation of a ``A | B | C`` union."""
        if len(members) < c.Infra.MIN_UNION_MEMBERS:
            return None
        escaped = [c.Infra.escape(m) for m in sorted(members)]
        part = rf"(?:{'|'.join(escaped)})"
        pattern: t.Infra.RegexPattern = c.Infra.compile(
            rf"\b{part}(?:\s*\|\s*{part}){{{len(members) - 1}}}\b"
        )
        return pattern

    def _modernize_typealias(self, source: str) -> str:
        """Convert ``X: TypeAlias = expr`` to ``type X = expr`` (PEP 695)."""
        for match in c.Infra.LEGACY_TYPEALIAS_RE.finditer(source):
            self._record_change(
                f"Converted legacy TypeAlias assignment: {match.group(1)}"
            )
        new_source: str = c.Infra.LEGACY_TYPEALIAS_RE.sub(r"type \1 = \2", source)
        return new_source

    @staticmethod
    def _is_typing_definition_file(file_path: Path | None) -> bool:
        """Return whether ``file_path`` is one of the typing definition files."""
        if file_path is None:
            return False
        if file_path.name in c.Infra.TYPING_DEFINITION_FILES:
            return True
        return any(part in c.Infra.TYPING_DEFINITION_FILES for part in file_path.parts)


__all__: list[str] = ["FlextInfraRefactorTypingUnifier"]
