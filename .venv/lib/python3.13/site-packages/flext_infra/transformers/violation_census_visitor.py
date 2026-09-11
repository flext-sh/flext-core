"""Collect code-pattern violations via regex source analysis.

Replaces CST visitor with regex-based detection for governance violations
(Any usage, bare object, type:ignore, cast, Literal, StrEnum, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraViolationCensusVisitor:
    """Detect governance violations via regex source analysis.

    All regex patterns centralized in ``c.Infra.CensusPatterns``.
    """

    def __init__(self, *, file_path: Path) -> None:
        """Initialize visitor state for one file violation census."""
        self._file_path = file_path
        self.records: list[t.Infra.CensusRecord] = []

    def scan_source(self, source: str) -> None:
        """Scan source text for governance violations."""
        self._check_container_invariance(source)
        self._check_literal_usage(source)
        self._check_cast_calls(source)
        self._check_imports(source)
        self._check_strenum_usage(source)
        self._check_assignments(source)
        self._check_type_aliases(source)

    def _check_container_invariance(self, source: str) -> None:
        """Check container invariance."""
        for _ in c.Infra.CensusPatterns.DICT_INVARIANCE_RE.finditer(source):
            self._add_record(
                kind="container_invariance",
                detail="Found t.MappingKV[str, t.JsonValue|t.JsonValue] style annotation.",
            )

    def _check_literal_usage(self, source: str) -> None:
        """Check literal usage."""
        for _ in c.Infra.CensusPatterns.LITERAL_RE.finditer(source):
            self._add_record(kind="literal_usage", detail="Found Literal[...] usage.")

    def _check_cast_calls(self, source: str) -> None:
        """Check cast calls."""
        for _ in c.Infra.CensusPatterns.CAST_RE.finditer(source):
            self._add_record(kind="redundant_cast", detail="Found cast(...) call.")

    def _check_imports(self, source: str) -> None:
        """Check imports."""
        for hit in c.Infra.CensusPatterns.DIRECT_SUBMODULE_RE.finditer(source):
            module_match = c.Infra.FLEXT_CORE_DIRECT_SUBMODULE_RE.search(hit.group(0))
            module_name = module_match.group(1) if module_match else "flext_core.*"
            self._add_record(
                kind="direct_submodule_import",
                detail=f"Found direct submodule import: from {module_name} import ...",
            )
        for _ in c.Infra.CensusPatterns.LEGACY_MAPPING_RE.finditer(source):
            self._add_record(
                kind="legacy_typing_mapping",
                detail="Found from typing import ... Mapping ...",
            )
        for match in c.Infra.CensusPatterns.FLEXT_CORE_IMPORT_RE.finditer(source):
            imported_names = [n.strip() for n in match.group(1).split(",")]
            if not any(
                name in c.Infra.DETECTION_CANONICAL_ALIASES for name in imported_names
            ):
                self._add_record(
                    kind="runtime_alias_violation",
                    detail="Found from flext_core import ... without runtime aliases.",
                )

    def _check_strenum_usage(self, source: str) -> None:
        """Check strenum usage."""
        for match in c.Infra.CensusPatterns.STRENUM_RE.finditer(source):
            class_name = match.group(1)
            self._add_record(
                kind="strenum_usage",
                detail=f"Class {class_name} inherits from StrEnum.",
            )

    def _check_assignments(self, source: str) -> None:
        """Check assignments."""
        for _ in c.Infra.CensusPatterns.CONSTANT_DICT_RE.finditer(source):
            self._add_record(
                kind="manual_mapping_constant",
                detail="Found constant assigned to dict literal.",
            )
        for match in c.Infra.CensusPatterns.COMPAT_ALIAS_RE.finditer(source):
            target, value = match.group(1), match.group(2)
            if self._is_pascal_case(target) and self._is_pascal_case(value):
                self._add_record(
                    kind="compatibility_alias",
                    detail="Found compatibility alias assignment Name = Name.",
                )

    def _check_type_aliases(self, source: str) -> None:
        """Check type aliases."""
        for _ in c.Infra.CensusPatterns.TYPE_ALIAS_OLD_RE.finditer(source):
            self._add_record(
                kind="manual_typing_alias",
                detail="Found TypeAlias-based manual type alias.",
            )
        for match in c.Infra.CensusPatterns.TYPE_ALIAS_PEP695_RE.finditer(source):
            alias_name = match.group(1)
            self._add_record(
                kind="manual_typing_alias",
                detail=f"Found PEP 695 type alias: {alias_name}.",
            )

    def _add_record(self, *, kind: str, detail: str) -> None:
        """Add record."""
        self.records.append({
            "file": str(self._file_path),
            "line": 0,
            "kind": kind,
            "detail": detail,
        })

    @staticmethod
    def _is_pascal_case(name: str) -> bool:
        """Is pascal case."""
        if not name or "_" in name:
            return False
        if not name[0].isupper():
            return False
        return any(character.islower() for character in name[1:])


__all__: list[str] = ["FlextInfraViolationCensusVisitor"]
