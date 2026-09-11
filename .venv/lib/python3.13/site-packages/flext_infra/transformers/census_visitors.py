"""Census usage detection via rope import analysis and regex patterns.

Replaces CST visitors with rope's get_module_imports for import discovery
and regex-based attribute access detection for usage collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraCensusImportDiscoveryVisitor:
    """Discover family alias and direct class imports via rope/regex.

    Parametrized by ``family_alias`` (e.g. ``"u"``) and
    ``facade_class_prefix`` (e.g. ``"FlextUtilities"``).
    """

    def __init__(self, *, family_alias: str, facade_class_prefix: str) -> None:
        """Initialize with family alias and facade class prefix."""
        self.family_alias = family_alias
        self.facade_class_prefix = facade_class_prefix
        self.alias_locals: t.Infra.StrSet = set()
        self.direct_imports: dict[str, str] = {}

    def scan_source(self, source: str) -> None:
        """Scan source text to discover imports matching family/facade patterns."""
        for match in c.Infra.FROM_IMPORT_SIMPLE_RE.finditer(source):
            module_str = match.group(1)
            if (
                c.Infra.PKG_CORE_UNDERSCORE not in module_str
                and c.Infra.PKG_PREFIX_UNDERSCORE not in module_str
            ):
                continue
            names_part = match.group(2).strip().rstrip("\\")
            for raw_name_entry in names_part.split(","):
                name_entry = raw_name_entry.strip()
                if not name_entry:
                    continue
                parts = c.Infra.AS_KEYWORD_RE.split(name_entry, maxsplit=1)
                imported_name = parts[0].strip()
                local_name = parts[1].strip() if len(parts) > 1 else imported_name
                if imported_name in {self.family_alias, self.facade_class_prefix}:
                    self.alias_locals.add(local_name)
                if imported_name.startswith(self.facade_class_prefix) and (
                    c.Infra.PKG_CORE_UNDERSCORE in module_str
                ):
                    self.direct_imports[local_name] = imported_name


class FlextInfraCensusUsageCollector:
    """Detect method accesses via regex attribute resolution.

    Detects three access modes:
    1. ``alias.method_name(...)`` — flat alias via facade
    2. ``alias.ClassName.method_name()`` — namespaced via inner class
    3. ``FlextXxxYyy.method_name()`` — direct class reference
    """

    def __init__(
        self,
        *,
        method_index: t.MappingKV[str, t.Infra.StrSet],
        flat_aliases: t.MappingKV[str, t.StrPair],
        inner_class_map: t.StrMapping,
        alias_locals: t.Infra.StrSet,
        direct_imports: t.StrMapping,
        file_path: Path,
        project_name: str,
    ) -> None:
        """Initialize with method index and import context."""
        self.method_index = method_index
        self.flat_aliases = flat_aliases
        self.inner_class_map = inner_class_map
        self.alias_locals = alias_locals
        self.direct_imports = direct_imports
        self.file_path = file_path
        self.project_name = project_name
        self.records: t.MutableSequenceOf[m.Infra.CensusUsageRecord] = []

    def scan_source(self, source: str) -> None:
        """Scan source text for attribute access patterns."""
        for alias in self.alias_locals:
            self._scan_flat_aliases(source, alias)
            self._scan_namespaced_aliases(source, alias)
        self._scan_direct_references(source)

    def _scan_flat_aliases(self, source: str, alias: str) -> None:
        """Detect alias.method_name patterns."""
        pattern = c.Infra.compile_attr_access(alias)
        for match in pattern.finditer(source):
            method_name = match.group(1)
            if method_name in self.flat_aliases:
                cls, orig = self.flat_aliases[method_name]
                self._record(cls, orig, c.Infra.CensusMode.ALIAS_FLAT)

    def _scan_namespaced_aliases(self, source: str, alias: str) -> None:
        """Detect alias.ClassName.method_name patterns."""
        pattern = c.Infra.compile_double_attr_access(alias)
        for match in pattern.finditer(source):
            inner_name = match.group(1)
            method_name = match.group(2)
            base_class = self.inner_class_map.get(inner_name, "")
            if (
                base_class in self.method_index
                and method_name in self.method_index[base_class]
            ):
                self._record(base_class, method_name, c.Infra.CensusMode.ALIAS_NS)

    def _scan_direct_references(self, source: str) -> None:
        """Detect DirectClass.method_name patterns."""
        for local_name, actual in self.direct_imports.items():
            pattern = c.Infra.compile_attr_access(local_name)
            for match in pattern.finditer(source):
                method_name = match.group(1)
                if (
                    actual in self.method_index
                    and method_name in self.method_index[actual]
                ):
                    self._record(actual, method_name, c.Infra.CensusMode.DIRECT)

    def _record(
        self, class_name: str, method_name: str, mode: c.Infra.CensusMode
    ) -> None:
        """Record."""
        self.records.append(
            m.Infra.CensusUsageRecord(
                class_name=class_name,
                method_name=method_name,
                access_mode=mode,
                file_path=str(self.file_path),
                project=self.project_name,
            )
        )


__all__: list[str] = [
    "FlextInfraCensusImportDiscoveryVisitor",
    "FlextInfraCensusUsageCollector",
]
