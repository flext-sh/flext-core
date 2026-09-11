"""FLEXT family-chain discovery + project-kind inference — extracted concern."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from flext_infra import t


class FlextInfraProjectClassifierFamilyMixin:
    """Parse facade family files and confirm FLEXT family chains + project kind.

    Composed into FlextInfraProjectClassifier via inheritance; borrows
    ``_src_path`` and ``_normalize_dependency_name`` from the facade/deps-mixin
    via FLEXT.
    """

    _CLASS_DEF_RE: t.Infra.RegexPattern = c.Infra.CLASS_WITH_BASES_RE

    if TYPE_CHECKING:
        _src_path: Path

        def _normalize_dependency_name(self, raw_name: str) -> str: ...

    def _discover_facade_inheritance(
        self,
    ) -> t.Pair[t.MappingKV[str, t.Infra.StrSet], t.Infra.StrSet]:
        """Discover facade inheritance."""
        family_bases: t.MappingKV[str, t.Infra.StrSet] = {
            family: set() for family in c.Infra.FAMILY_SUFFIXES
        }
        local_facade_classes: t.Infra.StrSet = set()
        if not self._src_path.is_dir():
            return (family_bases, local_facade_classes)
        for family, suffix in c.Infra.FAMILY_SUFFIXES.items():
            file_pattern = c.Infra.FAMILY_FILES[family]
            for file_path in u.Infra.iter_matching_files(
                self._src_path, includes=[file_pattern]
            ):
                class_bases, class_names = self._parse_family_file(file_path, suffix)
                family_bases[family].update(class_bases)
                local_facade_classes.update(class_names)
        return (family_bases, local_facade_classes)

    def _parse_family_file(
        self, file_path: Path, suffix: str
    ) -> t.Pair[t.Infra.StrSet, t.Infra.StrSet]:
        """Parse family file."""
        source = u.Cli.files_read_text(file_path).unwrap()
        base_names: t.Infra.StrSet = set()
        class_names: t.Infra.StrSet = set()
        for match in self._CLASS_DEF_RE.finditer(source):
            name = match.group(1)
            if not name.endswith(suffix):
                continue
            class_names.add(name)
            bases_str = match.group(2)
            for raw_base_part in bases_str.split(","):
                base_part = raw_base_part.strip()
                if not base_part:
                    continue
                base_name = base_part.split("[")[0].rsplit(".", maxsplit=1)[-1].strip()
                if base_name:
                    base_names.add(base_name)
        return (base_names, class_names)

    def _build_confirmed_family_chains(
        self,
        *,
        internal_dependencies: t.StrSequence,
        family_bases: t.MappingKV[str, t.Infra.StrSet],
    ) -> t.MappingKV[str, t.StrSequence]:
        """Build confirmed family chains."""
        family_chains: MutableMapping[str, t.StrSequence] = {}
        for family, suffix in c.Infra.FAMILY_SUFFIXES.items():
            expected_parents = self._expected_parents_for_family(
                family_suffix=suffix, internal_dependencies=internal_dependencies
            )
            confirmed_bases = family_bases.get(family, set())
            confirmed_expected = [
                parent for parent in expected_parents if parent in confirmed_bases
            ]
            extra_confirmed = sorted(
                base for base in confirmed_bases if base not in set(confirmed_expected)
            )
            family_chains[family] = [*confirmed_expected, *extra_confirmed]
        return family_chains

    def _expected_parents_for_family(
        self, *, family_suffix: str, internal_dependencies: t.StrSequence
    ) -> t.StrSequence:
        """Return the expected parents for a family."""
        expected: t.MutableSequenceOf[str] = []
        for dependency in internal_dependencies:
            stem = self._dependency_to_class_stem(dependency)
            if not stem:
                continue
            candidate = f"{stem}{family_suffix}"
            if candidate not in expected:
                expected.append(candidate)
        return expected

    def _dependency_to_class_stem(self, dependency: str) -> str:
        """Dependency to class stem."""
        normalized = self._normalize_dependency_name(dependency)
        if normalized == c.Infra.PKG_CORE:
            return "Flext"
        if normalized.startswith(c.Infra.PKG_PREFIX_HYPHEN):
            tail = normalized.removeprefix(c.Infra.PKG_PREFIX_HYPHEN)
            parts = [part for part in tail.split("-") if part]
            if not parts:
                return ""
            return "Flext" + "".join(part.capitalize() for part in parts)
        parts = [part for part in normalized.split("-") if part]
        if not parts:
            return ""
        return "".join(part.capitalize() for part in parts)

    def _infer_project_kind(
        self,
        *,
        internal_dependencies: t.StrSequence,
        local_facade_classes: t.Infra.StrSet,
    ) -> str:
        """Infer project kind."""
        if not internal_dependencies:
            return "core"
        has_domain_dependency = any(
            dependency in c.Infra.DOMAIN_PACKAGES
            for dependency in internal_dependencies
        )
        has_platform_dependency = any(
            dependency in c.Infra.PLATFORM_PACKAGES
            for dependency in internal_dependencies
        )
        dependency_kind = "app"
        if has_domain_dependency and has_platform_dependency:
            dependency_kind = "integration"
        elif has_domain_dependency:
            dependency_kind = "domain"
        elif has_platform_dependency:
            dependency_kind = "platform"
        has_integration_facade = any(
            class_name.startswith(c.Infra.INTEGRATION_CLASS_PREFIXES)
            for class_name in local_facade_classes
        )
        if dependency_kind == "integration" and (not has_integration_facade):
            return "app"
        if (
            dependency_kind == "app"
            and has_domain_dependency
            and has_platform_dependency
            and has_integration_facade
        ):
            return "integration"
        return dependency_kind


__all__: list[str] = ["FlextInfraProjectClassifierFamilyMixin"]
