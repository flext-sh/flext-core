"""Project dependency-list parsing (PEP 621 + poetry) — extracted concern."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraProjectClassifierDepsMixin:
    """Collect + normalize a project's declared dependency names.

    Composed into FlextInfraProjectClassifier via inheritance; borrows
    ``_as_mapping`` from the facade via FLEXT.
    """

    if TYPE_CHECKING:

        def _as_mapping(
            self, raw_value: t.Infra.InfraValue | None
        ) -> t.MappingKV[str, t.Infra.InfraValue]: ...

    def _append_project_dependencies(
        self,
        *,
        raw_project: t.MappingKV[str, t.Infra.InfraValue],
        dependencies: t.MutableSequenceOf[str],
    ) -> None:
        """Append project dependencies."""
        raw_dependencies = raw_project.get(c.Infra.DEPENDENCIES)
        if not isinstance(raw_dependencies, list):
            return
        for raw_dependency in raw_dependencies:
            if not isinstance(raw_dependency, str):
                continue
            dependency_name = self._extract_dependency_name(raw_dependency)
            self._append_unique_dependency(
                dependency_name=dependency_name, dependencies=dependencies
            )

    def _append_poetry_dependencies(
        self,
        *,
        raw_poetry: t.MappingKV[str, t.Infra.InfraValue],
        dependencies: t.MutableSequenceOf[str],
    ) -> None:
        """Append poetry dependencies."""
        self._append_poetry_dependency_mapping(
            raw_mapping=self._as_mapping(raw_poetry.get(c.Infra.DEPENDENCIES)),
            dependencies=dependencies,
        )
        raw_group = self._as_mapping(raw_poetry.get(c.Infra.GROUP))
        raw_test_group = self._as_mapping(raw_group.get(c.Infra.TEST))
        self._append_poetry_dependency_mapping(
            raw_mapping=self._as_mapping(raw_test_group.get(c.Infra.DEPENDENCIES)),
            dependencies=dependencies,
        )

    def _append_poetry_dependency_mapping(
        self,
        *,
        raw_mapping: t.MappingKV[str, t.Infra.InfraValue],
        dependencies: t.MutableSequenceOf[str],
    ) -> None:
        """Append poetry dependency mapping."""
        dependency_keys = self._ordered_mapping_keys(raw_mapping)
        for dependency_key in dependency_keys:
            dependency_name = self._extract_dependency_name(dependency_key)
            if dependency_name == "python":
                continue
            self._append_unique_dependency(
                dependency_name=dependency_name, dependencies=dependencies
            )

    def _ordered_mapping_keys(
        self, raw_mapping: t.MappingKV[str, t.Infra.InfraValue]
    ) -> t.StrSequence:
        """Ordered mapping keys."""
        keys = list(raw_mapping.keys())
        if self._mapping_order_is_trusted(raw_mapping):
            return keys
        return sorted(keys)

    def _mapping_order_is_trusted(
        self, raw_mapping: t.MappingKV[str, t.Infra.InfraValue]
    ) -> bool:
        """Check whether the mapping order is trusted."""
        return isinstance(raw_mapping, dict)

    def _append_unique_dependency(
        self, *, dependency_name: str, dependencies: t.MutableSequenceOf[str]
    ) -> None:
        """Append unique dependency."""
        if (not dependency_name) or (dependency_name in dependencies):
            return
        dependencies.append(dependency_name)

    def _internal_dependencies(
        self, *, dependencies: t.StrSequence, project_name: str
    ) -> t.StrSequence:
        """Return the internal dependencies."""
        return [
            dependency
            for dependency in dependencies
            if dependency.startswith(c.Infra.PKG_PREFIX_HYPHEN)
            and dependency != project_name
        ]

    def _extract_dependency_name(self, raw_dependency: str) -> str:
        """Extract dependency name."""
        cleaned = raw_dependency.strip().split(";", maxsplit=1)[0].strip()
        if not cleaned:
            return ""
        left_side = cleaned.split(" @ ", maxsplit=1)[0].strip()
        base_token = left_side.split()[0]
        base_token = base_token.split("[", maxsplit=1)[0]
        base_token = c.Infra.DEPENDENCY_VERSION_OP_RE.split(base_token, maxsplit=1)[0]
        if "/" in base_token:
            path = Path(base_token)
            base_token = path.name
        return self._normalize_dependency_name(base_token)

    def _normalize_dependency_name(self, raw_name: str) -> str:
        """Normalize dependency name."""
        normalized: str = u.norm_str(raw_name, case="lower").replace("_", "-")
        return normalized.strip("./")


__all__: list[str] = ["FlextInfraProjectClassifierDepsMixin"]
