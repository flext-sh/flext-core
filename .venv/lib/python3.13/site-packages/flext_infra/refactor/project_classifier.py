"""Project classification and family chain discovery for flext workspace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, override

from flext_infra import c, m, t, u
from flext_infra.refactor._project_classifier_deps import (
    FlextInfraProjectClassifierDepsMixin,
)
from flext_infra.refactor._project_classifier_family import (
    FlextInfraProjectClassifierFamilyMixin,
)

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraProjectClassifier(
    FlextInfraProjectClassifierDepsMixin, FlextInfraProjectClassifierFamilyMixin
):
    """Classify a project by kind and discover FLEXT family chains."""

    def __init__(
        self, project_root: Path, *, pyproject_payload: t.JsonMapping | None = None
    ) -> None:
        """Initialize classifier for the given project root."""
        self._project_root = project_root.resolve()
        self._pyproject_path = self._project_root / c.Infra.PYPROJECT_FILENAME
        self._pyproject_payload = pyproject_payload
        self._src_path = self._project_root / c.Infra.DEFAULT_SRC_DIR

    def classify(self) -> m.Infra.ProjectClassification:
        """Return classification and family chains for this project."""
        project_name, dependencies = self._read_project_metadata()
        internal_dependencies = self._internal_dependencies(
            dependencies=dependencies, project_name=project_name
        )
        family_bases, local_facade_classes = self._discover_facade_inheritance()
        family_chains = self._build_confirmed_family_chains(
            internal_dependencies=internal_dependencies, family_bases=family_bases
        )
        project_kind = self._infer_project_kind(
            internal_dependencies=internal_dependencies,
            local_facade_classes=local_facade_classes,
        )
        return m.Infra.ProjectClassification(
            project_kind=project_kind, family_chains={**family_chains}
        )

    def _read_project_metadata(self) -> t.Infra.TransformResult:
        """Read project metadata."""
        if self._pyproject_payload is not None:
            return self._project_metadata_from_payload(self._pyproject_payload)
        empty_dependencies: list[str] = []
        if not self._pyproject_path.is_file():
            return ("", empty_dependencies)
        data_result = u.Cli.toml_read_json(self._pyproject_path)
        if data_result.failure:
            return ("", empty_dependencies)
        return self._project_metadata_from_payload(
            t.Infra.INFRA_MAPPING_ADAPTER.validate_python(data_result.value)
        )

    def _project_metadata_from_payload(
        self, parsed: t.JsonMapping
    ) -> t.Infra.TransformResult:
        """Project metadata from payload."""
        raw_project = self._as_mapping(parsed.get(c.Infra.PROJECT))
        project_name = self._normalized_name_from_mapping(raw_project)
        dependencies: t.MutableSequenceOf[str] = []
        self._append_project_dependencies(
            raw_project=raw_project, dependencies=dependencies
        )
        raw_tool = self._as_mapping(parsed.get(c.Infra.TOOL))
        raw_poetry = self._as_mapping(raw_tool.get(c.Infra.POETRY))
        if not project_name:
            project_name = self._normalized_name_from_mapping(raw_poetry)
        self._append_poetry_dependencies(
            raw_poetry=raw_poetry, dependencies=dependencies
        )
        return (project_name, dependencies)

    @override
    def _as_mapping(
        self, raw_value: t.Infra.InfraValue | None
    ) -> t.MappingKV[str, t.Infra.InfraValue]:
        """As mapping."""
        if isinstance(raw_value, Mapping):
            validated: t.MappingKV[str, t.Infra.InfraValue] = (
                t.Infra.INFRA_MAPPING_ADAPTER.validate_python(raw_value)
            )
            return validated
        return {}

    def _normalized_name_from_mapping(
        self, raw_mapping: t.MappingKV[str, t.Infra.InfraValue]
    ) -> str:
        """Return the normalized name from a mapping."""
        raw_name = raw_mapping.get(c.Infra.NAME)
        if isinstance(raw_name, str):
            return self._normalize_dependency_name(raw_name)
        return ""


__all__: list[str] = ["FlextInfraProjectClassifier"]
