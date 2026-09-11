"""Dependency typings analysis + container-value conversion helpers for detection."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import override

from flext_core import r
from flext_infra import c, m, p, t
from flext_infra.deps._detection_runners import (
    FlextInfraDependencyDetectionRunnersMixin,
)


class FlextInfraDependencyDetectionAnalysis(FlextInfraDependencyDetectionRunnersMixin):
    """Typings analysis + conversion helpers composed with the tool-runner mixin."""

    @override
    def _to_toml_config(
        self, payload: t.MappingKV[str, t.Infra.InfraValue]
    ) -> t.JsonMapping:
        """To toml config."""
        normalized: MutableMapping[str, t.Infra.InfraValue] = {}
        for key, value in payload.items():
            if value is None:
                normalized[key] = None
                continue
            converted = FlextInfraDependencyDetectionAnalysis.to_infra_value(value)
            if converted is None:
                continue
            normalized[key] = converted
        return normalized

    @staticmethod
    def to_infra_value(value: t.Infra.InfraValue | None) -> t.Infra.InfraValue | None:
        """Convert container value to namespaced infra value."""
        if value is None:
            return None
        if isinstance(value, t.PRIMITIVES_TYPES):
            primitive: t.Infra.InfraValue = value
            return primitive
        scalar_types = t.PRIMITIVES_TYPES
        if isinstance(value, list):
            try:
                sequence = t.Cli.JSON_LIST_ADAPTER.validate_python(value)
            except c.ValidationError:
                return None
            converted: t.MutableSequenceOf[t.Infra.InfraValue] = []
            for item in sequence:
                if item is None:
                    converted.append(None)
                    continue
                conv = FlextInfraDependencyDetectionAnalysis.to_infra_value(item)
                if conv is None or not isinstance(conv, scalar_types):
                    return None
                converted.append(conv)
            return list(t.Cli.JSON_LIST_ADAPTER.validate_python(converted))
        try:
            mapping_value = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(value)
        except c.ValidationError:
            return None
        converted_map: MutableMapping[str, t.Infra.InfraValue] = {}
        for key, map_item in mapping_value.items():
            if map_item is None:
                converted_map[key] = None
                continue
            conv = FlextInfraDependencyDetectionAnalysis.to_infra_value(map_item)
            if conv is None or not isinstance(conv, scalar_types):
                return None
            converted_map[key] = conv
        mapping: t.Infra.InfraValue = t.json_dict_adapter().validate_python(
            converted_map
        )
        return mapping

    def _mapping_from_value(self, value: t.Infra.InfraValue | None) -> t.JsonMapping:
        """Build a mapping from a value."""
        if not isinstance(value, Mapping):
            return {}
        return self._to_toml_config(value)

    def get_current_typings_from_pyproject(self, project_path: Path) -> t.StrSequence:
        """Extract currently declared typing packages from project pyproject.toml."""
        pyproject = project_path / c.Infra.PYPROJECT_FILENAME
        read_result = self._read_plain(pyproject)
        if read_result.failure:
            return []
        data = self._to_toml_config(read_result.value)
        if not data:
            return []
        names: t.Infra.StrSet = set()
        tool = self._mapping_from_value(data.get(c.Infra.TOOL))
        poetry = self._mapping_from_value(tool.get(c.Infra.POETRY))
        group = self._mapping_from_value(poetry.get(c.Infra.GROUP))
        typings_group = self._mapping_from_value(group.get(c.Infra.TYPINGS))
        deps = self._mapping_from_value(typings_group.get(c.Infra.DEPENDENCIES))
        names.update(key for key in deps)
        project = self._mapping_from_value(data.get(c.Infra.PROJECT))
        optional = self._mapping_from_value(project.get(c.Infra.OPTIONAL_DEPENDENCIES))
        typings = optional.get(c.Infra.TYPINGS)
        if isinstance(typings, list):
            for spec in typings:
                spec_text = str(spec)
                names.add(
                    spec_text
                    .split("[", maxsplit=1)[0]
                    .split(">=", maxsplit=1)[0]
                    .split("==", maxsplit=1)[0]
                    .strip()
                )
        elif isinstance(typings, Mapping):
            names.update(key for key in typings)
        return sorted(names)

    def get_required_typings(
        self,
        project_path: Path,
        limits_path: Path | None = None,
        *,
        include_mypy: bool = True,
    ) -> p.Result[m.Infra.TypingsReport]:
        """Analyze project and generate typing stubs requirements report."""
        limits = self.load_dependency_limits(limits_path)
        exclude_set: t.Infra.StrSet = set()
        typing_libraries = limits.get(c.Infra.TYPING_LIBRARIES)
        if isinstance(typing_libraries, Mapping):
            excluded = typing_libraries.get(c.Infra.EXCLUDE)
            if isinstance(excluded, list):
                exclude_set = {str(e) for e in excluded}
        hinted: t.StrSequence = []
        missing_modules: t.StrSequence = []
        if include_mypy:
            hints_result = self.run_mypy_stub_hints(project_path)
            if hints_result.failure:
                return r[m.Infra.TypingsReport].fail(
                    hints_result.error or "typing hint detection failed"
                )
            typed_hints: t.Pair[t.StrSequence, t.StrSequence] = hints_result.value
            hinted, missing_modules = typed_hints
        required_set: t.Infra.StrSet = set(hinted)
        for module_name in missing_modules:
            package = self.module_to_types_package(module_name, limits)
            if package:
                required_set.add(package)
        required_set -= exclude_set
        current = self.get_current_typings_from_pyproject(project_path)
        current_set = set(current)
        python_cfg = limits.get(c.Infra.PYTHON)
        version_val = (
            python_cfg.get(c.Infra.VERSION) if isinstance(python_cfg, Mapping) else None
        )
        python_version = str(version_val) if version_val is not None else None
        report = m.Infra.TypingsReport(
            required_packages=sorted(required_set),
            hinted=hinted,
            missing_modules=missing_modules,
            current=current,
            to_add=sorted(required_set - current_set),
            to_remove=sorted(current_set - required_set),
            limits_applied=bool(limits),
            python_version=python_version,
        )
        return r[m.Infra.TypingsReport].ok(report)

    def load_dependency_limits(
        self, limits_path: Path | None = None
    ) -> t.MappingKV[str, t.Infra.InfraValue]:
        """Load dependency limits configuration from TOML file."""
        path = (
            limits_path
            or Path(__file__).resolve().parent / c.Infra.DEPENDENCY_LIMITS_FILENAME
        )
        result = self._read_plain(path)
        if result.failure:
            return {}
        return result.value

    def module_to_types_package(
        self, module_name: str, limits: t.MappingKV[str, t.Infra.InfraValue]
    ) -> str | None:
        """Map a module name to its corresponding types-* package."""
        root = module_name.split(".", 1)[0]
        if root.startswith(c.Infra.INTERNAL_PREFIXES):
            return None
        typing_libraries = limits.get(c.Infra.TYPING_LIBRARIES)
        if isinstance(typing_libraries, Mapping):
            module_to_package = typing_libraries.get(c.Infra.MODULE_TO_PACKAGE)
            if isinstance(module_to_package, Mapping) and root in module_to_package:
                value = module_to_package.get(root)
                return str(value) if value is not None else None
        # 5a7d1d710 removed the hardcoded module->types-* table: the declared
        # typing_libraries.module_to_package config is the only source, so an
        # undeclared module is unknown rather than a guessed types-* name.
        return None


__all__: list[str] = ["FlextInfraDependencyDetectionAnalysis"]
