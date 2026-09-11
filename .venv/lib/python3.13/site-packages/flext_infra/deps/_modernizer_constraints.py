"""Dependency-constraint rewriting — extracted concern of FlextInfraPyprojectModernizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraPyprojectModernizerConstraintsMixin:
    """Rewrite PEP 621 / Poetry dependency constraints from the locked uv.lock state.

    Composed into FlextInfraPyprojectModernizer via inheritance; self-contained
    (operates only on the passed payload + locked-version inputs).
    """

    @staticmethod
    def _rewrite_requirement_group(
        raw_requirements: t.Infra.InfraValue,
        *,
        locked_versions: t.MappingKV[str, str],
        internal_names: t.StrSequence,
        location: str,
    ) -> tuple[t.JsonValueList | None, t.StrSequence]:
        """Rewrite one sequence of PEP 621 requirement strings in place."""
        if not isinstance(raw_requirements, list):
            return (None, [])
        updated_requirements: t.MutableSequenceOf[t.JsonValue] = []
        changes: t.MutableSequenceOf[str] = []
        for raw_requirement in raw_requirements:
            requirement = str(raw_requirement)
            rewritten = u.Infra.rewrite_requirement_constraint(
                requirement,
                locked_versions=locked_versions,
                internal_names=internal_names,
            )
            updated_requirements.append(rewritten or requirement)
            if rewritten is not None and rewritten != requirement:
                changes.append(f"{location}: {requirement} -> {rewritten}")
        if not changes:
            return (None, [])
        return (list(updated_requirements), tuple(changes))

    @staticmethod
    def _rewrite_poetry_dependency_table(
        dependencies: t.MutableJsonMapping,
        *,
        locked_versions: t.MappingKV[str, str],
        internal_names: t.StrSequence,
        location: str,
    ) -> t.StrSequence:
        """Rewrite one Poetry dependency table using the locked version policy."""
        changes: t.MutableSequenceOf[str] = []
        for dependency_name in list(dependencies):
            current_value = dependencies.get(dependency_name)
            rewritten_value = u.Infra.rewrite_poetry_constraint(
                dependency_name,
                current_value,
                locked_versions=locked_versions,
                internal_names=internal_names,
            )
            if rewritten_value is None:
                continue
            dependencies[dependency_name] = rewritten_value
            changes.append(
                f"{location}.{dependency_name}: {current_value!r} -> {rewritten_value!r}"
            )
        return tuple(changes)

    def _rewrite_dependency_constraints_payload(
        self,
        payload: t.MutableJsonMapping,
        *,
        locked_versions: t.MappingKV[str, str],
        internal_names: t.StrSequence,
    ) -> t.StrSequence:
        """Rewrite supported dependency tables from the current ``uv.lock`` state."""
        changes: t.MutableSequenceOf[str] = []
        project_view = u.Cli.toml_mapping_child(payload, c.Infra.PROJECT)
        if project_view is not None:
            project = u.Cli.toml_mapping_ensure_table(payload, c.Infra.PROJECT)
            project_deps, project_changes = self._rewrite_requirement_group(
                project.get(c.Infra.DEPENDENCIES),
                locked_versions=locked_versions,
                internal_names=internal_names,
                location="project.dependencies",
            )
            if project_deps is not None:
                project[c.Infra.DEPENDENCIES] = project_deps
                changes.extend(project_changes)
            optional_dependencies = u.Cli.toml_mapping_child(
                project, c.Infra.OPTIONAL_DEPENDENCIES
            )
            if optional_dependencies is not None:
                optional_dependencies = u.Cli.toml_mapping_ensure_table(
                    project, c.Infra.OPTIONAL_DEPENDENCIES
                )
                for group_name in list(optional_dependencies):
                    group_deps, group_changes = self._rewrite_requirement_group(
                        optional_dependencies.get(group_name),
                        locked_versions=locked_versions,
                        internal_names=internal_names,
                        location=f"project.optional-dependencies.{group_name}",
                    )
                    if group_deps is None:
                        continue
                    optional_dependencies[group_name] = group_deps
                    changes.extend(group_changes)
        dependency_groups_view = u.Cli.toml_mapping_child(
            payload, c.Infra.DEPENDENCY_GROUPS
        )
        if dependency_groups_view is not None:
            dependency_groups = u.Cli.toml_mapping_ensure_table(
                payload, c.Infra.DEPENDENCY_GROUPS
            )
            for group_name in list(dependency_groups):
                group_deps, group_changes = self._rewrite_requirement_group(
                    dependency_groups.get(group_name),
                    locked_versions=locked_versions,
                    internal_names=internal_names,
                    location=f"dependency-groups.{group_name}",
                )
                if group_deps is None:
                    continue
                dependency_groups[group_name] = group_deps
                changes.extend(group_changes)
        poetry_dependencies = u.Cli.toml_mapping_path(
            payload, (c.Infra.TOOL, c.Infra.POETRY, c.Infra.DEPENDENCIES)
        )
        if poetry_dependencies is not None:
            changes.extend(
                self._rewrite_poetry_dependency_table(
                    poetry_dependencies,
                    locked_versions=locked_versions,
                    internal_names=internal_names,
                    location="tool.poetry.dependencies",
                )
            )
        poetry_groups = u.Cli.toml_mapping_path(
            payload, (c.Infra.TOOL, c.Infra.POETRY, c.Infra.GROUP)
        )
        if poetry_groups is not None:
            for group_name in list(poetry_groups):
                group_dependencies = u.Cli.toml_mapping_path(
                    payload,
                    (
                        c.Infra.TOOL,
                        c.Infra.POETRY,
                        c.Infra.GROUP,
                        group_name,
                        c.Infra.DEPENDENCIES,
                    ),
                )
                if group_dependencies is None:
                    continue
                changes.extend(
                    self._rewrite_poetry_dependency_table(
                        group_dependencies,
                        locked_versions=locked_versions,
                        internal_names=internal_names,
                        location=(f"tool.poetry.group.{group_name}.dependencies"),
                    )
                )
        return tuple(changes)


__all__: list[str] = ["FlextInfraPyprojectModernizerConstraintsMixin"]
