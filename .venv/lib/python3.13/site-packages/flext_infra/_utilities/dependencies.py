"""Dependency parsing and inspection helpers for flext-infra utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra._utilities.pyproject import FlextInfraUtilitiesPyproject
from flext_infra.constants import c

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra.typings import t


class FlextInfraUtilitiesDependencies:
    """Static helpers for inspecting dependency declarations in pyproject payloads."""

    @staticmethod
    def dep_name(requirement: str) -> str | None:
        """Extract normalized dependency name from one requirement spec."""
        text = requirement.strip()
        if not text:
            return None
        if ";" in text:
            text = text.split(";", maxsplit=1)[0].strip()
        if " @ " in text:
            text = text.split(" @ ", maxsplit=1)[0].strip()
        for separator in ("[", "==", ">=", "<=", "~=", "!=", ">", "<"):
            if separator in text:
                text = text.split(separator, maxsplit=1)[0].strip()
        if "/" in text:
            text = text.rsplit("/", maxsplit=1)[-1].strip()
        normalized = text.lower()
        return normalized or None

    @staticmethod
    def constraint_specifier(version: str) -> str:
        """Return the resolved lock version as an open-ended dependency floor.

        PEP 440 permits a local version label only with ``==`` or ``!=``, so a
        floor built straight from a locally tagged resolution is rejected by
        every build backend. The public release is what a floor means, and the
        local build satisfies it.
        """
        public_version = version.strip().partition("+")[0]
        return f">={public_version}" if public_version else ""

    @classmethod
    def locked_dependency_versions(
        cls, lock_path: Path, *, sources: t.StrSequence = ("registry",)
    ) -> t.MappingKV[str, str]:
        """Return normalized package versions from one ``uv.lock`` file.

        ``sources`` selects the lock source kinds to read (``registry`` by
        default; ``git`` yields the siblings consumed through a pinned ref).
        """
        result: t.MappingKV[str, str] = {}
        if lock_path.is_file():
            try:
                raw_text = lock_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            except OSError:
                pass
            else:
                payload_source = u.Cli.toml_mapping_from_text(raw_text)
                if payload_source is not None:
                    payload = FlextInfraUtilitiesPyproject.validate_infra_payload(
                        payload_source
                    )
                    if payload is not None:
                        raw_packages = payload.get("package")
                        if isinstance(raw_packages, list):
                            versions: dict[str, str] = {}
                            for raw_package in raw_packages:
                                if not isinstance(raw_package, Mapping):
                                    continue
                                raw_source = raw_package.get("source")
                                if not isinstance(raw_source, Mapping) or not any(
                                    kind in raw_source for kind in sources
                                ):
                                    continue
                                raw_name = raw_package.get("name")
                                raw_version = raw_package.get(c.Infra.VERSION)
                                if not isinstance(raw_name, str) or not isinstance(
                                    raw_version, str
                                ):
                                    continue
                                dependency_name = cls.dep_name(raw_name)
                                if dependency_name is None:
                                    continue
                                versions[dependency_name] = raw_version.strip()
                            result = dict(versions)
        return result

    @classmethod
    def rewrite_requirement_constraint(
        cls,
        requirement: str,
        *,
        locked_versions: t.MappingKV[str, str],
        internal_names: t.StrSequence = (),
    ) -> str | None:
        """Rewrite one PEP 621 requirement to the resolved uv.lock floor."""
        result: str | None = None
        raw_text = requirement.strip()
        if raw_text:
            requirement_part, marker_separator, marker_part = raw_text.partition(";")
            if " @ " not in requirement_part:
                head_match = c.Infra.PEP621_REQUIREMENT_HEAD_RE.match(
                    requirement_part.strip()
                )
                if head_match is not None:
                    head = head_match.group("head").strip()
                    dependency_name = cls.dep_name(head)
                    internal_set = set(internal_names)
                    if (
                        dependency_name is not None
                        and dependency_name not in internal_set
                    ):
                        locked_version = locked_versions.get(dependency_name)
                        if locked_version is not None:
                            rewritten = (
                                f"{head}{cls.constraint_specifier(locked_version)}"
                            )
                            marker_text = marker_part.strip()
                            if marker_separator and marker_text:
                                rewritten = f"{rewritten}; {marker_text}"
                            result = rewritten if rewritten != raw_text else None
        return result

    @classmethod
    def rewrite_poetry_constraint(
        cls,
        dependency_name: str,
        raw_value: t.Infra.InfraValue,
        *,
        locked_versions: t.MappingKV[str, str],
        internal_names: t.StrSequence = (),
    ) -> t.Infra.InfraValue | None:
        """Rewrite one Poetry dependency to the resolved uv.lock floor."""
        result: t.Infra.InfraValue | None = None
        normalized_name = cls.dep_name(dependency_name)
        internal_set = set(internal_names)
        if (
            normalized_name is not None
            and normalized_name != "python"
            and normalized_name not in internal_set
        ):
            locked_version = locked_versions.get(normalized_name)
            if locked_version is not None:
                rewritten_specifier = cls.constraint_specifier(locked_version)
                if isinstance(raw_value, str):
                    result = (
                        rewritten_specifier
                        if raw_value != rewritten_specifier
                        else None
                    )
                elif isinstance(raw_value, Mapping) and not any(
                    key in raw_value for key in (c.Infra.PATH, "git", "url")
                ):
                    updated: t.MutableJsonMapping = dict(raw_value)
                    if updated.get(c.Infra.VERSION) != rewritten_specifier:
                        updated[c.Infra.VERSION] = rewritten_specifier
                        result = dict(updated)
        return result

    @staticmethod
    def dedupe_specs(specs: t.StrSequence) -> t.StrSequence:
        """Return deterministic unique dependency specs keyed by normalized name."""
        selected_by_name: dict[str, str] = {}
        for raw in specs:
            item = raw.strip()
            if not item:
                continue
            dependency_name = FlextInfraUtilitiesDependencies.dep_name(item)
            if dependency_name is None or dependency_name in selected_by_name:
                continue
            selected_by_name[dependency_name] = item
        return tuple(selected_by_name[name] for name in sorted(selected_by_name))

    @classmethod
    def declared_dependency_names(cls, document: t.Cli.TomlDocument) -> t.StrSequence:
        """Return normalized dependency names from one TOML document."""
        normalized = FlextInfraUtilitiesPyproject.normalized_toml_payload(document)
        if not normalized:
            return ()
        return cls.declared_dependency_names_from_payload(normalized)

    @classmethod
    def declared_dependency_names_from_payload(
        cls, payload: t.JsonMapping
    ) -> t.StrSequence:
        """Return normalized dependency names across supported dependency tables."""
        names: set[str] = set()
        cls._append_project_dependency_names(payload=payload, names=names)
        cls._append_dependency_group_names(payload=payload, names=names)
        cls._append_poetry_dependency_names(payload=payload, names=names)
        return tuple(sorted(names))

    @classmethod
    def _append_project_dependency_names(
        cls, *, payload: t.JsonMapping, names: set[str]
    ) -> None:
        """Append project dependency names."""
        project = payload.get(c.Infra.PROJECT)
        if not isinstance(project, Mapping):
            return
        cls._append_requirement_names(
            raw_requirements=project.get(c.Infra.DEPENDENCIES), names=names
        )
        optional_dependencies = project.get(c.Infra.OPTIONAL_DEPENDENCIES)
        if not isinstance(optional_dependencies, Mapping):
            return
        for raw_requirements in optional_dependencies.values():
            cls._append_requirement_names(
                raw_requirements=raw_requirements, names=names
            )

    @classmethod
    def _append_dependency_group_names(
        cls, *, payload: t.JsonMapping, names: set[str]
    ) -> None:
        """Append dependency group names."""
        dependency_groups = payload.get(c.Infra.DEPENDENCY_GROUPS)
        if not isinstance(dependency_groups, Mapping):
            return
        for raw_requirements in dependency_groups.values():
            cls._append_requirement_names(
                raw_requirements=raw_requirements, names=names
            )

    @classmethod
    def _append_poetry_dependency_names(
        cls, *, payload: t.JsonMapping, names: set[str]
    ) -> None:
        """Append poetry dependency names."""
        tool = payload.get(c.Infra.TOOL)
        if not isinstance(tool, Mapping):
            return
        poetry = tool.get(c.Infra.POETRY)
        if not isinstance(poetry, Mapping):
            return
        cls._append_mapping_dependency_names(
            raw_mapping=poetry.get(c.Infra.DEPENDENCIES), names=names
        )
        poetry_groups = poetry.get(c.Infra.GROUP)
        if not isinstance(poetry_groups, Mapping):
            return
        for raw_group in poetry_groups.values():
            if not isinstance(raw_group, Mapping):
                continue
            cls._append_mapping_dependency_names(
                raw_mapping=raw_group.get(c.Infra.DEPENDENCIES), names=names
            )

    @classmethod
    def _append_requirement_names(
        cls, *, raw_requirements: t.Infra.InfraValue, names: set[str]
    ) -> None:
        """Append requirement names."""
        if not isinstance(raw_requirements, list):
            return
        for raw_requirement in raw_requirements:
            dependency_name = cls.dep_name(str(raw_requirement))
            if dependency_name is None:
                continue
            names.add(dependency_name)

    @classmethod
    def _append_mapping_dependency_names(
        cls, *, raw_mapping: t.Infra.InfraValue, names: set[str]
    ) -> None:
        """Append mapping dependency names."""
        if not isinstance(raw_mapping, Mapping):
            return
        for raw_name in raw_mapping:
            dependency_name = cls.dep_name(raw_name)
            if dependency_name is None or dependency_name == "python":
                continue
            names.add(dependency_name)

    @classmethod
    def local_dependency_names_from_payload(
        cls, payload: t.JsonMapping, *, workspace_project_names: t.StrSequence = ()
    ) -> t.StrSequence:
        """Return workspace-local dependency names from one payload."""
        declared = set(cls.declared_dependency_names_from_payload(payload))
        if not workspace_project_names:
            return ()
        workspace_names = set(workspace_project_names)
        return tuple(sorted(name for name in declared if name in workspace_names))

    @staticmethod
    def project_dev_groups_from_payload(
        payload: t.JsonMapping,
    ) -> t.MappingKV[str, t.StrSequence]:
        """Collect optional dependency groups from one normalized payload."""
        project = u.Cli.json_as_mapping(payload.get(c.Infra.PROJECT, None))
        optional = u.Cli.json_as_mapping(
            project.get(c.Infra.OPTIONAL_DEPENDENCIES, None)
        )
        groups = {
            str(group): tuple(
                str(item) for item in u.Cli.json_as_sequence(optional.get(group, None))
            )
            for group in c.Infra.CANONICAL_DEV_DEPENDENCY_GROUPS
        }
        return {group: values for group, values in groups.items() if values}

    @classmethod
    def project_dev_groups(
        cls, document: t.Cli.TomlDocument
    ) -> t.MappingKV[str, t.StrSequence]:
        """Collect optional dependency groups from one TOML document."""
        normalized = FlextInfraUtilitiesPyproject.normalized_toml_payload(document)
        if not normalized:
            # flext-j47u (codex): keep the empty mapping immutable and fully typed.
            return MappingProxyType(dict[str, tuple[str, ...]]())
        return cls.project_dev_groups_from_payload(normalized)

    @classmethod
    def canonical_dev_dependencies(cls, document: t.Cli.TomlDocument) -> t.StrSequence:
        """Merge all canonical dev dependency groups from one TOML document."""
        normalized = FlextInfraUtilitiesPyproject.normalized_toml_payload(document)
        if not normalized:
            return ()
        return cls.canonical_dev_dependencies_from_payload(normalized)

    @classmethod
    def canonical_dev_dependencies_from_payload(
        cls, payload: t.JsonMapping
    ) -> t.StrSequence:
        """Merge all canonical dev dependency groups from one normalized payload."""
        groups = cls.project_dev_groups_from_payload(payload)
        return cls.dedupe_specs([
            requirement
            for group in c.Infra.CANONICAL_DEV_DEPENDENCY_GROUPS
            for requirement in groups.get(str(group), ())
        ])

    @classmethod
    def flext_dependency_namespaces(cls, document: t.Cli.TomlDocument) -> t.StrSequence:
        """Extract declared FLEXT dependency namespaces from one TOML document."""
        normalized = FlextInfraUtilitiesPyproject.normalized_toml_payload(document)
        if not normalized:
            return ()
        return cls.flext_dependency_namespaces_from_payload(normalized)

    @classmethod
    def flext_dependency_namespaces_from_payload(
        cls, payload: t.MappingKV[str, t.Infra.InfraValue]
    ) -> t.StrSequence:
        """Extract every declared ``flext-*`` dependency as a Python namespace."""
        # flext-j47u (codex): FLEXT dependencies are first-party contracts even
        # when their uv source declaration is owned by an enclosing workspace.
        normalized = FlextInfraUtilitiesPyproject.validate_infra_payload(payload)
        if normalized is None:
            return ()
        return tuple(
            sorted(
                name.replace("-", "_")
                for name in cls.declared_dependency_names_from_payload(normalized)
                if name == "flext" or name.startswith(c.Infra.PKG_PREFIX_HYPHEN)
            )
        )


__all__: list[str] = ["FlextInfraUtilitiesDependencies"]
