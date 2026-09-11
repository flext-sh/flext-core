"""Autonomous library pyproject conformance through the flext-cli TOML facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from flext_cli import r, u
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t
from flext_infra._utilities.dependencies import FlextInfraUtilitiesDependencies
from flext_infra._utilities.repository import FlextInfraUtilitiesRepository

if TYPE_CHECKING:
    from flext_infra.protocols import p


class FlextInfraUtilitiesPyprojectConform:
    """Render root workspace and autonomous library metadata deterministically."""

    # This pure renderer replaces the mutating dependency path-sync command;
    # codegen is the only public orchestrator.

    @classmethod
    def pyproject_conform(
        cls,
        pyproject_content: str,
        *,
        providers: t.SequenceOf[m.Infra.ProviderSpec],
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
        toolchain: p.Infra.ToolchainSpec,
        required_dev_dependencies: t.StrSequence,
        uv_link_mode: str | None = None,
        uv_exclude_newer: str | None = None,
        dependency_cooldown_exclusions: t.StrSequence | None = None,
        dependency_cooldown_overrides: t.StrMapping | None = None,
        uv_exclude_dependencies: t.SequenceOf[p.Model] = (),
    ) -> p.Result[str]:
        """Return canonical TOML with autonomous dependencies and root workspace.

        ``uv_exclude_newer`` is the per-project overlay over the fleet cooldown.
        The fleet default is a ROLLING window, which silently ages past a
        security floor declared in override-dependencies and makes resolution
        unsatisfiable; a project carrying such a floor pins the absolute cutoff
        instead. ``None`` keeps the fleet window.
        """
        source = u.Cli.toml_parse_text(pyproject_content)
        if source is None:
            return r[str].fail("pyproject content is not valid TOML")
        project = u.Cli.toml_table_child(source, c.Infra.PROJECT)
        if project is None:
            return r[str].fail("pyproject content must define [project]")
        project_name_raw = u.Cli.toml_value(project, c.Infra.NAME)
        if not isinstance(project_name_raw, str) or not project_name_raw.strip():
            return r[str].fail("[project].name must be a non-empty string")
        project_name = project_name_raw.strip()
        cls._sync_dependency_groups(
            source,
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
            required_dev_dependencies=required_dev_dependencies,
        )
        normalized = cls._normalize_requirements(
            source,
            project_name=project_name,
            providers=providers,
            workspace=workspace,
            workspace_mode=workspace_mode,
            canonicalize_all=True,
        )
        if normalized.failure:
            return r[str].fail(normalized.error or "dependency normalization failed")
        cls._remove_legacy_tooling(source)
        typecheck_paths = cls._sync_typecheck_paths(source)
        if typecheck_paths.failure:
            return r[str].fail(
                typecheck_paths.error or "type checker path conformance failed"
            )
        sources_result = cls._sync_uv_sources(
            source,
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
            link_mode=uv_link_mode or toolchain.uv_link_mode,
            exclude_newer=uv_exclude_newer or toolchain.uv_exclude_newer,
            exclude_newer_packages=(
                toolchain.dependency_cooldown_exclusions
                if dependency_cooldown_exclusions is None
                else dependency_cooldown_exclusions
            ),
            exclude_newer_overrides=(
                toolchain.dependency_cooldown_overrides
                if dependency_cooldown_overrides is None
                else dependency_cooldown_overrides
            ),
            exclude_dependencies=uv_exclude_dependencies,
            constraint_dependencies=toolchain.dependency_constraints,
            uv_environments=toolchain.uv_environments,
        )
        if sources_result.failure:
            return r[str].fail(sources_result.error or "uv source conformance failed")
        provenance_result = cls._validate_dependency_provenance(
            source,
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )
        if provenance_result.failure:
            return r[str].fail(
                provenance_result.error or "dependency provenance validation failed"
            )

        rendered = u.Cli.toml_dumps(source)
        if u.Cli.toml_parse_text(rendered) is None:
            return r[str].fail("canonical pyproject rendering produced invalid TOML")
        return r[str].ok(rendered)

    @classmethod
    def pyproject_dependencies_conform(
        cls,
        pyproject_content: str,
        *,
        providers: t.SequenceOf[m.Infra.ProviderSpec],
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
    ) -> p.Result[str]:
        """Conform only internal requirements and their root workspace overlay."""
        source = u.Cli.toml_parse_text(pyproject_content)
        if source is None:
            return r[str].fail("pyproject content is not valid TOML")
        project = u.Cli.toml_table_child(source, c.Infra.PROJECT)
        if project is None:
            return r[str].fail("pyproject content must define [project]")
        project_name_raw = u.Cli.toml_value(project, c.Infra.NAME)
        if not isinstance(project_name_raw, str) or not project_name_raw.strip():
            return r[str].fail("[project].name must be a non-empty string")
        project_name = project_name_raw.strip()
        workspace_context_root = cls._is_workspace_context_root(
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )
        if workspace_context_root:
            sources_result = cls._validate_root_uv_sources(
                source, workspace=workspace, providers=providers
            )
            if sources_result.failure:
                return r[str].fail(
                    sources_result.error or "uv source conformance failed"
                )
        normalized = cls._normalize_requirements(
            source,
            project_name=project_name,
            providers=providers,
            workspace=workspace,
            workspace_mode=workspace_mode,
            canonicalize_all=False,
        )
        if normalized.failure:
            return r[str].fail(normalized.error or "dependency normalization failed")
        cls._sync_workspace_dependency_group(
            source,
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )
        sources_result = (
            r[bool].ok(True)
            if workspace_context_root
            else cls._sync_uv_sources(
                source,
                project_name=project_name,
                workspace=workspace,
                workspace_mode=workspace_mode,
            )
        )
        if sources_result.failure:
            return r[str].fail(sources_result.error or "uv source conformance failed")
        provenance_result = cls._validate_dependency_provenance(
            source,
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )
        if provenance_result.failure:
            return r[str].fail(
                provenance_result.error or "dependency provenance validation failed"
            )
        rendered = u.Cli.toml_dumps(source)
        if u.Cli.toml_parse_text(rendered) is None:
            return r[str].fail("dependency conformance produced invalid TOML")
        return r[str].ok(rendered)

    @classmethod
    def _normalize_requirements(
        cls,
        document: t.Cli.TomlDocument,
        *,
        project_name: str,
        providers: t.SequenceOf[m.Infra.ProviderSpec],
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
        canonicalize_all: bool,
    ) -> p.Result[bool]:
        """Render internal requirements for root workspace or detached operation."""
        available = (workspace.repository, *workspace.declared_repositories)
        # Only the root expresses the active workspace overlay in its own
        # requirements. A publishable project keeps its configured Git source
        # so the same pyproject remains resolvable in a standalone checkout; uv
        # replaces it with workspace=true from the repository root.
        workspace_dependencies = (
            frozenset(
                project.distribution for project in workspace.declared_repositories
            )
            if cls._is_workspace_context_root(
                project_name=project_name,
                workspace=workspace,
                workspace_mode=workspace_mode,
            )
            else frozenset()
        )
        project = u.Cli.toml_ensure_table(document, c.Infra.PROJECT)
        normalized = cls._normalize_requirement_field(
            project,
            c.Infra.DEPENDENCIES,
            repositories=available,
            providers=providers,
            canonicalize_all=canonicalize_all,
            workspace_dependencies=workspace_dependencies,
        )
        if normalized.failure:
            return normalized
        for section_name in (c.Infra.OPTIONAL_DEPENDENCIES, c.Infra.DEPENDENCY_GROUPS):
            parent = (
                project if section_name == c.Infra.OPTIONAL_DEPENDENCIES else document
            )
            section = u.Cli.toml_table_child(parent, section_name)
            if section is None:
                continue
            for group_name in tuple(section):
                group_result = cls._normalize_requirement_field(
                    section,
                    group_name,
                    repositories=available,
                    providers=providers,
                    canonicalize_all=canonicalize_all,
                    workspace_dependencies=workspace_dependencies,
                )
                if group_result.failure:
                    return group_result
        return r[bool].ok(True)

    @classmethod
    def _normalize_requirement_field(
        cls,
        container: t.Cli.TomlDocument | t.Cli.TomlTable,
        key: str,
        *,
        repositories: t.SequenceOf[p.Infra.RepositoryRef],
        providers: t.SequenceOf[m.Infra.ProviderSpec],
        canonicalize_all: bool,
        workspace_dependencies: frozenset[str],
    ) -> p.Result[bool]:
        """Normalize one dependency array and fail on model-less entries."""
        raw_value = u.Cli.toml_value(container, key)
        if raw_value is None:
            return r[bool].ok(True)
        raw_items = u.Cli.json_as_sequence(raw_value)
        try:
            items = t.Infra.STR_SEQ_ADAPTER.validate_python(raw_items, strict=True)
        except c.ValidationError as exc:
            return r[bool].fail_op(f"validate dependency group {key}", exc)
        normalized_items: t.MutableSequenceOf[str] = []
        for item in items:
            normalized = cls._canonical_requirement(
                item,
                repositories=repositories,
                providers=providers,
                workspace_dependencies=workspace_dependencies,
            )
            if normalized.failure:
                return r[bool].fail(
                    normalized.error or f"normalize dependency group {key} failed"
                )
            normalized_items.append(normalized.value)
        canonical = tuple(dict.fromkeys(normalized_items))
        if canonicalize_all:

            def requirement_key(requirement: str) -> tuple[str, str]:
                name = FlextInfraUtilitiesDependencies.dep_name(requirement) or ""
                return name, requirement

            canonical = tuple(sorted(canonical, key=requirement_key))
        u.Cli.toml_sync_string_list(container, key, canonical)
        return r[bool].ok(True)

    @classmethod
    def _canonical_requirement(
        cls,
        requirement: str,
        *,
        repositories: t.SequenceOf[p.Infra.RepositoryRef],
        providers: t.SequenceOf[m.Infra.ProviderSpec],
        workspace_dependencies: frozenset[str],
    ) -> p.Result[str]:
        """Render one internal requirement from its local topology reference."""
        dependency_name = FlextInfraUtilitiesDependencies.dep_name(requirement)
        if dependency_name is None or not dependency_name.startswith("flext-"):
            return r[str].ok(requirement.strip())
        requirement_part, separator, marker = requirement.partition(";")
        head_match = c.Infra.PEP621_REQUIREMENT_HEAD_RE.match(requirement_part.strip())
        if head_match is None:
            return r[str].fail(f"invalid internal requirement: {requirement}")
        head = head_match.group("head").strip()
        marker_text = marker.strip()
        if dependency_name in workspace_dependencies:
            if "@" in requirement_part:
                return r[str].fail(
                    "workspace dependency declares a conflicting direct source: "
                    f"{dependency_name}"
                )
            return r[str].ok(
                f"{head}; {marker_text}" if separator and marker_text else head
            )
        reference_result = cls._repository_reference(
            dependency_name, repositories=repositories, providers=providers
        )
        if reference_result.failure:
            return r[str].fail(
                reference_result.error
                or f"repository resolution failed: {dependency_name}"
            )
        reference = reference_result.value
        provider = FlextInfraUtilitiesRepository.repository_provider(
            reference, providers
        )
        if provider.failure:
            return r[str].fail(
                provider.error or "repository provider resolution failed"
            )
        git_url = cls._git_requirement_url(reference.url)
        if git_url.failure:
            return r[str].fail(git_url.error or "repository URL validation failed")
        canonical = f"{head} @ {git_url.value}@{provider.value.branch}"
        return r[str].ok(
            f"{canonical}; {marker_text}" if separator and marker_text else canonical
        )

    @staticmethod
    def _repository_reference(
        distribution: str,
        *,
        repositories: t.SequenceOf[p.Infra.RepositoryRef],
        providers: t.SequenceOf[m.Infra.ProviderSpec],
    ) -> p.Result[p.Infra.RepositoryRef]:
        """Return one unambiguous reference for a distribution.

        A distribution the workspace does not declare is still resolvable: its
        canonical source is the provider contract plus its own name. That is
        derived from generic policy, never from a catalog of projects that
        flext-infra is forbidden to own.
        """
        matches = tuple(
            repository
            for repository in repositories
            if repository.distribution == distribution
        )
        if not matches:
            return r.ok(
                FlextInfraUtilitiesRepository.derived_repository_ref(
                    distribution, provider=providers[0]
                )
            )
        reference = matches[0]
        if any(
            item.url != reference.url or item.provider != reference.provider
            for item in matches[1:]
        ):
            return r.fail(
                f"repository catalog conflicts for distribution: {distribution}"
            )
        return r.ok(reference)

    @staticmethod
    def _git_requirement_url(url: str) -> p.Result[str]:
        """Render the configured HTTPS clone URL as a PEP 508 Git URL."""
        if not url.startswith("https://"):
            return r[str].fail(
                f"repository URL must use the configured HTTPS transport: {url}"
            )
        return r[str].ok(f"git+{url}")

    @classmethod
    def _sync_dependency_groups(
        cls,
        document: t.Cli.TomlDocument,
        *,
        project_name: str,
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
        required_dev_dependencies: t.StrSequence,
    ) -> None:
        """Migrate optional dev dependencies and normalize declared groups."""
        project = u.Cli.toml_ensure_table(document, c.Infra.PROJECT)
        groups = u.Cli.toml_ensure_table(document, c.Infra.DEPENDENCY_GROUPS)
        optional = u.Cli.toml_table_child(project, c.Infra.OPTIONAL_DEPENDENCIES)
        optional_dev: t.StrSequence = ()
        if optional is not None:
            optional_dev = u.Cli.toml_as_string_list(
                u.Cli.toml_value(optional, str(c.Infra.DEV))
            )
        # SSOT required floors win over existing same-name pins: dedupe_specs
        # keeps the first occurrence, so toolchain floors must lead the merge.
        # Otherwise stale member pins (e.g. rumdl>=0.2.46) block exclude-newer.
        required_dev = tuple(
            requirement
            for requirement in required_dev_dependencies
            if FlextInfraUtilitiesDependencies.dep_name(requirement) != project_name
        )
        dev = [
            *required_dev,
            *u.Cli.toml_as_string_list(u.Cli.toml_value(groups, str(c.Infra.DEV))),
            *optional_dev,
        ]
        if dev:
            u.Cli.toml_sync_string_list(
                groups,
                str(c.Infra.DEV),
                FlextInfraUtilitiesDependencies.dedupe_specs(tuple(dev)),
            )
        else:
            u.Cli.toml_remove_key_if_present(groups, str(c.Infra.DEV))

        codegen = u.Cli.toml_as_string_list(u.Cli.toml_value(groups, "codegen"))
        if codegen:
            u.Cli.toml_sync_string_list(
                groups,
                "codegen",
                FlextInfraUtilitiesDependencies.dedupe_specs(tuple(codegen)),
            )
        else:
            u.Cli.toml_remove_key_if_present(groups, "codegen")
        cls._sync_workspace_dependency_group(
            document,
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )

        if optional is not None:
            u.Cli.toml_remove_key_if_present(optional, str(c.Infra.DEV))
            if not tuple(optional):
                u.Cli.toml_remove_key_if_present(project, c.Infra.OPTIONAL_DEPENDENCIES)

    @classmethod
    def _sync_workspace_dependency_group(
        cls,
        document: t.Cli.TomlDocument,
        *,
        project_name: str,
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
    ) -> None:
        """Keep the generated workspace dependency group only at the root."""
        repository_root = cls._is_workspace_context_root(
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )
        groups = u.Cli.toml_table_child(document, c.Infra.DEPENDENCY_GROUPS)
        if groups is None:
            if not repository_root:
                return
            # The root dependency overlay is complete even when an older
            # pyproject has no groups table yet.
            groups = u.Cli.toml_ensure_table(document, c.Infra.DEPENDENCY_GROUPS)
        if repository_root:
            u.Cli.toml_sync_string_list(
                groups,
                "workspace",
                tuple(
                    sorted(
                        project.distribution
                        for project in workspace.declared_repositories
                    )
                ),
            )
            return
        u.Cli.toml_remove_key_if_present(groups, "workspace")

    @staticmethod
    def _is_repository_root(
        *, project_name: str, workspace: p.Infra.WorkspaceSpec
    ) -> bool:
        """Identify the real multi-project root, not an autonomous repository."""
        return bool(workspace.declared_repositories) and (
            project_name == workspace.repository.distribution
        )

    @classmethod
    def _is_workspace_context_root(
        cls,
        *,
        project_name: str,
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
    ) -> bool:
        """Identify the root only when the active topology is a workspace."""
        return (
            workspace_mode is c.Infra.MakeProfile.WORKSPACE
            and cls._is_repository_root(project_name=project_name, workspace=workspace)
        )

    @staticmethod
    def _remove_legacy_tooling(document: t.Cli.TomlDocument) -> None:
        """Delete legacy packaging owners superseded by canonical conformance.

        The ``[tool.flext]`` table is preserved because it carries project-local
        tooling policy unrelated to repository topology.
        """
        tool = u.Cli.toml_table_child(document, c.Infra.TOOL)
        if tool is None:
            return
        u.Cli.toml_remove_key_if_present(tool, c.Infra.POETRY)

    @staticmethod
    def _sync_typecheck_paths(document: t.Cli.TomlDocument) -> p.Result[bool]:
        """Remove checkout-absolute type checker interpreter pins.

        Search paths belong to FlextInfraExtraPathsManager. Top-level
        ``venv`` / ``venvPath`` belong to deps modernize (root vs child
        runtime). Conform must not strip those or gen oscillates.
        """
        tool = u.Cli.toml_table_child(document, c.Infra.TOOL)
        if tool is None:
            return r[bool].ok(True)
        pyrefly = u.Cli.toml_table_child(tool, c.Infra.PYREFLY)
        if pyrefly is not None:
            u.Cli.toml_remove_key_if_present(pyrefly, "python-interpreter-path")

        pyright = u.Cli.toml_table_child(tool, c.Infra.PYRIGHT)
        if pyright is None:
            return r[bool].ok(True)

        # venv / venvPath are owned by deps modernize (workspace vs child
        # runtime). Conform only strips checkout-absolute interpreter pins.
        interpreter_keys = ("pythonPath", "pythonInterpreterPath")
        for key in interpreter_keys:
            u.Cli.toml_remove_key_if_present(pyright, key)
        raw_environments = u.Cli.json_as_sequence(
            u.Cli.toml_value(pyright, "executionEnvironments")
        )
        normalized_environments: t.JsonValueList = []
        for index, environment in enumerate(raw_environments):
            if not isinstance(environment, Mapping):
                return r[bool].fail(
                    f"tool.pyright.executionEnvironments[{index}] must be a mapping"
                )
            mapping = t.Cli.JSON_MAPPING_ADAPTER.validate_python(environment)
            normalized: t.JsonDict = dict(mapping)
            root = normalized.get("root")
            normalized[c.Infra.EXTRA_PATHS] = ["src"] if root == "src" else [".", "src"]
            for key in ("venv", "venvPath", "pythonPath", "pythonInterpreterPath"):
                normalized.pop(key, None)
            normalized_environments.append(normalized)
        if raw_environments:
            u.Cli.toml_sync_value(
                pyright, "executionEnvironments", normalized_environments
            )
        return r[bool].ok(True)

    @classmethod
    def _sync_uv_sources(
        cls,
        document: t.Cli.TomlDocument,
        *,
        project_name: str,
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
        link_mode: str | None = None,
        exclude_newer: str | None = None,
        exclude_newer_packages: t.StrSequence | None = None,
        exclude_newer_overrides: t.StrMapping | None = None,
        constraint_dependencies: t.SequenceOf[str] | None = None,
        exclude_dependencies: t.SequenceOf[p.Model] | None = None,
        uv_environments: t.StrSequence | None = None,
    ) -> p.Result[bool]:
        """Keep managed uv sources only as the root local-workspace overlay."""
        repository_root = cls._is_workspace_context_root(
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )
        nothing_to_write = (
            not repository_root
            and link_mode is None
            and exclude_newer is None
            and not exclude_newer_packages
            and not exclude_dependencies
            and not constraint_dependencies
        )
        tool = u.Cli.toml_table_child(document, c.Infra.TOOL)
        if tool is None:
            if nothing_to_write:
                return r[bool].ok(True)
            tool = u.Cli.toml_ensure_table(document, c.Infra.TOOL)
        uv = u.Cli.toml_table_child(tool, "uv")
        if uv is None:
            if nothing_to_write:
                return r[bool].ok(True)
            uv = u.Cli.toml_ensure_table(tool, "uv")
        u.Cli.toml_remove_key_if_present(uv, "required-version")
        # The fleet toolchain SSOT owns the constraints of every generated
        # pyproject, root and member alike, so each lock resolves inside the
        # range every sibling can install (flext-yoxv7: the floor rewrite raised
        # flext-core's structlog floor to its own lock and broke every meltano
        # consumer). The dependencies-only conform carries no toolchain and
        # keeps what the file declares.
        existing_constraints = u.Cli.toml_as_string_list(
            u.Cli.toml_value(uv, "constraint-dependencies")
        )
        selected_constraints = (
            tuple(constraint_dependencies)
            if constraint_dependencies is not None
            else existing_constraints
        )
        retained_constraints = tuple(
            requirement
            for requirement in selected_constraints
            if FlextInfraUtilitiesDependencies.dep_name(requirement) != "uv"
        )
        if retained_constraints:
            u.Cli.toml_sync_string_list(
                uv, "constraint-dependencies", retained_constraints
            )
        else:
            u.Cli.toml_remove_key_if_present(uv, "constraint-dependencies")
        if link_mode is not None:
            u.Cli.toml_sync_value(uv, "link-mode", link_mode)
        if exclude_newer is not None:
            u.Cli.toml_sync_value(uv, "exclude-newer", exclude_newer)
        # Environments come from the fleet toolchain SSOT: an empty declaration
        # removes the key so uv resolves every environment, and a declared
        # sequence skips the splits the fleet does not deploy to.
        if uv_environments is not None and uv_environments:
            # Declared as list[JsonValue], not list[str]: `list` is invariant,
            # so the narrower element type is not assignable to the writer's
            # parameter even though every element is a valid JsonValue.
            environments: list[t.JsonValue] = list(uv_environments)
            u.Cli.toml_sync_value(uv, "environments", environments)
        elif uv_environments is not None:
            u.Cli.toml_remove_key_if_present(uv, "environments")
        # Two shapes share this uv key. A bare exemption is `false` (waive the
        # cooldown entirely, for a reviewed security floor). An override is a
        # timestamp, needed when the shared cutoff predates a floor the project
        # legitimately requires: uv then reports the requirement unsatisfiable
        # and names this key as the remedy, so switching the cooldown off is not
        # enough — the cutoff has to move to a specific instant. Overrides win
        # on collision, being the more specific declaration of the two.
        if exclude_newer_packages is not None or exclude_newer_overrides is not None:
            exclude_newer_payload: t.JsonDict = dict.fromkeys(
                sorted(exclude_newer_packages or ()), False
            )
            exclude_newer_payload.update(
                sorted((exclude_newer_overrides or {}).items())
            )
            if exclude_newer_payload:
                u.Cli.toml_sync_value(
                    uv, "exclude-newer-package", exclude_newer_payload
                )
            else:
                u.Cli.toml_remove_key_if_present(uv, "exclude-newer-package")
        # Project is a flext-infra routing key only; uv scoped form is
        # {package={name, version?}, dependencies=[...]} (uv settings docs).
        # Emit on every owning pyproject so standalone CI clones resolve;
        # do not gate on owns_uv_root_policy (that stripped member excludes).
        if exclude_dependencies is not None:
            exclude_payload = list(
                t.Cli.JSON_LIST_ADAPTER.validate_python([
                    {
                        key: value
                        for key, value in item.model_dump(
                            mode="json", exclude_none=True
                        ).items()
                        if key != "project"
                    }
                    for item in exclude_dependencies
                ])
            )
            if exclude_payload:
                u.Cli.toml_sync_value(uv, "exclude-dependencies", exclude_payload)
            else:
                u.Cli.toml_remove_key_if_present(uv, "exclude-dependencies")
        member_paths = tuple(
            member.path.as_posix() for member in workspace.declared_repositories
        )
        # A uv workspace with no members is not an empty workspace, it is a
        # declaration: uv reads the table's presence, not its contents, so an
        # empty one makes this project a *nested* workspace and refuses to set
        # up any parent that lists it as a member.
        if repository_root and member_paths:
            workspace_table = u.Cli.toml_table_child(uv, "workspace")
            if workspace_table is None:
                workspace_table = u.Cli.toml_ensure_table(uv, "workspace")
            u.Cli.toml_sync_string_list(workspace_table, "members", member_paths)
        else:
            u.Cli.toml_remove_key_if_present(uv, "workspace")
        sources = u.Cli.toml_table_child(uv, "sources")
        if sources is None and repository_root:
            sources = u.Cli.toml_ensure_table(uv, "sources")
        if sources is None:
            if not repository_root and not tuple(uv):
                u.Cli.toml_remove_key_if_present(tool, "uv")
            return r[bool].ok(True)
        workspace_names = {
            member.distribution for member in workspace.declared_repositories
        }
        for source_name in tuple(sources):
            # Preserve resolved
            # TOML tables in place so conformance cannot accumulate blank trivia.
            if source_name.startswith("flext-") and (
                not repository_root or source_name not in workspace_names
            ):
                u.Cli.toml_remove_key_if_present(sources, source_name)
        if repository_root:
            for member in workspace.declared_repositories:
                u.Cli.toml_sync_mapping_table(
                    sources, member.distribution, {"workspace": True}
                )
        elif not tuple(sources):
            u.Cli.toml_remove_key_if_present(uv, "sources")
        if not repository_root and not tuple(uv):
            u.Cli.toml_remove_key_if_present(tool, "uv")
        return r[bool].ok(True)

    @classmethod
    def _resolved_root_sources(
        cls,
        *,
        workspace: p.Infra.WorkspaceSpec,
        providers: t.SequenceOf[m.Infra.ProviderSpec],
    ) -> p.Result[dict[str, dict[str, t.JsonValue]]]:
        """Resolve the workspace source overlay from typed metadata."""
        candidates = (workspace.repository, *workspace.declared_repositories)
        for distribution in dict.fromkeys(item.distribution for item in candidates):
            reference_result = cls._repository_reference(
                distribution, repositories=candidates, providers=providers
            )
            if reference_result.failure:
                return r.fail(reference_result.error or "repository resolution failed")
        return r.ok({
            member.distribution: {"workspace": True}
            for member in workspace.declared_repositories
        })

    @staticmethod
    def _validate_root_uv_sources(
        document: t.Cli.TomlDocument,
        *,
        workspace: p.Infra.WorkspaceSpec,
        providers: t.SequenceOf[m.Infra.ProviderSpec],
    ) -> p.Result[bool]:
        """Validate the root overlay without rewriting out-of-order TOML tables."""
        payload = u.Cli.toml_as_mapping(document)
        if payload is None:
            return r[bool].fail("pyproject document is not a TOML mapping")
        tool = payload.get(c.Infra.TOOL)
        if not isinstance(tool, Mapping):
            return r[bool].fail("root pyproject must define [tool]")
        uv = tool.get("uv")
        if not isinstance(uv, Mapping):
            return r[bool].fail("root pyproject must define [tool.uv]")
        if "override-dependencies" in uv:
            return r[bool].fail(
                "root pyproject must not define tool.uv.override-dependencies"
            )
        uv_workspace = uv.get("workspace")
        if not isinstance(uv_workspace, Mapping):
            return r[bool].fail("root pyproject must define [tool.uv.workspace]")
        try:
            members = t.Infra.STR_SEQ_ADAPTER.validate_python(
                uv_workspace.get("members"), strict=True
            )
        except c.ValidationError as exc:
            return r[bool].fail_op("validate root uv workspace package entries", exc)
        expected_members = tuple(
            member.path.as_posix() for member in workspace.declared_repositories
        )
        if tuple(members) != expected_members:
            return r[bool].fail(
                "root uv workspace package entries differ from workspace SSOT"
            )
        sources = uv.get("sources")
        if not isinstance(sources, Mapping):
            return r[bool].fail("root pyproject must define [tool.uv.sources]")
        resolved_result = FlextInfraUtilitiesPyprojectConform._resolved_root_sources(
            workspace=workspace, providers=providers
        )
        if resolved_result.failure:
            return r[bool].fail(resolved_result.error or "repository resolution failed")
        expected_sources = resolved_result.value
        if tuple(sources) != tuple(expected_sources):
            return r[bool].fail("root uv workspace sources differ from workspace SSOT")
        for source_name, expected_source in expected_sources.items():
            source = sources.get(source_name)
            if (
                not isinstance(source, Mapping)
                or tuple(source) != tuple(expected_source)
                or dict(source) != expected_source
            ):
                return r[bool].fail(
                    f"root uv workspace sources differ from workspace SSOT: {source_name}"
                )
        return r[bool].ok(True)

    @classmethod
    def _validate_dependency_provenance(
        cls,
        document: t.Cli.TomlDocument,
        *,
        project_name: str,
        workspace: p.Infra.WorkspaceSpec,
        workspace_mode: c.Infra.MakeProfile,
    ) -> p.Result[bool]:
        """Require one internal dependency provenance for the active topology."""
        payload = u.Cli.toml_as_mapping(document)
        if payload is None:
            return r[bool].fail("pyproject document is not a TOML mapping")
        member_names = frozenset(
            member.distribution for member in workspace.declared_repositories
        )
        raw_values: list[str] = []
        project = payload.get(c.Infra.PROJECT)
        if not isinstance(project, Mapping):
            return r[bool].fail("pyproject content must define [project]")
        workspace_context_root = cls._is_workspace_context_root(
            project_name=project_name,
            workspace=workspace,
            workspace_mode=workspace_mode,
        )
        for key in (c.Infra.DEPENDENCIES, c.Infra.OPTIONAL_DEPENDENCIES):
            value = project.get(key)
            if isinstance(value, Mapping):
                for group in value.values():
                    raw_values.extend(u.Cli.toml_as_string_list(group))
            else:
                raw_values.extend(u.Cli.toml_as_string_list(value))
        groups = payload.get(c.Infra.DEPENDENCY_GROUPS)
        if isinstance(groups, Mapping):
            for group in groups.values():
                raw_values.extend(u.Cli.toml_as_string_list(group))
        for requirement in raw_values:
            dependency_name = FlextInfraUtilitiesDependencies.dep_name(requirement)
            if dependency_name not in member_names:
                continue
            has_direct_source = "@" in requirement.partition(";")[0]
            if workspace_context_root and has_direct_source:
                return r[bool].fail(
                    "workspace dependency declares a conflicting direct source: "
                    f"{dependency_name}"
                )
            if not workspace_context_root and not has_direct_source:
                return r[bool].fail(
                    "publishable dependency lacks configured Git source: "
                    f"{dependency_name}"
                )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraUtilitiesPyprojectConform"]
