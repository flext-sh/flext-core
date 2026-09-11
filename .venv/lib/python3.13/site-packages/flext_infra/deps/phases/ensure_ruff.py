"""Phase: Ensure standard Ruff configuration inline with known-first-party overlay."""

from __future__ import annotations

from pathlib import Path

from flext_infra import c, config, m, t, u
from flext_infra._utilities.project_managed_artifacts import (
    FlextInfraUtilitiesProjectManagedArtifacts,
)
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector


class FlextInfraEnsureRuffConfigPhase:
    """Ensure standard Ruff configuration inline with known-first-party overlay."""

    def __init__(
        self,
        tool_config: m.Infra.ToolConfigDocument,
        managed_artifacts: m.Infra.ProjectManagedArtifactsResolution | None = None,
    ) -> None:
        """Store tool configuration used to build canonical Ruff settings."""
        self._tool_config = tool_config
        self._managed_artifacts = managed_artifacts

    @staticmethod
    def _workspace_project_namespaces(project_dir: Path) -> t.StrSequence:
        """Discover child project packages when generating repository root settings."""
        if not (project_dir / c.Infra.PYPROJECT_FILENAME).is_file():
            return ()
        discovered = u.Infra.discover_projects(project_dir)
        if discovered.failure:
            return ()
        return sorted({
            project.package_name
            for project in discovered.value
            if (
                project.package_name
                and project.package_name.isidentifier()
                and project.declared_subproject
            )
        })

    @staticmethod
    def _workspace_exclusion_globs(project_dir: Path) -> t.StrSequence:
        """Return immutable repository and explicit exclusion path globs.

        Content-only repositories are foreign, read-only trees and therefore
        never enter Ruff. Explicit ``exclusions`` extend that same typed scope
        for non-repository paths without duplicating repository declarations.
        """
        if not (project_dir / c.Infra.PYPROJECT_FILENAME).is_file():
            return ()
        paths = FlextInfraWorkspaceDetector.analysis_exclusion_paths(project_dir)
        if paths.failure:
            raise ValueError(paths.error or "workspace analysis scope is unavailable")
        return sorted(path.as_posix() for path in paths.value)

    @staticmethod
    def _remove_stale_lint_section(doc: t.Cli.TomlDocument) -> t.StrSequence:
        """Remove the stale top-level ``[lint]`` table left by old configs."""
        if c.Infra.LINT_SECTION not in doc:
            return ()
        del doc[c.Infra.LINT_SECTION]
        return ["removed stale top-level [lint] section"]

    @staticmethod
    def _remove_stale_lint_section_payload(
        payload: t.MutableJsonMapping,
    ) -> t.StrSequence:
        """Remove the stale top-level ``[lint]`` table from one plain payload."""
        if u.Cli.toml_mapping_remove_key_if_present(payload, c.Infra.LINT_SECTION):
            return ["removed stale top-level [lint] section"]
        return ()

    @staticmethod
    def _project_per_file_ignores(
        project_dir: Path,
        managed_artifacts: m.Infra.ProjectManagedArtifactsResolution | None = None,
    ) -> t.MappingKV[str, t.StrSequence]:
        """Load validated project-owned Ruff additions from ``config/*.yaml``."""
        if managed_artifacts is not None:
            return managed_artifacts.artifacts.Ruff.per_file_ignores
        if not project_dir.is_dir():
            # A scaffold target is materialized by this same plan, so it owns
            # no declared exemption yet. A directory that does exist but cannot
            # be inspected still fails loud below.
            return {}
        loaded = (
            FlextInfraUtilitiesProjectManagedArtifacts.load_project_managed_artifacts(
                project_dir
            )
        )
        if loaded.failure:
            raise ValueError(loaded.error or "project artifact load failed")
        return loaded.value.artifacts.Ruff.per_file_ignores

    def _per_file_ignores(self, project_dir: Path) -> t.MappingKV[str, t.StrSequence]:
        """Compose global policy with the current project's managed additions."""
        return FlextInfraEnsureRuffConfigPhase.compose_per_file_ignores(
            project_dir,
            global_ignores=self._tool_config.tools.ruff.lint.per_file_ignores,
            managed_artifacts=self._managed_artifacts,
        )

    @staticmethod
    def compose_per_file_ignores(
        project_dir: Path,
        *,
        global_ignores: t.MappingKV[str, t.StrSequence] | None = None,
        managed_artifacts: m.Infra.ProjectManagedArtifactsResolution | None = None,
    ) -> t.MappingKV[str, t.StrSequence]:
        """Return the effective Ruff exemption map for one project.

        The fleet policy is not the whole contract: a repository may declare an
        operator-authorized exemption in its own ``config/*.yaml``
        ``ManagedArtifacts`` block. Both the in-place pyproject edit and the
        full template render must see the same composed result, or a render
        silently drops the local overlay and reports findings the operator has
        already ruled on.
        """
        effective_global = (
            config.Infra.tooling.tools.ruff.lint.per_file_ignores
            if global_ignores is None
            else global_ignores
        )
        local_ignores = FlextInfraEnsureRuffConfigPhase._project_per_file_ignores(
            project_dir, managed_artifacts
        )
        return {
            pattern: tuple(sorted({*effective_global.get(pattern, ()), *rules}))
            for pattern, rules in {**effective_global, **local_ignores}.items()
        }

    def _phase(
        self,
        *,
        path: Path,
        workspace_namespaces: t.StrSequence,
        stale_patterns: t.StrSequence,
        per_file_ignores: t.MappingKV[str, t.StrSequence],
        include_handler: bool,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> m.Infra.Deps.Toml.PhaseConfig:
        """Build the canonical Ruff phase for one project path."""
        ruff_cfg = self._tool_config.tools.ruff
        effective_src = sorted(ruff_cfg.src)
        workspace_exclusions = (
            self._workspace_exclusion_globs(path.parent)
            if analysis_exclusions is None
            else ()
        )
        provided_exclusions = () if analysis_exclusions is None else analysis_exclusions
        effective_exclude = sorted({
            *ruff_cfg.exclude,
            *workspace_exclusions,
            *provided_exclusions,
        })
        # Models stay declaration-only; the
        # Ruff phase owns the derived union consumed by emitted tool config.
        effective_ignore = tuple(
            sorted({*ruff_cfg.lint.ignore, *ruff_cfg.lint.ignored_rule_rationales})
        )
        detected_packages = sorted({
            *config.Infra.tooling.tools.deptry.known_first_party,
            *u.Infra.discover_first_party_namespaces(path.parent),
            *workspace_namespaces,
            *self._workspace_project_namespaces(path.parent),
        })
        lint_nested_values: t.SequenceOf[tuple[str, t.JsonValue]] = (
            ("select", u.normalize_to_json_value(sorted(ruff_cfg.lint.select))),
            (c.Infra.IGNORE, u.normalize_to_json_value(effective_ignore)),
        )
        isort_values: list[tuple[str, t.JsonValue]] = [
            ("combine-as-imports", ruff_cfg.lint.isort.combine_as_imports),
            ("force-single-line", ruff_cfg.lint.isort.force_single_line),
            ("split-on-trailing-comma", ruff_cfg.lint.isort.split_on_trailing_comma),
        ]
        if detected_packages:
            isort_values.append((
                c.Infra.KNOWN_FIRST_PARTY_HYPHEN,
                u.normalize_to_json_value(detected_packages),
            ))
        phase = (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("ruff")
            .table(c.Infra.RUFF)
            .deprecated(c.Infra.EXTEND)
            .list(c.Infra.EXCLUDE, effective_exclude)
            .list("namespace-packages", sorted(ruff_cfg.namespace_packages))
            .value("fix", ruff_cfg.fix)
            .value("line-length", ruff_cfg.line_length)
            .value("preview", ruff_cfg.preview)
            .value("respect-gitignore", ruff_cfg.respect_gitignore)
            .value("show-fixes", ruff_cfg.show_fixes)
            .value("target-version", ruff_cfg.target_version)
            .list("src", effective_src)
            .nested(
                "format",
                values=(
                    ("docstring-code-format", ruff_cfg.format.docstring_code_format),
                    ("indent-style", ruff_cfg.format.indent_style),
                    ("line-ending", ruff_cfg.format.line_ending),
                    ("quote-style", ruff_cfg.format.quote_style),
                    (
                        "skip-magic-trailing-comma",
                        ruff_cfg.format.skip_magic_trailing_comma,
                    ),
                ),
            )
            .nested(c.Infra.LINT_SECTION, values=lint_nested_values)
            .nested(
                c.Infra.LINT_SECTION,
                "flake8-tidy-imports",
                "banned-api",
                values=tuple(
                    (name, u.normalize_to_json_value({"msg": message}))
                    for name, message in ruff_cfg.lint.banned_api.items()
                ),
            )
            .nested(c.Infra.LINT_SECTION, c.Infra.ISORT, values=tuple(isort_values))
            .nested(
                c.Infra.LINT_SECTION,
                "per-file-ignores",
                values=tuple(
                    (pattern, u.normalize_to_json_value(sorted(rules)))
                    for pattern, rules in per_file_ignores.items()
                ),
                deprecated_keys=stale_patterns,
            )
            .build()
        )
        if include_handler:
            configured: m.Infra.Deps.Toml.PhaseConfig = phase.model_copy(
                update={"custom_handler": self._remove_stale_lint_section}
            )
            return configured
        return phase

    def apply(
        self,
        doc: t.Cli.TomlDocument,
        *,
        path: Path,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> t.StrSequence:
        """Apply canonical Ruff tables with namespace-aware first-party detection."""
        effective_ignores = self._per_file_ignores(path.parent)
        per_file_ignores = u.Cli.toml_table_path(
            doc, (c.Infra.TOOL, c.Infra.RUFF, c.Infra.LINT_SECTION, "per-file-ignores")
        )
        stale_patterns = (
            [
                pattern
                for pattern in per_file_ignores
                if pattern not in effective_ignores
            ]
            if per_file_ignores is not None
            else ()
        )
        return FlextInfraTomlPhaseService.apply_phases(
            doc,
            self._phase(
                path=path,
                # Installed and workspace FLEXT dependencies
                # share the same first-party import contract.
                workspace_namespaces=u.Infra.flext_dependency_namespaces(doc),
                stale_patterns=stale_patterns,
                per_file_ignores=effective_ignores,
                include_handler=True,
                analysis_exclusions=analysis_exclusions,
            ),
        )

    def apply_payload(
        self,
        payload: t.MutableJsonMapping,
        *,
        path: Path,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> t.StrSequence:
        """Apply canonical Ruff settings directly to one normalized payload."""
        effective_ignores = self._per_file_ignores(path.parent)
        per_file_ignores = u.Cli.toml_mapping_path(
            payload,
            (c.Infra.TOOL, c.Infra.RUFF, c.Infra.LINT_SECTION, "per-file-ignores"),
        )
        stale_patterns = (
            [
                pattern
                for pattern in list(per_file_ignores)
                if pattern not in effective_ignores
            ]
            if per_file_ignores is not None
            else ()
        )
        changes = list(
            FlextInfraTomlPhaseService.apply_payload_phases(
                payload,
                self._phase(
                    path=path,
                    workspace_namespaces=tuple(
                        u.Infra.flext_dependency_namespaces_from_payload(payload)
                    )
                    + config.Infra.tooling.tools.deptry.known_first_party
                    + (
                        u.Infra.project_name_from_payload(path, payload).replace(
                            "-", "_"
                        ),
                    ),
                    stale_patterns=stale_patterns,
                    per_file_ignores=effective_ignores,
                    include_handler=False,
                    analysis_exclusions=analysis_exclusions,
                ),
            )
        )
        changes.extend(self._remove_stale_lint_section_payload(payload))
        return changes


__all__: list[str] = ["FlextInfraEnsureRuffConfigPhase"]
