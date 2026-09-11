"""Phase: Ensure standard Pyright configuration for strict type checking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m, t, u
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector

if TYPE_CHECKING:
    from flext_infra.deps.extra_paths import FlextInfraExtraPathsManager


class FlextInfraEnsurePyrightConfigPhase:
    """Ensure standard Pyright configuration for strict type checking."""

    def __init__(self, tool_config: m.Infra.ToolConfigDocument) -> None:
        """Initialize the phase with the canonical tool configuration."""
        self._tool_config = tool_config

    def _report_private_usage_for_env(
        self,
        env_dir: str,
        *,
        rules: m.Infra.PyrightConfig.PathRulesConfig | None = None,
    ) -> str:
        """Only ``source_dir`` (src/) is strict; all auto-discovered dirs relax."""
        effective_rules = rules or self._tool_config.tools.pyright.path_rules
        if env_dir == effective_rules.source_dir:
            source_report: str = effective_rules.source_report_private_usage
            return source_report
        test_report: str = effective_rules.test_like_report_private_usage
        return test_report

    def _env_entry(
        self,
        *,
        env_dir: str,
        root: str,
        extra_paths: t.StrSequence,
        rules: m.Infra.PyrightConfig.PathRulesConfig,
    ) -> m.Infra.PyrightConfig.ExecutionEnvironment:
        """Env entry."""
        return m.Infra.PyrightConfig.ExecutionEnvironment(
            root=root,
            report_private_usage=self._report_private_usage_for_env(
                env_dir, rules=rules
            ),
            extra_paths=[*extra_paths],
        )

    def _extra_paths_for_env(
        self, *, env_dir: str, source_path: str, project_root: str, source_dir: str
    ) -> t.StrSequence:
        """``src/`` owns only its own path; every other discovered dir also imports from src + root."""
        if env_dir == source_dir:
            return [source_path]
        if source_path != project_root:
            return [project_root, source_path]
        return [project_root]

    def _envs_for_dirs(
        self,
        *,
        env_dirs: t.StrSequence,
        source_path: str,
        project_root: str,
        rules: m.Infra.PyrightConfig.PathRulesConfig,
    ) -> t.SequenceOf[m.Infra.PyrightConfig.ExecutionEnvironment]:
        """Envs for dirs."""
        return [
            self._env_entry(
                env_dir=env_dir,
                root=env_dir,
                extra_paths=self._extra_paths_for_env(
                    env_dir=env_dir,
                    source_path=source_path,
                    project_root=project_root,
                    source_dir=rules.source_dir,
                ),
                rules=rules,
            )
            for env_dir in env_dirs
        ]

    def _diagnostic_override_envs(
        self, *, project_dir: Path | None, root_prefix: Path | None, source_path: str
    ) -> t.SequenceOf[m.Infra.PyrightConfig.ExecutionEnvironment]:
        """Resolve configured overrides only when their project path exists."""
        if project_dir is None:
            return ()
        rules = self._tool_config.tools.pyright.path_rules
        environments: t.MutableSequenceOf[
            m.Infra.PyrightConfig.ExecutionEnvironment
        ] = []
        for override in rules.diagnostic_path_overrides:
            if not (project_dir / override.root).is_dir():
                continue
            resolved_root = (
                root_prefix / override.root if root_prefix else Path(override.root)
            ).as_posix()
            environments.append(
                m.Infra.PyrightConfig.ExecutionEnvironment(
                    root=resolved_root,
                    report_private_usage=override.report_private_usage,
                    extra_paths=(source_path,),
                    rationale=override.rationale,
                )
            )
        return tuple(environments)

    def _project_source_path(self, *, prefix: str = "") -> str:
        """Project source path."""
        rules = self._tool_config.tools.pyright.path_rules
        return f"{prefix}/{rules.source_dir}" if prefix else rules.source_dir

    def _expected_envs(
        self,
        *,
        is_root: bool,
        repository_root: Path | None,
        project_dir: Path | None,
        project_roots: t.StrSequence,
    ) -> t.SequenceOf[m.Infra.PyrightConfig.ExecutionEnvironment]:
        """Return the expected execution environments."""
        if not is_root or repository_root is None:
            rules = self._tool_config.tools.pyright.path_rules
            return (
                *self._diagnostic_override_envs(
                    project_dir=project_dir,
                    root_prefix=None,
                    source_path=self._project_source_path(),
                ),
                *self._envs_for_dirs(
                    env_dirs=project_roots,
                    source_path=self._project_source_path(),
                    project_root=rules.project_root,
                    rules=rules,
                ),
            )
        rules = self._tool_config.tools.pyright.path_rules
        expected_envs: t.MutableSequenceOf[
            m.Infra.PyrightConfig.ExecutionEnvironment
        ] = []
        root_source_path = self._project_source_path()
        # Specific roots precede the broad source environment.
        expected_envs.extend(
            self._diagnostic_override_envs(
                project_dir=repository_root,
                root_prefix=None,
                source_path=root_source_path,
            )
        )
        for env_dir in project_roots:
            if (repository_root / env_dir / c.Infra.PYPROJECT_FILENAME).is_file():
                continue
            expected_envs.append(
                self._env_entry(
                    env_dir=env_dir,
                    root=env_dir,
                    extra_paths=self._extra_paths_for_env(
                        env_dir=env_dir,
                        source_path=root_source_path,
                        project_root=rules.project_root,
                        source_dir=rules.source_dir,
                    ),
                    rules=rules,
                )
            )
        return expected_envs

    def environment_payloads_for_dirs(
        self, env_dirs: t.StrSequence
    ) -> t.SequenceOf[t.JsonDict]:
        """Render configured environments for Python roots declared before writes."""
        rules = self._tool_config.tools.pyright.path_rules
        environments = self._envs_for_dirs(
            env_dirs=self._declared_environment_dirs(env_dirs),
            source_path=self._project_source_path(),
            project_root=rules.project_root,
            rules=rules,
        )
        return tuple(self._environment_payload(item) for item in environments)

    @staticmethod
    def _declared_environment_dirs(env_dirs: t.StrSequence) -> t.StrSequence:
        """Apply canonical Python discovery exclusions to pre-write declarations."""
        # Declared and on-disk roots must
        # select the same first-class analyzer environments in the first pass.
        return tuple(
            env_dir
            for env_dir in env_dirs
            if env_dir not in c.Infra.PYTHON_DISCOVERY_SKIP_DIRS
        )

    def _environment_payload(
        self, environment: m.Infra.PyrightConfig.ExecutionEnvironment
    ) -> t.JsonDict:
        """Render one environment with its closed, scope-specific diagnostics."""
        rules = self._tool_config.tools.pyright.path_rules
        env_dir = Path(environment.root).name
        diagnostics: t.MutableStrMapping = {
            **self._tool_config.tools.pyright.lazy_import_suppressions
        }
        if env_dir == rules.source_dir:
            diagnostics.update(self._tool_config.tools.pyright.source_env_suppressions)
        elif env_dir in rules.test_like_dirs:
            diagnostics.update(
                self._tool_config.tools.pyright.test_like_env_suppressions
            )
        payload: t.JsonDict = environment.model_dump(mode="json", by_alias=True)
        payload.update(diagnostics)
        return payload

    def _override_for_kind(
        self, project_kind: str
    ) -> m.Infra.ProjectTypeOverrideConfig | None:
        """Return the project-type override settings for the given kind."""
        overrides = self._tool_config.project_type_overrides
        kind_map: t.MappingKV[str, m.Infra.ProjectTypeOverrideConfig] = {
            "core": overrides.core,
            "domain": overrides.domain,
            "platform": overrides.platform,
            "integration": overrides.integration,
            "app": overrides.app,
        }
        raw = kind_map.get(project_kind)
        if raw is None:
            return None
        validated: m.Infra.ProjectTypeOverrideConfig = (
            m.Infra.ProjectTypeOverrideConfig.model_validate(raw)
        )
        return validated

    def _venv_settings(self, *, is_root: bool) -> t.StrMapping:
        """Return the Pyright venv location for this topology."""
        rules = self._tool_config.tools.pyright.path_rules
        venv_path = "." if is_root else ".."
        return {c.Infra.VENV_PATH: venv_path, "venv": rules.venv_name}

    def _expected_excludes(
        self, project_root: Path | None, analysis_exclusions: t.StrSequence | None
    ) -> t.StrSequence:
        """Return the complete config-owned Pyright exclude list."""
        rules = self._tool_config.tools.pyright.path_rules
        workspace_excludes: t.StrSequence = ()
        if analysis_exclusions is None and project_root is not None:
            excluded = FlextInfraWorkspaceDetector.analysis_exclusion_paths(
                project_root
            )
            if excluded.failure:
                raise ValueError(
                    excluded.error or "workspace analysis scope is unavailable"
                )
            workspace_excludes = tuple(path.as_posix() for path in excluded.value)
        provided_exclusions = () if analysis_exclusions is None else analysis_exclusions
        return sorted({
            *rules.default_excludes,
            *workspace_excludes,
            *provided_exclusions,
        })

    def _existing_paths(
        self, base_dir: Path | None, configured_paths: t.StrSequence
    ) -> t.StrSequence:
        """Existing paths."""
        if base_dir is None:
            return ()
        existing: t.StrSequence = [
            relative_path
            for relative_path in configured_paths
            if (base_dir / relative_path).is_dir()
        ]
        return existing

    def _expected_ignores(
        self, *, is_root: bool, repository_root: Path | None, project_dir: Path | None
    ) -> t.StrSequence:
        """Ignore typings and stub diagnostics."""
        rules = self._tool_config.tools.pyright.path_rules
        ignores: t.MutableSequenceOf[str] = []
        if is_root:
            root_dir = repository_root or project_dir
            ignores.extend(self._existing_paths(root_dir, rules.root_typings_paths))
        else:
            ignores.extend(
                self._existing_paths(project_dir, rules.project_typings_paths)
            )
        for pattern in rules.ignored_diagnostic_globs:
            if pattern not in ignores:
                ignores.append(pattern)
        return list(ignores)

    def _expected_includes(
        self,
        *,
        is_root: bool,
        repository_root: Path | None,
        project_roots: t.StrSequence,
    ) -> t.StrSequence:
        """Return only Python roots owned by the selected project manifest."""
        _ = is_root, repository_root
        return list(project_roots)

    def _expected_project_roots(
        self,
        *,
        is_root: bool,
        repository_root: Path | None,
        project_dir: Path | None,
        declared_python_dirs: t.StrSequence,
        declared_python_dirs_are_complete: bool,
        generated_roots: t.StrSequence,
    ) -> t.StrSequence:
        """Resolve the one analyzer-root set consumed by includes and environments."""
        declared = self._declared_environment_dirs(
            tuple(dict.fromkeys((*declared_python_dirs, *generated_roots)))
        )
        if (
            is_root
            and repository_root is not None
            and (repository_root / c.Infra.GITMODULES).is_file()
        ):
            return u.Infra.analyzer_python_roots(repository_root, generated_roots)
        if declared_python_dirs_are_complete:
            return declared
        if project_dir is not None:
            return u.Infra.analyzer_python_roots(project_dir, declared)
        if declared:
            return declared
        return self._tool_config.tools.pyright.path_rules.env_dirs

    def _phase(
        self,
        *,
        is_root: bool,
        repository_root: Path | None = None,
        project_dir: Path | None = None,
        project_kind: str = "core",
        paths_manager: FlextInfraExtraPathsManager | None = None,
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> m.Infra.Deps.Toml.PhaseConfig:
        """Build the managed pyright phase for one project context."""
        project_root = repository_root if is_root else project_dir
        expected_excludes = self._expected_excludes(project_root, analysis_exclusions)
        expected_ignores = self._expected_ignores(
            is_root=is_root, repository_root=repository_root, project_dir=project_dir
        )
        generated_roots = (
            paths_manager.generated_python_roots if paths_manager is not None else ()
        )
        expected_roots = self._expected_project_roots(
            is_root=is_root,
            repository_root=repository_root,
            project_dir=project_dir,
            declared_python_dirs=declared_python_dirs,
            declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            generated_roots=generated_roots,
        )
        expected_includes = self._expected_includes(
            is_root=is_root,
            repository_root=repository_root,
            project_roots=expected_roots,
        )
        stub_rules = self._tool_config.tools.pyright.path_rules
        expected_stub_path: str | None = (
            stub_rules.root_typings_paths[0]
            if is_root and stub_rules.root_typings_paths
            else (
                stub_rules.project_typings_paths[0]
                if stub_rules.project_typings_paths
                else None
            )
        )
        expected_envs = self._expected_envs(
            is_root=is_root,
            repository_root=repository_root,
            project_dir=project_dir,
            project_roots=expected_roots,
        )
        phase_builder = m.Infra.Deps.Toml.PhaseConfig.Builder("pyright").table(
            c.Infra.PYRIGHT
        )
        if expected_excludes:
            phase_builder = phase_builder.list(c.Infra.EXCLUDE, expected_excludes)
        else:
            phase_builder = phase_builder.deprecated(c.Infra.EXCLUDE)
        if expected_ignores:
            phase_builder = phase_builder.list(c.Infra.IGNORE, expected_ignores)
        else:
            phase_builder = phase_builder.deprecated(c.Infra.IGNORE)
        if expected_includes:
            phase_builder = phase_builder.list("include", expected_includes)
        else:
            phase_builder = phase_builder.deprecated("include")
        if project_root is not None and paths_manager is not None:
            phase_builder = phase_builder.list(
                "extraPaths",
                paths_manager.pyright_extra_paths(
                    project_dir=project_root, is_root=is_root
                ),
            )
        if expected_stub_path is not None:
            existing = project_root / expected_stub_path if project_root else None
            if existing is not None and existing.is_dir():
                phase_builder = phase_builder.value("stubPath", expected_stub_path)
            else:
                phase_builder = phase_builder.deprecated("stubPath")
        else:
            phase_builder = phase_builder.deprecated("stubPath")
        phase_builder = phase_builder.deprecated("venv").deprecated(c.Infra.VENV_PATH)
        phase_builder = phase_builder.value(
            "executionEnvironments",
            [
                u.normalize_to_json_value(self._environment_payload(expected_env))
                for expected_env in expected_envs
            ],
        )
        if is_root:
            for key, value in self._tool_config.tools.pyright.strict_settings.items():
                phase_builder = phase_builder.value(key, value)
            for key, value in self._tool_config.tools.pyright.extended_settings.items():
                phase_builder = phase_builder.value(key, value)
            return phase_builder.build()
        for key, value in self._tool_config.tools.pyright.strict_settings.items():
            phase_builder = phase_builder.value(key, value)
        merged_settings: t.MutableStrMapping = {
            **self._tool_config.tools.pyright.extended_settings
        }
        override = self._override_for_kind(project_kind)
        if override is not None:
            merged_settings.update(override.pyright)
        for key, value in merged_settings.items():
            phase_builder = phase_builder.value(key, value)
        return phase_builder.build()

    def apply(
        self,
        doc: t.Cli.TomlDocument,
        *,
        is_root: bool,
        repository_root: Path | None = None,
        project_dir: Path | None = None,
        project_kind: str = "core",
        paths_manager: FlextInfraExtraPathsManager | None = None,
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> t.StrSequence:
        """Apply the managed pyright configuration for one TOML document."""
        return FlextInfraTomlPhaseService.apply_phases(
            doc,
            self._phase(
                is_root=is_root,
                repository_root=repository_root,
                project_dir=project_dir,
                project_kind=project_kind,
                paths_manager=paths_manager,
                declared_python_dirs=declared_python_dirs,
                declared_python_dirs_are_complete=declared_python_dirs_are_complete,
                analysis_exclusions=analysis_exclusions,
            ),
        )

    def apply_payload(
        self,
        payload: t.MutableJsonMapping,
        *,
        is_root: bool,
        repository_root: Path | None = None,
        project_dir: Path | None = None,
        project_kind: str = "core",
        paths_manager: FlextInfraExtraPathsManager | None = None,
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> t.StrSequence:
        """Apply managed pyright settings directly to one normalized payload."""
        return FlextInfraTomlPhaseService.apply_payload_phases(
            payload,
            self._phase(
                is_root=is_root,
                repository_root=repository_root,
                project_dir=project_dir,
                project_kind=project_kind,
                paths_manager=paths_manager,
                declared_python_dirs=declared_python_dirs,
                declared_python_dirs_are_complete=declared_python_dirs_are_complete,
                analysis_exclusions=analysis_exclusions,
            ),
        )


__all__: list[str] = ["FlextInfraEnsurePyrightConfigPhase"]
