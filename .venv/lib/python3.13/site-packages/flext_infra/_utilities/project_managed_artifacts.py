"""Load and compose project-owned managed-artifact configuration."""

from __future__ import annotations

import os
import stat
from fnmatch import fnmatchcase
from pathlib import Path

from flext_core import r
from flext_cli import u
from flext_infra import c, config, m, p, t


class FlextInfraUtilitiesProjectManagedArtifacts:
    """Single owner for ``ManagedArtifacts`` across ``config/*.yaml`` files."""

    @classmethod
    def snapshot_config_sources(
        cls, project_dir: Path
    ) -> p.Result[tuple[m.Cli.AtomicFileState, ...]]:
        """Capture one stable, physical, direct ``config/*.yaml`` file set."""
        project_identity = cls._required_directory_identity(
            project_dir, purpose="project root"
        )
        if project_identity.failure:
            return r[tuple[m.Cli.AtomicFileState, ...]].from_failure(project_identity)
        config_dir = project_dir / c.CONFIG_DIR_NAME
        identity = cls._config_directory_identity(config_dir)
        if identity.failure:
            return r[tuple[m.Cli.AtomicFileState, ...]].from_failure(identity)
        if not identity.value:
            return r[tuple[m.Cli.AtomicFileState, ...]].ok(())
        paths = cls._config_yaml_paths(config_dir, identity.value)
        if paths.failure:
            return r[tuple[m.Cli.AtomicFileState, ...]].from_failure(paths)
        sources: list[m.Cli.AtomicFileState] = []
        for path in paths.value:
            source = u.Cli.atomic_read_binary_file_state(path, required=True)
            if source.failure:
                return r[tuple[m.Cli.AtomicFileState, ...]].from_failure(source)
            sources.append(source.value)
        stable_paths = cls._config_yaml_paths(config_dir, identity.value)
        if stable_paths.failure:
            return r[tuple[m.Cli.AtomicFileState, ...]].from_failure(stable_paths)
        if stable_paths.value != paths.value:
            return r[tuple[m.Cli.AtomicFileState, ...]].fail(
                f"project config source topology changed: {config_dir}"
            )
        stable_project = cls._required_directory_identity(
            project_dir, purpose="project root"
        )
        if stable_project.failure:
            return r[tuple[m.Cli.AtomicFileState, ...]].from_failure(stable_project)
        if stable_project.value != project_identity.value:
            return r[tuple[m.Cli.AtomicFileState, ...]].fail(
                f"project root changed during config snapshot: {project_dir}"
            )
        return r[tuple[m.Cli.AtomicFileState, ...]].ok(tuple(sources))

    @classmethod
    def _required_directory_identity(
        cls, path: Path, *, purpose: str
    ) -> p.Result[tuple[int, ...]]:
        try:
            state = path.lstat()
        except OSError as exc:
            return r[tuple[int, ...]].fail_op(f"inspect {purpose}", exc)
        attributes = getattr(state, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if not stat.S_ISDIR(state.st_mode) or attributes & reparse:
            return r[tuple[int, ...]].fail(
                f"{purpose} is not a physical directory: {path}"
            )
        return r[tuple[int, ...]].ok(cls._directory_state_key(state))

    @classmethod
    def _config_directory_identity(cls, config_dir: Path) -> p.Result[tuple[int, ...]]:
        """Return the directory's state key, or ``()`` when it does not exist.

        A repository legitimately ships no ``config/`` directory, which is an
        empty declaration set rather than a failure. The empty tuple is
        unambiguous: a real key always carries the eight ``stat`` fields, and a
        successful Result can never carry ``None``.
        """
        try:
            state = config_dir.lstat()
        except FileNotFoundError:
            return r[tuple[int, ...]].ok(())
        except OSError as exc:
            return r[tuple[int, ...]].fail_op("inspect project config directory", exc)
        attributes = getattr(state, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if not stat.S_ISDIR(state.st_mode) or attributes & reparse:
            return r[tuple[int, ...]].fail(
                f"project config path is not a physical directory: {config_dir}"
            )
        return r[tuple[int, ...]].ok(cls._directory_state_key(state))

    @classmethod
    def _config_yaml_paths(
        cls, config_dir: Path, expected_identity: tuple[int, ...]
    ) -> p.Result[tuple[Path, ...]]:
        try:
            paths = tuple(
                sorted(path for path in config_dir.iterdir() if path.suffix == ".yaml")
            )
        except OSError as exc:
            return r[tuple[Path, ...]].fail_op("enumerate project config sources", exc)
        current = cls._config_directory_identity(config_dir)
        if current.failure:
            return r[tuple[Path, ...]].from_failure(current)
        if current.value != expected_identity:
            return r[tuple[Path, ...]].fail(
                f"project config directory changed during snapshot: {config_dir}"
            )
        return r[tuple[Path, ...]].ok(paths)

    @staticmethod
    def _directory_state_key(state: os.stat_result) -> tuple[int, ...]:
        return (
            state.st_dev,
            state.st_ino,
            state.st_mode,
            state.st_uid,
            state.st_gid,
            state.st_size,
            state.st_mtime_ns,
            state.st_ctime_ns,
        )

    @staticmethod
    def validate_mise_tool_selectors(
        selectors: t.StrSequence, *, source: Path
    ) -> p.Result[bool]:
        """Reject alternate distributions of fleet-owned tool identities."""
        toolchain = config.Infra.codegen.toolchain
        protected_tools = tuple(
            (owner, getattr(toolchain, owner))
            for owner in toolchain.protected_mise_tools
        )
        for selector in selectors:
            for owner, tool in protected_tools:
                if not any(
                    fnmatchcase(selector, pattern) for pattern in tool.selector_patterns
                ):
                    continue
                if selector == tool.selector:
                    break
                return r[bool].fail(
                    "project Mise selector declares an alternate distribution "
                    f"for fleet identity {owner!r}: {selector!r} in "
                    f"{source}; canonical selector is {tool.selector!r}"
                )
        return r[bool].ok(True)

    @classmethod
    def load_project_managed_artifacts(
        cls, project_dir: Path
    ) -> p.Result[m.Infra.ProjectManagedArtifactsResolution]:
        """Snapshot every YAML once and parse that exact source set."""
        snapshot = cls.snapshot_project_managed_artifacts(project_dir)
        if snapshot.failure:
            return r[m.Infra.ProjectManagedArtifactsResolution].from_failure(snapshot)
        return r[m.Infra.ProjectManagedArtifactsResolution].ok(
            snapshot.value.resolution
        )

    @classmethod
    def snapshot_project_managed_artifacts(
        cls, project_dir: Path
    ) -> p.Result[m.Infra.ProjectManagedArtifactsSnapshot]:
        """Capture and parse exactly one immutable project configuration view."""
        source_snapshot = cls.snapshot_config_sources(project_dir)
        if source_snapshot.failure:
            return r[m.Infra.ProjectManagedArtifactsSnapshot].from_failure(
                source_snapshot
            )
        resolution = cls.load_project_managed_artifacts_from_snapshot(
            source_snapshot.value
        )
        if resolution.failure:
            return r[m.Infra.ProjectManagedArtifactsSnapshot].from_failure(resolution)
        return r[m.Infra.ProjectManagedArtifactsSnapshot].ok(
            m.Infra.ProjectManagedArtifactsSnapshot(
                sources=source_snapshot.value, resolution=resolution.value
            )
        )

    @classmethod
    def load_project_managed_artifacts_from_snapshot(
        cls, source_snapshot: tuple[m.Cli.AtomicFileState, ...]
    ) -> p.Result[m.Infra.ProjectManagedArtifactsResolution]:
        """Parse one caller-owned immutable project YAML snapshot."""
        empty = m.Infra.ProjectManagedArtifactsResolution(
            artifacts=m.Infra.ProjectManagedArtifactsConfig(), mise_tool_sources={}
        )
        if not source_snapshot:
            return r[m.Infra.ProjectManagedArtifactsResolution].ok(empty)
        ruff_ignores: dict[str, set[str]] = {}
        mise_tools: dict[str, m.Infra.ProjectMiseTool] = {}
        mise_sources: dict[str, Path] = {}
        gitignore_patterns: list[str] = []
        fleet_platforms = frozenset(config.Infra.codegen.toolchain.mise_lock_platforms)
        for source_state in source_snapshot:
            source = source_state.path
            if source_state.content is None:
                return r[m.Infra.ProjectManagedArtifactsResolution].fail(
                    f"project config snapshot is absent: {source}"
                )
            try:
                source_text = source_state.content.decode(c.Cli.ENCODING_DEFAULT)
            except UnicodeDecodeError as exc:
                return r[m.Infra.ProjectManagedArtifactsResolution].fail_op(
                    f"decode project config source {source}", exc
                )
            loaded = u.Cli.yaml_parse(source_text)
            if loaded.failure:
                return r[m.Infra.ProjectManagedArtifactsResolution].fail(
                    loaded.error or f"project config load failed: {source}"
                )
            managed = loaded.value.get("ManagedArtifacts")
            if not managed:
                continue
            project_config = m.Infra.ProjectConfigDocument.model_validate({
                "ManagedArtifacts": managed
            })
            artifacts = project_config.ManagedArtifacts
            for pattern, rules in artifacts.Ruff.per_file_ignores.items():
                ruff_ignores.setdefault(pattern, set()).update(rules)
            for pattern in artifacts.Gitignore.patterns:
                if pattern not in gitignore_patterns:
                    gitignore_patterns.append(pattern)
            for selector, tool in artifacts.Mise.tools.items():
                previous = mise_sources.get(selector)
                if previous is not None:
                    return r[m.Infra.ProjectManagedArtifactsResolution].fail(
                        "duplicate project Mise selector "
                        f"{selector!r}: {previous} and {source}"
                    )
                unknown = sorted(set(tool.platforms or ()) - fleet_platforms)
                if unknown:
                    return r[m.Infra.ProjectManagedArtifactsResolution].fail(
                        f"project Mise selector {selector!r} in {source} declares "
                        f"platforms outside the fleet lock platforms: {unknown}"
                    )
                mise_tools[selector] = tool
                mise_sources[selector] = source
        artifacts = m.Infra.ProjectManagedArtifactsConfig(
            Ruff=m.Infra.ProjectRuffConfig(
                per_file_ignores={
                    pattern: tuple(sorted(rules))
                    for pattern, rules in sorted(ruff_ignores.items())
                }
            ),
            Mise=m.Infra.ProjectMiseConfig(tools=dict(sorted(mise_tools.items()))),
            Gitignore=m.Infra.ProjectGitignoreConfig(
                patterns=tuple(gitignore_patterns)
            ),
        )
        return r[m.Infra.ProjectManagedArtifactsResolution].ok(
            m.Infra.ProjectManagedArtifactsResolution(
                artifacts=artifacts,
                mise_tool_sources=dict(sorted(mise_sources.items())),
            )
        )

    @classmethod
    def compose_mise_toml(cls, project_dir: Path, rendered: str) -> p.Result[str]:
        """Snapshot project YAML and compose Mise from those exact bytes."""
        source_snapshot = cls.snapshot_config_sources(project_dir)
        if source_snapshot.failure:
            return r[str].from_failure(source_snapshot)
        return cls.compose_mise_toml_from_snapshot(source_snapshot.value, rendered)

    @classmethod
    def compose_mise_toml_from_snapshot(
        cls, source_snapshot: tuple[m.Cli.AtomicFileState, ...], rendered: str
    ) -> p.Result[str]:
        """Add local tools from one caller-owned immutable YAML snapshot."""
        resolved = cls.load_project_managed_artifacts_from_snapshot(source_snapshot)
        if resolved.failure:
            return r[str].fail(resolved.error or "project artifact load failed")
        local_tools = resolved.value.artifacts.Mise.tools
        if not local_tools:
            return r[str].ok(rendered)
        for selector in local_tools:
            selector_validation = cls.validate_mise_tool_selectors(
                (selector,), source=resolved.value.mise_tool_sources[selector]
            )
            if selector_validation.failure:
                return r[str].fail(
                    selector_validation.error
                    or "project Mise identity validation failed"
                )
        doc = u.Cli.toml_parse_text(rendered)
        if doc is None:
            return r[str].fail("canonical .mise.toml template is invalid")
        tools = u.Cli.toml_ensure_table(doc, "tools")
        for selector, tool in local_tools.items():
            if selector in tools:
                source = resolved.value.mise_tool_sources[selector]
                return r[str].fail(
                    "project Mise selector collides with fleet tool "
                    f"{selector!r}: global .mise.toml template and {source}"
                )
            tools[selector] = tool.version
        return r[str].ok(u.Cli.toml_dumps(doc))

    @classmethod
    def lock_platform_exclusions(
        cls, project_dir: Path
    ) -> p.Result[t.MappingKV[str, frozenset[str]]]:
        """Snapshot YAML and derive lock exclusions from those exact bytes."""
        source_snapshot = cls.snapshot_config_sources(project_dir)
        if source_snapshot.failure:
            return r[t.MappingKV[str, frozenset[str]]].from_failure(source_snapshot)
        return cls.lock_platform_exclusions_from_snapshot(source_snapshot.value)

    @classmethod
    def lock_platform_exclusions_from_snapshot(
        cls, source_snapshot: tuple[m.Cli.AtomicFileState, ...]
    ) -> p.Result[t.MappingKV[str, frozenset[str]]]:
        """Derive lock exclusions from one caller-owned immutable YAML snapshot."""
        resolved = cls.load_project_managed_artifacts_from_snapshot(source_snapshot)
        if resolved.failure:
            return r[t.MappingKV[str, frozenset[str]]].fail(
                resolved.error or "project artifact load failed"
            )
        fleet_platforms = frozenset(config.Infra.codegen.toolchain.mise_lock_platforms)
        # Absent platforms: the tool locks on every fleet platform. A declared
        # tuple (possibly empty, for backends without per-platform assets)
        # records exactly the platforms the lock may carry.
        exclusions = {
            selector: fleet_platforms - frozenset(tool.platforms)
            for selector, tool in resolved.value.artifacts.Mise.tools.items()
            if tool.platforms is not None
        }
        return r[t.MappingKV[str, frozenset[str]]].ok(exclusions)


__all__: tuple[str, ...] = ("FlextInfraUtilitiesProjectManagedArtifacts",)
