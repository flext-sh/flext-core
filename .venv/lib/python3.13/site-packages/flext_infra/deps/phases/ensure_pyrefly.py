"""Phase: Ensure standard Pyrefly configuration for max-strict typing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m, t, u
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService

if TYPE_CHECKING:
    from flext_infra.deps.extra_paths import FlextInfraExtraPathsManager


class FlextInfraEnsurePyreflyConfigPhase:
    """Ensure standard Pyrefly configuration for max-strict typing."""

    def __init__(self, tool_config: m.Infra.ToolConfigDocument) -> None:
        """Store tool configuration used when enforcing pyrefly project settings."""
        self._tool_config = tool_config

    def _phase(
        self,
        *,
        is_root: bool,
        project_dir: Path | None = None,
        paths_manager: FlextInfraExtraPathsManager | None = None,
        stale_error_keys: t.StrSequence = (),
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
    ) -> m.Infra.Deps.Toml.PhaseConfig:
        """Build the canonical pyrefly phase definition."""
        pyrefly_rules = self._tool_config.tools.pyrefly
        if project_dir is not None and paths_manager is not None:
            expected_search = paths_manager.pyrefly_search_paths(
                project_dir=project_dir, is_root=is_root
            )
            expected_includes = paths_manager.pyrefly_project_includes(
                project_dir=project_dir, is_root=is_root
            )
        else:
            expected_search = [c.Infra.DEFAULT_SRC_DIR]
            expected_includes = [f"{c.Infra.DEFAULT_SRC_DIR}/**/*.py*"]
        # flext-j47u (codex): keep pre-write Pyrefly scope identical to the first
        # post-write discovery without fabricating directories on disk.
        if declared_python_dirs_are_complete:
            declared_import_roots = (
                (pyrefly_rules.path_rules.source_dir,)
                if pyrefly_rules.path_rules.source_dir in declared_python_dirs
                else ()
            )
            # Why (ai-hub-qwoc, fleet-wide fix): sorted({...}) places "."
            # before "src" (ASCII '.' < 's'), so pyrefly resolves every
            # module twice (ai_hub.X via src AND src.ai_hub.X via "."),
            # producing phantom bad-argument-type errors. The declared
            # source import root must stay first; everything else (typically
            # ".", for tests.* resolution) sorts after it.
            merged_search = {
                *expected_search,
                *pyrefly_rules.path_rules.project_shared_search_paths,
                pyrefly_rules.path_rules.project_root,
            }
            if not declared_import_roots:
                merged_search.discard(pyrefly_rules.path_rules.source_dir)
            merged_search.difference_update(declared_import_roots)
            expected_search = [*declared_import_roots, *sorted(merged_search)]
            # NOTE (multi-agent, flext-wkii.17.9.2.1): analysis roots belong in
            # project-includes; only import roots belong in search-path.
            expected_includes = tuple(
                f"{directory}/**/*.py*" for directory in declared_python_dirs
            )
        error_values: t.SequenceOf[tuple[str, t.JsonValue]] = tuple(
            (error_rule, "error")
            for error_rule in self._tool_config.tools.pyrefly.strict_errors
        )
        phase_builder = (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("pyrefly")
            .table(c.Infra.PYREFLY)
            .value(c.Infra.PYTHON_VERSION_HYPHEN, pyrefly_rules.python_version)
            .deprecated("python-interpreter-path")
            .deprecated("disable-search-path-heuristics")
            .deprecated("fallback-python-interpreter-name")
            # Interpreter discovery resolves PEP 660 editable sibling packages.
            .deprecated("site-package-path")
            .deprecated("skip-interpreter-query")
            .deprecated("ignore-errors-in-generated-code")
            # sort=False: search-path order is semantic (see comment above);
            # the default sort=True would silently re-alphabetize "." before
            # "src" here at TOML-emit time even after ordering it correctly.
            .list(c.Infra.SEARCH_PATH, expected_search, sort=False)
        )
        if declared_python_dirs_are_complete and not expected_includes:
            phase_builder = phase_builder.value(c.Infra.PROJECT_INCLUDES, [])
        else:
            phase_builder = phase_builder.list(
                c.Infra.PROJECT_INCLUDES, expected_includes
            )
        return (
            phase_builder
            .value(
                "disable-project-excludes-heuristics",
                pyrefly_rules.disable_project_excludes_heuristics,
            )
            .value("use-ignore-files", pyrefly_rules.use_ignore_files)
            .list(
                c.Infra.PROJECT_EXCLUDES,
                sorted(set(pyrefly_rules.project_exclude_globs)),
            )
            .nested("errors", values=error_values, deprecated_keys=stale_error_keys)
            .build()
        )

    def _configured_error_keys(self) -> frozenset[str]:
        """Return pyrefly error keys governed by the canonical tool config."""
        return frozenset(self._tool_config.tools.pyrefly.strict_errors)

    def apply(
        self,
        doc: t.Cli.TomlDocument,
        *,
        is_root: bool,
        project_dir: Path | None = None,
        paths_manager: FlextInfraExtraPathsManager | None = None,
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
    ) -> t.StrSequence:
        """Apply canonical pyrefly table values, paths, and strict error toggles."""
        configured_error_keys = self._configured_error_keys()
        errors_table = u.Cli.toml_table_path(
            doc, (c.Infra.TOOL, c.Infra.PYREFLY, "errors")
        )
        stale_error_keys = (
            tuple(key for key in errors_table if key not in configured_error_keys)
            if errors_table is not None
            else ()
        )
        return FlextInfraTomlPhaseService.apply_phases(
            doc,
            self._phase(
                is_root=is_root,
                project_dir=project_dir,
                paths_manager=paths_manager,
                stale_error_keys=stale_error_keys,
                declared_python_dirs=declared_python_dirs,
                declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            ),
        )

    def apply_payload(
        self,
        payload: t.MutableJsonMapping,
        *,
        is_root: bool,
        project_dir: Path | None = None,
        paths_manager: FlextInfraExtraPathsManager | None = None,
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
    ) -> t.StrSequence:
        """Apply canonical pyrefly settings to one normalized payload."""
        configured_error_keys = self._configured_error_keys()
        errors_table = u.Cli.toml_mapping_path(
            payload, (c.Infra.TOOL, c.Infra.PYREFLY, "errors")
        )
        stale_error_keys = (
            tuple(key for key in errors_table if key not in configured_error_keys)
            if errors_table is not None
            else ()
        )
        return FlextInfraTomlPhaseService.apply_payload_phases(
            payload,
            self._phase(
                is_root=is_root,
                project_dir=project_dir,
                paths_manager=paths_manager,
                stale_error_keys=stale_error_keys,
                declared_python_dirs=declared_python_dirs,
                declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            ),
        )


__all__: list[str] = ["FlextInfraEnsurePyreflyConfigPhase"]
