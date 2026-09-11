"""Modernize workspace pyproject.toml files to standardized format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, config, m, t, u
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.deps._modernizer_constraints import (
    FlextInfraPyprojectModernizerConstraintsMixin,
)
from flext_infra.deps._modernizer_document import (
    FlextInfraPyprojectModernizerDocumentMixin,
)
from flext_infra.deps._modernizer_payload import (
    FlextInfraPyprojectModernizerPayloadMixin,
)
from flext_infra.deps._modernizer_run import FlextInfraPyprojectModernizerRunMixin
from flext_infra.deps.extra_paths import FlextInfraExtraPathsManager
from flext_infra.deps.phases.ensure_pyright import FlextInfraEnsurePyrightConfigPhase

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraPyprojectModernizer(
    FlextInfraProjectSelectionServiceBase[bool],
    FlextInfraPyprojectModernizerConstraintsMixin,
    FlextInfraPyprojectModernizerPayloadMixin,
    FlextInfraPyprojectModernizerDocumentMixin,
    FlextInfraPyprojectModernizerRunMixin,
):
    """Modernize all workspace pyproject.toml files."""

    audit: Annotated[
        bool, m.Field(False, description="Audit pyproject changes without writing")
    ] = False
    skip_check: Annotated[
        bool, m.Field(alias="skip-check", description="Skip post-write validation")
    ] = False
    skip_comments: Annotated[
        bool, m.Field(alias="skip-comments", description="Skip managed comment updates")
    ] = False
    rewrite_constraints: Annotated[
        bool,
        m.Field(
            alias="rewrite-constraints",
            description="Rewrite dependency constraints from uv.lock",
        ),
    ] = False
    managed_artifacts: Annotated[
        m.Infra.ProjectManagedArtifactsResolution | None,
        m.Field(
            default=None,
            exclude=True,
            description="Caller-owned project ManagedArtifacts resolution",
        ),
    ] = None
    tomlsort_sort_first: t.StrSequence = m.Field(
        default_factory=lambda: config.Infra.tooling.tools.tomlsort.sort_first,
        exclude=True,
        description="Config-owned top-level TOML section order",
    )

    def conform_source(
        self,
        source: str,
        *,
        path: Path,
        format_source: bool = True,
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
        generated_python_roots: t.StrSequence = (),
        project_kind: str | None = None,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> p.Result[str]:
        """Return one canonical pyproject using the same phases as workspace apply.

        ``declared_python_dirs_are_complete`` says the caller enumerated EVERY
        Python root, so discovery must not widen the set. An atomic scaffold
        knows its future roots before they exist on disk; filesystem discovery
        would find none and silently produce a different fixed point than the
        post-write conformance pass.
        """
        payload_source = u.Cli.toml_mapping_from_text(source)
        if payload_source is None:
            return r[str].fail(f"invalid TOML: {path}")
        try:
            payload = t.Infra.MUTABLE_INFRA_MAPPING_ADAPTER.validate_python(
                payload_source
            )
            canonical_dev = t.Infra.STR_SEQ_ADAPTER.validate_python(
                u.Infra.canonical_dev_dependencies_from_payload(payload)
            )
        except c.ValidationError as exc:
            return r[str].fail_op("pyproject model validation", exc)
        state = m.Infra.PyprojectDocumentState(
            pyproject_path=path, original_rendered=source, payload=payload
        )
        # flext-j47u (codex): atomic scaffolds provide validated future roots;
        # existing repositories keep filesystem discovery through the empty default.
        changes = self._process_document_state(
            state,
            canonical_dev=canonical_dev,
            dry_run=True,
            skip_comments=False,
            format_source=format_source,
            declared_python_dirs=declared_python_dirs,
            declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            generated_python_roots=generated_python_roots,
            project_kind=project_kind,
            analysis_exclusions=analysis_exclusions,
        )
        if not state.rendered:
            return r[str].fail(
                changes[0] if changes else f"pyproject tooling render failed: {path}"
            )
        return r[str].ok(state.rendered)

    def resolve_tooling_context(
        self,
        *,
        project_name: t.NonEmptyStr,
        package_name: t.NonEmptyStr,
        path: Path,
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
        project_kind: str | None = None,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> p.Result[m.Infra.ToolingRuntimeContext]:
        """Resolve typed project/workspace values for the complete Jinja template."""
        # flext-j47u (codex): resolve values only; template retains the full structure.
        seed = u.Cli.toml_document()
        project = u.Cli.toml_table()
        project.add(c.Infra.NAME, project_name)
        seed.add(c.Infra.PROJECT, project)
        tool = u.Cli.toml_table()
        flext = u.Cli.toml_table()
        docs = u.Cli.toml_table()
        docs.add("package_name", package_name)
        flext.add("docs", docs)
        tool.add("flext", flext)
        seed.add(c.Infra.TOOL, tool)
        # NOTE(flext-p68a.5, agent codex): resolve from the declared future roots
        # so first generation and post-write conformance are the same fixed point.
        conformed = self.conform_source(
            u.Cli.toml_dumps(seed),
            path=path,
            format_source=False,
            declared_python_dirs=declared_python_dirs,
            declared_python_dirs_are_complete=declared_python_dirs_are_complete,
            project_kind=project_kind,
            analysis_exclusions=analysis_exclusions,
        )
        if conformed.failure:
            return r[m.Infra.ToolingRuntimeContext].fail(
                conformed.error or f"tooling resolution failed: {path}"
            )
        payload = u.Cli.toml_mapping_from_text(conformed.value)
        if payload is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution produced invalid TOML: {path}"
            )
        tooling = u.Cli.toml_mapping_child(payload, c.Infra.TOOL)
        if tooling is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution produced no [tool] table: {path}"
            )
        coverage = u.Cli.toml_mapping_path(tooling, ("coverage", "report"))
        deptry = u.Cli.toml_mapping_child(tooling, "deptry")
        mypy = u.Cli.toml_mapping_child(tooling, c.Infra.MYPY)
        pyrefly = u.Cli.toml_mapping_child(tooling, c.Infra.PYREFLY)
        pyright = u.Cli.toml_mapping_child(tooling, c.Infra.PYRIGHT)
        ruff = u.Cli.toml_mapping_child(tooling, c.Infra.RUFF)
        ruff_lint = (
            u.Cli.toml_mapping_child(ruff, c.Infra.LINT_SECTION)
            if ruff is not None
            else None
        )
        ruff_isort = (
            u.Cli.toml_mapping_path(ruff, ("lint", "isort"))
            if ruff is not None
            else None
        )
        if coverage is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted coverage.report: {path}"
            )
        if deptry is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted deptry: {path}"
            )
        if mypy is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted mypy: {path}"
            )
        if pyrefly is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted pyrefly: {path}"
            )
        if pyright is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted pyright: {path}"
            )
        if ruff is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted ruff: {path}"
            )
        if ruff_lint is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted ruff.lint: {path}"
            )
        if ruff_isort is None:
            return r[m.Infra.ToolingRuntimeContext].fail(
                f"tooling resolution omitted ruff.lint.isort: {path}"
            )
        scalar_keys = frozenset({
            c.Infra.EXCLUDE,
            c.Infra.IGNORE,
            c.Infra.INCLUDE,
            c.Infra.EXTRA_PATHS,
            "executionEnvironments",
            "venv",
            "venvPath",
        })
        raw_environments = u.Cli.json_as_sequence(pyright.get("executionEnvironments"))
        if declared_python_dirs:
            raw_environments = FlextInfraEnsurePyrightConfigPhase(
                config.Infra.tooling
            ).environment_payloads_for_dirs(declared_python_dirs)
        declared_pyrefly_includes = (
            FlextInfraExtraPathsManager.pyrefly_include_globs(declared_python_dirs)
            if declared_python_dirs
            else ()
        )
        # Seed for a project whose analyzer paths were never synced yet.
        #
        # The manager derives from directories that EXIST, so during scaffolding
        # -- before src/ is written -- it returns []. Writing that empty list
        # made the next plan re-derive ['src', '.'] once the tree existed, so
        # apply never reached its fixed point and no new project could be
        # generated. Prefer the DECLARED roots, which is exactly how the
        # ensure-pyrefly phase keeps pre-write scope identical to the first
        # post-write discovery without fabricating directories on disk.
        seed_manager = FlextInfraExtraPathsManager(repository_root=self.root)
        discovered_search = seed_manager.pyrefly_search_paths(
            project_dir=path.parent, is_root=True
        )
        discovered_extra = seed_manager.pyright_extra_paths(
            project_dir=path.parent, is_root=True
        )
        path_rules = config.Infra.tooling.tools.pyrefly.path_rules
        # A shared search path belongs to the project when the scaffold
        # declares it or the tree already has it — the same rule
        # `pyrefly_search_paths` applies after the write. Seeding every
        # configured path regardless left `scripts` in the render and out of
        # the sync, so apply could never reach a fixed point.
        project_dir = path.parent
        declared_roots = (
            (
                path_rules.source_dir,
                *(
                    shared
                    for shared in path_rules.project_shared_search_paths
                    if shared in declared_python_dirs or (project_dir / shared).is_dir()
                ),
                path_rules.project_root,
            )
            if path_rules.source_dir in declared_python_dirs
            else ()
        )
        # Why: partial disk discovery returns ('.',) before src/ exists, which is
        # truthy and blocked declared_roots ('src', '.'). Prefer declared roots
        # for search/mypy whenever scaffolding supplied them; pyright extras keep
        # discovery order (sorted {'.', 'src'}) so the first write matches sync.
        derived_search_path = declared_roots or discovered_search
        # mypy and pyrefly diverge here (cosmos-45hiv, 2026-08-31): mypy
        # enumerates each search-path root as a package root, so roots that
        # re-spell the same files (". " vs a rooted "scripts/") make it abort
        # with source-file-found-twice. pyrefly resolves first-match and needs
        # the extra roots. mypy keeps declared source roots; pyrefly keeps the
        # full derivation.
        derived_mypy_path = (
            tuple(root for root in declared_roots if root != ".")
            if declared_roots
            else derived_search_path
        )
        derived_extra_paths = discovered_extra or declared_roots
        resolved_project_kind = project_kind or "core"
        if project_kind is None and path.parent.resolve() != self.root.resolve():
            classified = self._classify_project(path.parent, payload=payload)
            if classified.failure:
                return r[m.Infra.ToolingRuntimeContext].fail(
                    classified.error or f"project classification failed: {path}"
                )
            resolved_project_kind = classified.value
        try:
            environments = self._tooling_pyright_environments(raw_environments)
            runtime = m.Infra.ToolingRuntimeContext.model_validate({
                "project_kind": resolved_project_kind,
                "coverage_fail_under": coverage.get("fail_under"),
                "first_party": ruff_isort.get("known-first-party"),
                # Absent analyzer-path keys fall back to the DERIVED value, not
                # to (). These keys are written by the analyzer-path sync, so a
                # project that has not run it yet -- a freshly scaffolded one, or
                # any isolated render fixture -- has them missing. A bare .get()
                # yields None, which the typed StrTuple contract rejects, so
                # conform could not render a new project at all. Defaulting to ()
                # instead writes an empty list that the NEXT plan immediately
                # re-derives as ['src', '.'], so apply never reaches its fixed
                # point. Seeding the derivation makes the first write final.
                # When scaffolding declares roots, conform_source may already
                # have written a partial mypy_path ('.' only). Prefer derivation
                # so the template matches post-write ExtraPaths sync.
                "mypy_path": (
                    derived_mypy_path
                    if declared_roots
                    else (mypy.get("mypy_path") or derived_mypy_path)
                ),
                "pyrefly_search_path": (
                    derived_search_path
                    if declared_roots
                    else (pyrefly.get(c.Infra.SEARCH_PATH) or derived_search_path)
                ),
                "pyrefly_project_includes": (
                    declared_pyrefly_includes
                    or pyrefly.get(c.Infra.PROJECT_INCLUDES, ())
                ),
                "pyright_exclude": pyright.get(c.Infra.EXCLUDE, ()),
                "pyright_ignore": pyright.get(c.Infra.IGNORE, ()),
                "pyright_include": (
                    declared_python_dirs or pyright.get(c.Infra.INCLUDE, ())
                ),
                "pyright_extra_paths": (
                    pyright.get(c.Infra.EXTRA_PATHS) or derived_extra_paths
                ),
                "pyright_settings": [
                    {"name": key, "value": value}
                    for key, value in sorted(pyright.items())
                    if key not in scalar_keys
                ],
                "pyright_execution_environments": environments,
                "ruff_src": ruff.get("src"),
                "ruff_exclude": ruff.get(c.Infra.EXCLUDE),
                "ruff_ignore": ruff_lint.get(c.Infra.IGNORE),
            })
        except c.ValidationError as exc:
            return r[m.Infra.ToolingRuntimeContext].fail_op(
                "tooling runtime context validation", exc
            )
        return r[m.Infra.ToolingRuntimeContext].ok(runtime)

    @staticmethod
    def _tooling_pyright_environments(
        raw_environments: t.SequenceOf[t.JsonValue],
    ) -> t.SequenceOf[m.Infra.ToolingPyrightEnvironment]:
        """Validate Pyright environments once into their canonical models."""
        # flext-j47u: nested tooling data crosses the TOML boundary as models, not dicts.
        environments: t.MutableSequenceOf[m.Infra.ToolingPyrightEnvironment] = []
        excluded = frozenset({"root", c.Infra.EXTRA_PATHS})
        for raw_environment in raw_environments:
            environment = t.Cli.JSON_MAPPING_ADAPTER.validate_python(raw_environment)
            environments.append(
                m.Infra.ToolingPyrightEnvironment.model_validate({
                    "root": environment.get("root"),
                    "extra_paths": environment.get(c.Infra.EXTRA_PATHS, ()),
                    "settings": tuple(
                        m.Infra.ToolingScalarSetting.model_validate({
                            "name": key,
                            "value": value,
                        })
                        for key, value in sorted(environment.items())
                        if key not in excluded
                    ),
                })
            )
        return tuple(environments)

    @override
    def execute(self) -> p.Result[bool]:
        """Execute pyproject modernization for the configured workspace."""
        exit_code = self.run()
        if exit_code != 0:
            return r[bool].fail("pyproject modernization failed")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraPyprojectModernizer"]
