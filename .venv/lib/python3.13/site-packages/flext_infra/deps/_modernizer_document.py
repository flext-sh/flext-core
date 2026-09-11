"""Document-state processing mixin for pyproject modernization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, config, m, t, u
from flext_infra.deps.extra_paths import FlextInfraExtraPathsManager
from flext_infra.deps.phases.consolidate_groups import FlextInfraConsolidateGroupsPhase
from flext_infra.deps.phases.ensure_coverage import FlextInfraEnsureCoverageConfigPhase
from flext_infra.deps.phases.ensure_formatting import (
    FlextInfraEnsureFormattingToolingPhase,
)
from flext_infra.deps.phases.ensure_mypy import FlextInfraEnsureMypyConfigPhase
from flext_infra.deps.phases.ensure_namespace import (
    FlextInfraEnsureNamespaceToolingPhase,
)
from flext_infra.deps.phases.ensure_packaging import FlextInfraEnsurePackagingPhase
from flext_infra.deps.phases.ensure_pydantic_mypy import (
    FlextInfraEnsurePydanticMypyConfigPhase,
)
from flext_infra.deps.phases.ensure_pyrefly import FlextInfraEnsurePyreflyConfigPhase
from flext_infra.deps.phases.ensure_pyright import FlextInfraEnsurePyrightConfigPhase
from flext_infra.deps.phases.ensure_pytest import FlextInfraEnsurePytestConfigPhase
from flext_infra.deps.phases.ensure_ruff import FlextInfraEnsureRuffConfigPhase
from flext_infra.deps.phases.ensure_vulture import FlextInfraEnsureVultureConfigPhase
from flext_infra.deps.phases.inject_comments import FlextInfraInjectCommentsPhase
from flext_infra.refactor.project_classifier import FlextInfraProjectClassifier

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from flext_infra import p


class FlextInfraPyprojectModernizerDocumentMixin:
    """Read, classify, and process one parsed pyproject document state."""

    if TYPE_CHECKING:
        # Members provided by the composed dependency modernizer.
        _rewrite_dependency_constraints_payload: Callable[..., t.StrSequence]
        managed_artifacts: m.Infra.ProjectManagedArtifactsResolution | None

        @property
        def root(self) -> Path: ...

        def _ensure_build_system_payload(
            self, payload: t.MutableJsonMapping
        ) -> t.StrSequence: ...

        def _remove_empty_poetry_groups_payload(
            self, payload: t.MutableJsonMapping
        ) -> t.StrSequence: ...

        # Declared as a plain attribute, not a property: the concrete owner
        # (FlextInfraPyprojectModernizer) supplies it as a Pydantic field whose
        # default comes from the tomlsort SSOT. A property here would make the
        # field an incompatible override of a read-only descriptor.
        tomlsort_sort_first: t.StrSequence

        def _reorder_document_inplace(
            self,
            doc: t.Cli.TomlDocument,
            *,
            preferred_first: t.StrSequence | None = None,
        ) -> None: ...

    def _classify_project(
        self, project_dir: Path, *, payload: t.JsonMapping | None = None
    ) -> p.Result[str]:
        """Classify project kind for pyright/coverage settings selection."""
        classifier = FlextInfraProjectClassifier(project_dir, pyproject_payload=payload)
        return r[str].ok(classifier.classify().project_kind)

    def _read_document_state(
        self, path: Path
    ) -> p.Result[m.Infra.PyprojectDocumentState]:
        """Read one pyproject once and keep one validated plain payload state."""
        read = u.Cli.files_read_text(path)
        if read.failure:
            return r[m.Infra.PyprojectDocumentState].fail(
                read.error or f"failed to read {path}"
            )
        original_rendered = read.value
        payload_source = u.Cli.toml_mapping_from_text(original_rendered)
        if payload_source is None:
            return r[m.Infra.PyprojectDocumentState].fail(f"invalid TOML: {path}")
        try:
            payload = t.Infra.MUTABLE_INFRA_MAPPING_ADAPTER.validate_python(
                payload_source
            )
        except c.ValidationError as exc:
            return r[m.Infra.PyprojectDocumentState].fail_op(
                "TOML payload validation", exc
            )
        return r[m.Infra.PyprojectDocumentState].ok(
            m.Infra.PyprojectDocumentState(
                pyproject_path=path,
                original_rendered=original_rendered,
                payload=payload,
            )
        )

    def _format_rendered_pyproject(self, path: Path, rendered: str) -> p.Result[str]:
        """Format rendered pyproject TOML with the workspace Taplo contract."""
        # Why (flext-6itas.4): restored delegation to the shared cached facade
        # dropped by merge 0d4d07b33; the inline duplicate reimplemented the
        # same mise/taplo invocation without the cache or relative-path fix.
        return u.Infra.format_toml_source(
            rendered,
            path=path,
            toolchain_root=self.root,
            taplo_version=config.Infra.codegen.toolchain.taplo_version,
        )

    def _process_document_state(
        self,
        state: m.Infra.PyprojectDocumentState,
        *,
        canonical_dev: t.StrSequence,
        dry_run: bool,
        skip_comments: bool,
        format_source: bool = True,
        rewrite_constraints: bool = False,
        locked_versions: t.MappingKV[str, str] | None = None,
        internal_names: t.StrSequence = (),
        declared_python_dirs: t.StrSequence = (),
        declared_python_dirs_are_complete: bool = False,
        generated_python_roots: t.StrSequence = (),
        project_kind: str | None = None,
        analysis_exclusions: t.StrSequence | None = None,
    ) -> t.StrSequence:
        """Process one parsed pyproject state and collect changes."""
        path = state.pyproject_path
        original_rendered = state.original_rendered
        payload = state.payload
        is_root = path.parent.resolve() == self.root.resolve()
        # Scaffold (pre-write) contexts have no on-disk project
        # root yet. Derive pyright/pyrefly configuration from declared roots only;
        # disk discovery converges on the first post-write conformance pass.
        project_root_exists = path.is_file()
        effective_project_dir = path.parent if project_root_exists else None
        effective_repository_root = self.root if project_root_exists else None
        paths_manager = FlextInfraExtraPathsManager(
            repository_root=self.root, generated_python_roots=generated_python_roots
        )
        effective_paths_manager = paths_manager if project_root_exists else None
        resolved_project_kind: str = project_kind or "core"
        if project_kind is None and not is_root:
            kind_result = self._classify_project(path.parent, payload=payload)
            if kind_result.success:
                resolved_project_kind = kind_result.value
        # Declared roots are topology facts only during atomic
        # creation; normal modernization still derives productive roots on disk.
        # Why (flext-6itas.4): restored after merge 0d4d07b33 dropped this gate
        # while ensure_pyrefly.py kept requiring it (declared_python_dirs_are_complete
        # still guards its replace-model; pyright moved to an additive-union model
        # in dd4c4b53f and no longer needs this flag).
        declared_roots_are_usable = (
            declared_python_dirs_are_complete or not project_root_exists
        )
        changes: t.MutableSequenceOf[str] = []
        changes.extend(self._ensure_build_system_payload(payload))
        changes.extend(self._remove_empty_poetry_groups_payload(payload))
        if rewrite_constraints:
            changes.extend(
                self._rewrite_dependency_constraints_payload(
                    payload,
                    locked_versions=locked_versions or {},
                    internal_names=internal_names,
                )
            )
        changes.extend(
            FlextInfraConsolidateGroupsPhase().apply_payload(payload, canonical_dev)
        )
        changes.extend(
            FlextInfraEnsurePytestConfigPhase(config.Infra.tooling).apply_payload(
                payload
            )
        )
        # Pyrefly derives its include globs from the canonical
        # Pyright roots, so resolve Pyright first and converge in one pass.
        changes.extend(
            FlextInfraEnsurePyrightConfigPhase(config.Infra.tooling).apply_payload(
                payload,
                is_root=is_root,
                repository_root=effective_repository_root,
                project_dir=effective_project_dir,
                project_kind=resolved_project_kind,
                paths_manager=effective_paths_manager,
                declared_python_dirs=declared_python_dirs,
                declared_python_dirs_are_complete=declared_python_dirs_are_complete,
                analysis_exclusions=analysis_exclusions,
            )
        )
        changes.extend(
            FlextInfraEnsurePyreflyConfigPhase(config.Infra.tooling).apply_payload(
                payload,
                is_root=is_root,
                project_dir=effective_project_dir,
                paths_manager=effective_paths_manager,
                declared_python_dirs=declared_python_dirs,
                declared_python_dirs_are_complete=declared_roots_are_usable,
            )
        )
        changes.extend(
            FlextInfraEnsureMypyConfigPhase(config.Infra.tooling).apply_payload(payload)
        )
        changes.extend(
            FlextInfraEnsurePydanticMypyConfigPhase(config.Infra.tooling).apply_payload(
                payload
            )
        )
        changes.extend(
            FlextInfraEnsureFormattingToolingPhase(config.Infra.tooling).apply_payload(
                payload
            )
        )
        changes.extend(
            FlextInfraEnsureNamespaceToolingPhase().apply_payload(payload, path=path)
        )
        changes.extend(
            FlextInfraEnsureRuffConfigPhase(
                config.Infra.tooling, self.managed_artifacts
            ).apply_payload(payload, path=path)
        )
        changes.extend(
            FlextInfraEnsurePackagingPhase(config.Infra.tooling).apply_payload(
                payload, path=path, is_root=is_root
            )
        )
        # Existing projects consume the same Vulture SSOT as scaffolds.
        changes.extend(
            FlextInfraEnsureVultureConfigPhase(config.Infra.tooling).apply_payload(
                payload
            )
        )
        changes.extend(
            FlextInfraEnsureCoverageConfigPhase(config.Infra.tooling).apply_payload(
                payload, project_kind=resolved_project_kind
            )
        )
        if effective_paths_manager is not None:
            changes.extend(
                effective_paths_manager.sync_payload(
                    payload, project_dir=path.parent, is_root=is_root
                )
            )
        doc: t.Cli.TomlDocument = u.Cli.toml_document_from_mapping(payload)
        self._reorder_document_inplace(doc, preferred_first=self.tomlsort_sort_first)
        state.payload = payload
        rendered = doc.as_string()
        if not skip_comments:
            rendered, comment_changes = FlextInfraInjectCommentsPhase().apply(rendered)
            changes.extend(comment_changes)
        if format_source:
            formatted_result = self._format_rendered_pyproject(path, rendered)
            if formatted_result.failure:
                return [formatted_result.error or "taplo format failed"]
            rendered = formatted_result.value
        normalized_original = original_rendered.rstrip() + "\n"
        normalized_rendered = rendered.rstrip() + "\n"
        state.rendered = normalized_rendered
        if normalized_rendered == normalized_original:
            return ()
        if not dry_run:
            # Persist the same normalized value used for change detection.
            u.write_file(path, normalized_rendered, encoding=c.Cli.ENCODING_DEFAULT)
        return changes


__all__: list[str] = ["FlextInfraPyprojectModernizerDocumentMixin"]
