"""Run orchestration mixin for pyproject modernization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, m, t, u

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from flext_infra import p


class FlextInfraPyprojectModernizerRunMixin:
    """Drive file iteration, modernization, and post-write build checks."""

    if TYPE_CHECKING:
        audit: bool
        check_only: bool
        skip_check: bool
        skip_comments: bool
        rewrite_constraints: bool

        @property
        def root(self) -> Path: ...

        @property
        def effective_dry_run(self) -> bool: ...

        @property
        def project_names(self) -> t.StrSequence | None: ...

        def _read_document_state(
            self, path: Path
        ) -> p.Result[m.Infra.PyprojectDocumentState]: ...

        def _process_document_state(
            self,
            state: m.Infra.PyprojectDocumentState,
            *,
            canonical_dev: t.StrSequence,
            dry_run: bool,
            skip_comments: bool,
            rewrite_constraints: bool = False,
            locked_versions: t.MappingKV[str, str] | None = None,
            internal_names: t.StrSequence = (),
            declared_python_dirs: t.StrSequence = (),
            analysis_exclusions: t.StrSequence | None = None,
        ) -> t.StrSequence: ...

    def process_file(
        self,
        path: Path,
        *,
        canonical_dev: t.StrSequence,
        dry_run: bool,
        skip_comments: bool,
    ) -> t.StrSequence:
        """Process one pyproject.toml file and collect changes."""
        document_state_result = self._read_document_state(path)
        if document_state_result.failure:
            return ["invalid TOML"]
        return self._process_document_state(
            document_state_result.value,
            canonical_dev=canonical_dev,
            dry_run=dry_run,
            skip_comments=skip_comments,
            rewrite_constraints=False,
        )

    def run(self) -> int:
        """Run pyproject modernization for the workspace."""
        check_mode = self.audit or self.check_only
        dry_run = check_mode or self.effective_dry_run
        project_names = list(self.project_names or [])
        # Modernization writes only the requested repository root and its
        # declared declared_repositories, never siblings.
        include_root = not project_names or "." in project_names
        selected_names = (
            [name for name in project_names if name != "."]
            if project_names
            else list(u.Infra.workspace_project_paths(self.root))
        )
        configured_declared_paths = {
            declared_repository_name: self.root / declared_repository_name
            for declared_repository_name in u.Infra.workspace_project_paths(self.root)
        }
        resolved_root = self.root.resolve()
        outside_declared_repository_names = [
            declared_repository_name
            for declared_repository_name, declared_path in configured_declared_paths.items()
            if not declared_path.resolve().is_relative_to(resolved_root)
        ]
        if outside_declared_repository_names:
            u.Cli.error(
                "workspace declared_repositories outside root: "
                f"{', '.join(sorted(outside_declared_repository_names))}"
            )
            return 2
        basename_aliases: dict[str, list[Path]] = {}
        declared_name_aliases: dict[str, list[Path]] = {}
        for configured_path in configured_declared_paths.values():
            basename_aliases.setdefault(configured_path.name, []).append(
                configured_path
            )
            state_result = self._read_document_state(
                configured_path / c.Infra.PYPROJECT_FILENAME
            )
            if state_result.failure:
                continue
            state = state_result.value
            try:
                declared_name = u.Infra.project_name_from_payload(
                    state.pyproject_path, state.payload
                )
            except c.EXC_TYPE_VALIDATION:
                continue
            declared_name_aliases.setdefault(declared_name, []).append(configured_path)
        project_paths: t.MutableSequenceOf[Path] = []
        missing_names: t.MutableSequenceOf[str] = []
        ambiguous_names: t.MutableSequenceOf[str] = []
        for project_name in selected_names:
            project_path = configured_declared_paths.get(project_name)
            if project_path is None:
                alias_matches: t.MutableSequenceOf[Path] = []
                for alias_path in (
                    *basename_aliases.get(project_name, []),
                    *declared_name_aliases.get(project_name, []),
                ):
                    if alias_path not in alias_matches:
                        alias_matches.append(alias_path)
                if len(alias_matches) > 1:
                    ambiguous_names.append(project_name)
                    continue
                project_path = alias_matches[0] if alias_matches else None
            if project_path is None or not project_path.is_dir():
                missing_names.append(project_name)
                continue
            project_paths.append(project_path)
        if ambiguous_names:
            u.Cli.error(f"ambiguous projects: {', '.join(sorted(ambiguous_names))}")
            return 2
        if missing_names:
            u.Cli.error(f"unknown projects: {', '.join(sorted(missing_names))}")
            return 2
        files_result = u.Infra.find_all_pyproject_files(
            self.root,
            skip_dirs=c.Infra.PYPROJECT_SKIP_DIRS,
            project_paths=project_paths,
        )
        files: t.SequenceOf[Path] = (
            [] if files_result.failure else sorted(files_result.unwrap())
        )
        root_pyproject_path = self.root / c.Infra.PYPROJECT_FILENAME
        if (
            include_root
            and root_pyproject_path.is_file()
            and root_pyproject_path not in files
        ):
            files = sorted([root_pyproject_path, *files])
        root_state_result = self._read_document_state(root_pyproject_path)
        if root_state_result.failure:
            return 2
        root_state = root_state_result.value
        root_project_name: str | None = None
        if include_root or self.rewrite_constraints:
            try:
                root_project_name = u.Infra.project_name_from_payload(
                    root_state.pyproject_path, root_state.payload
                )
            except c.EXC_TYPE_VALIDATION as exc:
                u.Cli.error(str(exc))
                return 2
        canonical_dev: t.StrSequence = t.Infra.STR_SEQ_ADAPTER.validate_python(
            u.Infra.canonical_dev_dependencies_from_payload(root_state.payload)
        )
        locked_versions: t.MappingKV[str, str] = {}
        internal_names: t.StrSequence = ()
        if self.rewrite_constraints:
            if root_project_name is None:
                u.Cli.error("root project name required for constraint rewriting")
                return 2
            lock_path = self.root / c.Infra.UV_LOCK_FILENAME
            locked_versions = u.Infra.locked_dependency_versions(lock_path)
            if not locked_versions:
                u.Cli.error(
                    f"missing or invalid {c.Infra.UV_LOCK_FILENAME} at {lock_path}"
                )
                return 2
            internal_names = tuple(
                sorted(
                    set(u.Infra.workspace_project_paths(self.root))
                    | {root_project_name}
                )
            )
        violations: MutableMapping[str, t.StrSequence] = {}
        document_states: t.MutableSequenceOf[m.Infra.PyprojectDocumentState] = []
        invalid_paths: t.MutableSequenceOf[Path] = []
        total = 0
        for file_path in files:
            document_state_result = (
                r[m.Infra.PyprojectDocumentState].ok(root_state)
                if file_path.resolve() == root_state.pyproject_path.resolve()
                else self._read_document_state(file_path)
            )
            if document_state_result.failure:
                invalid_paths.append(file_path)
                changes = ["invalid TOML"]
            else:
                document_state = document_state_result.value
                document_states.append(document_state)
                changes = self._process_document_state(
                    document_state,
                    canonical_dev=canonical_dev,
                    dry_run=dry_run,
                    skip_comments=self.skip_comments,
                    rewrite_constraints=self.rewrite_constraints,
                    locked_versions=locked_versions,
                    internal_names=internal_names,
                )
            if not changes:
                continue
            resolved_file_path = file_path.resolve()
            resolved_root = self.root.resolve()
            rel = (
                str(resolved_file_path.relative_to(resolved_root))
                if resolved_file_path.is_relative_to(resolved_root)
                else str(resolved_file_path)
            )
            violations[rel] = changes
            total += len(changes)
        if violations:
            for rel_path, changes in violations.items():
                u.Cli.info(f"{rel_path}:")
                for change in changes:
                    u.Cli.info(f"  - {change}")
            u.Cli.info(f"Total: {total} change(s) across {len(violations)} file(s)")
            if dry_run:
                u.Cli.info("(dry-run — no files modified)")
        if check_mode and total > 0:
            return 1
        if not dry_run and (not self.skip_check):
            return self._run_build_check(document_states, invalid_paths=invalid_paths)
        return 0

    def _run_build_check(
        self,
        document_states: t.SequenceOf[m.Infra.PyprojectDocumentState],
        *,
        invalid_paths: t.SequenceOf[Path] = (),
    ) -> int:
        """Validate pyproject.toml files have hatchling build backend."""
        has_warning = False
        for invalid_path in invalid_paths:
            u.Cli.info(f"{invalid_path}: invalid TOML")
            has_warning = True
        for document_state in document_states:
            path = document_state.pyproject_path
            build_sys = u.Cli.toml_mapping_child(document_state.payload, "build-system")
            if build_sys is None:
                u.Cli.info(f"{path}: missing [build-system]")
                has_warning = True
                continue
            backend = u.norm_str(str(build_sys.get("build-backend", "")))
            if backend != "hatchling.build":
                u.Cli.info(f"{path}: expected hatchling.build, got {backend}")
                has_warning = True
        return 1 if has_warning else 0


__all__: list[str] = ["FlextInfraPyprojectModernizerRunMixin"]
