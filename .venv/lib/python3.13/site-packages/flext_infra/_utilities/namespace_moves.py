"""Move and compatibility rewrites for namespace refactors."""

from __future__ import annotations

import tokenize
from collections import defaultdict
from io import StringIO
from pathlib import Path

from flext_cli import u
from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.namespace_common import (
    FlextInfraUtilitiesRefactorNamespaceCommon,
)
from flext_infra._utilities.protected_edit import FlextInfraUtilitiesProtectedEdit
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra._utilities.rope_imports import FlextInfraUtilitiesRopeImports
from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime
from flext_infra._utilities.rope_source import FlextInfraUtilitiesRopeSource
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.transformers import _header
from flext_infra.transformers.project_alias_migrator import (
    FlextInfraRefactorProjectAliasMigrator,
)
from flext_infra.typings import t


class FlextInfraUtilitiesRefactorNamespaceMoves:
    """Helpers for block moves and compatibility-alias rewrites."""

    @classmethod
    def rewrite_import_violations(
        cls, *, py_files: t.SequenceOf[Path], project_package: str
    ) -> None:
        """Rewrite import violations."""
        if not py_files:
            return
        with FlextInfraUtilitiesRopeCore.open_project(
            FlextInfraUtilitiesRefactorNamespaceCommon.shared_repository_root(
                py_files=py_files
            )
        ) as rope_project:
            for file_path in py_files:
                if file_path.name == c.Infra.INIT_PY:
                    continue
                project_root = FlextInfraUtilitiesDiscovery.project_root(file_path)
                if project_root is not None and (
                    FlextInfraUtilitiesDiscovery.contextual_runtime_alias_sources(
                        project_root=project_root, file_path=file_path
                    )
                ):
                    continue
                source = file_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
                if FlextInfraUtilitiesRopeSource.looks_like_facade_file(
                    file_path=file_path, source=source
                ):
                    continue
                resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, file_path
                )
                if resource is None:
                    continue
                rewritten = (
                    FlextInfraUtilitiesRopeImports.collapse_submodule_alias_imports(
                        rope_project,
                        resource,
                        package_name=project_package,
                        aliases=tuple(
                            sorted(u.runtime_alias_names(c.Infra.PKG_INFRA_UNDERSCORE))
                        ),
                        apply=True,
                    )
                )
                if rewritten is None:
                    continue
                cleanup_result = FlextInfraUtilitiesRopeImports.normalize_imports(
                    rope_project,
                    file_paths=(file_path,),
                    preserve_canonical_aliases=True,
                )
                if cleanup_result.failure:
                    msg = cleanup_result.error or "rope import cleanup failed"
                    raise RuntimeError(msg)

    @staticmethod
    def rewrite_namespace_source_violations(
        *,
        violations: t.SequenceOf[m.Infra.NamespaceSourceViolation],
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation],
        gates: t.StrSequence | None = None,
    ) -> None:
        """Rewrite runtime aliases imported from a foreign FLEXT package source."""
        _ = parse_failures, gates
        grouped: t.MappingKV[Path, t.MutableMappingKV[tuple[str, str], set[str]]] = (
            defaultdict(lambda: defaultdict(set))
        )
        for violation in violations:
            grouped[Path(violation.file)][
                violation.current_source, violation.correct_source
            ].add(violation.alias)
        if not grouped:
            return
        repository_root = (
            FlextInfraUtilitiesRefactorNamespaceCommon.shared_repository_root(
                py_files=tuple(grouped)
            )
        )
        with FlextInfraUtilitiesRopeCore.open_project(repository_root) as rope_project:
            for file_path, moves in grouped.items():
                resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, file_path
                )
                if resource is None:
                    msg = (
                        "rope resource unavailable for namespace source rewrite: "
                        f"{file_path}"
                    )
                    raise RuntimeError(msg)
                changed = False
                for (current_source, correct_source), aliases in sorted(moves.items()):
                    updated = (
                        FlextInfraUtilitiesRopeImports.relocate_from_import_aliases(
                            rope_project,
                            resource,
                            source_module=current_source,
                            target_module=correct_source,
                            aliases=tuple(sorted(aliases)),
                            apply=True,
                        )
                    )
                    changed = changed or updated is not None
                if not changed:
                    continue
                cleanup_result = FlextInfraUtilitiesRopeImports.normalize_imports(
                    rope_project,
                    file_paths=(file_path,),
                    preserve_canonical_aliases=True,
                )
                if cleanup_result.failure:
                    msg = cleanup_result.error or "rope import cleanup failed"
                    raise RuntimeError(msg)

    @classmethod
    def rewrite_runtime_alias_violations(
        cls, *, py_files: t.SequenceOf[Path], gates: t.StrSequence | None = None
    ) -> None:
        """Rewrite runtime alias violations."""
        if not py_files:
            return
        repository_root = (
            FlextInfraUtilitiesRefactorNamespaceCommon.shared_repository_root(
                py_files=py_files
            )
        )
        with FlextInfraUtilitiesRopeCore.open_project(repository_root) as rope_project:
            for file_path in py_files:
                expected = c.Infra.NAMESPACE_FAMILY_EXPECTED_ALIAS.get(file_path.name)
                if expected is None:
                    continue
                alias_name, expected_suffix = expected
                resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, file_path
                )
                if resource is None:
                    continue
                class_candidates = [
                    info.name
                    for info in FlextInfraUtilitiesRopeAnalysis.get_class_info(
                        rope_project, resource
                    )
                    if info.name.endswith(expected_suffix)
                ]
                if len(class_candidates) != 1:
                    continue
                target_class = class_candidates[0]
                lines = file_path.read_text(
                    encoding=c.Cli.ENCODING_DEFAULT
                ).splitlines()
                kept = [
                    line
                    for line in lines
                    if not line.strip().startswith(f"{alias_name} = ")
                ]
                rewritten = (
                    "\n".join(kept).rstrip() + f"\n\n{alias_name} = {target_class}\n"
                )
                original_source = file_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
                if rewritten == original_source:
                    continue
                _ = FlextInfraUtilitiesProtectedEdit.protected_source_write(
                    file_path,
                    request=m.Infra.ProtectedSourceWriteRequest(
                        workspace=repository_root,
                        updated_source=rewritten,
                        keep_backup=True,
                        gates=gates,
                    ),
                )

    @staticmethod
    def rewrite_manual_protocol_violations(
        *,
        project_root: Path,
        py_files: t.SequenceOf[Path],
        violations: t.SequenceOf[m.Infra.ManualProtocolViolation],
        gates: t.StrSequence | None = None,
    ) -> None:
        """Rewrite manual protocol violations."""
        grouped: t.MappingKV[Path, t.Infra.StrSet] = defaultdict(set)
        for violation in violations:
            grouped[Path(violation.file)].add(violation.name)
        protocol_moves: t.MutableSequenceOf[
            t.Triple[Path, Path, t.VariadicTuple[str]]
        ] = []
        for source_file, protocol_names in grouped.items():
            move = FlextInfraUtilitiesRefactorNamespaceMoves._move_named_blocks(
                project_root=project_root,
                source_file=source_file,
                target_filename=c.Infra.PROTOCOLS_PY,
                names=protocol_names,
                header_prefix="class ",
                gates=gates,
            )
            if move is not None:
                protocol_moves.append(move)
        if protocol_moves:
            FlextInfraUtilitiesRefactorNamespaceMoves._rewrite_moved_imports(
                project_root=project_root, py_files=py_files, moves=protocol_moves
            )

    @staticmethod
    def rewrite_manual_typing_alias_violations(
        *,
        project_root: Path,
        violations: t.SequenceOf[m.Infra.ManualTypingAliasViolation],
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation],
        gates: t.StrSequence | None = None,
    ) -> None:
        """Rewrite manual typing alias violations."""
        _ = parse_failures
        grouped: t.MappingKV[Path, t.Infra.StrSet] = defaultdict(set)
        for violation in violations:
            grouped[Path(violation.file)].add(violation.name)
        for source_file, alias_names in grouped.items():
            FlextInfraUtilitiesRefactorNamespaceMoves._move_typing_alias_lines(
                project_root=project_root,
                source_file=source_file,
                alias_names=alias_names,
                gates=gates,
            )

    @staticmethod
    def rewrite_loose_object_violations(
        *,
        project_root: Path,
        violations: t.SequenceOf[m.Infra.LooseObjectViolation],
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation],
        gates: t.StrSequence | None = None,
    ) -> None:
        """Rewrite loose namespace objects whose canonical mover is deterministic."""
        _ = parse_failures
        typing_grouped: t.MappingKV[Path, t.Infra.StrSet] = defaultdict(set)
        for violation in violations:
            if violation.kind in {"typealias", "typevar"}:
                typing_grouped[Path(violation.file)].add(violation.name)
        for source_file, alias_names in typing_grouped.items():
            FlextInfraUtilitiesRefactorNamespaceMoves._move_typing_alias_lines(
                project_root=project_root,
                source_file=source_file,
                alias_names=alias_names,
                gates=gates,
            )

    @staticmethod
    def rewrite_compatibility_alias_violations(
        *,
        violations: t.SequenceOf[m.Infra.CompatibilityAliasViolation],
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation],
        gates: t.StrSequence | None = None,
    ) -> None:
        """Rewrite compatibility alias violations."""
        _ = parse_failures
        assignment_grouped: t.MappingKV[Path, t.MutableStrMapping] = defaultdict(dict)
        compat_import_grouped: t.MappingKV[
            Path, t.MutableSequenceOf[m.Infra.CompatibilityAliasViolation]
        ] = defaultdict(list)
        project_alias_grouped: t.MappingKV[
            Path, t.MutableSequenceOf[m.Infra.CompatibilityAliasViolation]
        ] = defaultdict(list)
        project_alias_owners = c.ENFORCEMENT_PROJECT_ALIAS_OWNERS
        for violation in violations:
            if not violation.module_name:
                assignment_grouped[Path(violation.file)][violation.alias_name] = (
                    violation.target_name
                )
                continue
            if (
                violation.alias_name == violation.target_name
                and violation.module_name in project_alias_owners
            ):
                # ENFORCE-080: canonical alias owned locally but imported from flext_core.
                project_alias_grouped[Path(violation.file)].append(violation)
            else:
                compat_import_grouped[Path(violation.file)].append(violation)
        for file_path, alias_map in assignment_grouped.items():
            FlextInfraUtilitiesRefactorNamespaceMoves._rewrite_compat_aliases_in_file(
                file_path=file_path, alias_map=alias_map, gates=gates
            )
        all_import_files = [
            *compat_import_grouped.keys(),
            *project_alias_grouped.keys(),
        ]
        repository_root = (
            FlextInfraUtilitiesRefactorNamespaceCommon.shared_repository_root(
                py_files=all_import_files
            )
            if all_import_files
            else None
        )
        if repository_root is None:
            return
        with FlextInfraUtilitiesRopeCore.open_project(repository_root) as rope_project:
            for file_path, file_violations in project_alias_grouped.items():
                current_project = file_violations[0].module_name
                FlextInfraUtilitiesRefactorNamespaceMoves._rewrite_project_alias_imports_in_file(
                    rope_project=rope_project,
                    file_path=file_path,
                    current_project=current_project,
                )
            for file_path, file_violations in compat_import_grouped.items():
                FlextInfraUtilitiesRefactorNamespaceMoves._rewrite_compat_import_aliases_in_file(
                    rope_project=rope_project,
                    file_path=file_path,
                    violations=file_violations,
                    gates=gates,
                )

    @staticmethod
    def _rewrite_compat_aliases_in_file(
        *, file_path: Path, alias_map: t.StrMapping, gates: t.StrSequence | None
    ) -> None:
        """Rewrite compat aliases in file."""
        source = file_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        kept_source = "\n".join(
            line
            for line in source.splitlines()
            if FlextInfraUtilitiesRefactorNamespaceCommon.compat_assignment_target(
                line, alias_map=alias_map
            )
            is None
        )
        rewritten = FlextInfraUtilitiesRefactorNamespaceCommon.apply_token_replacements(
            source=kept_source, alias_map=alias_map
        )
        if rewritten != source:
            _ = FlextInfraUtilitiesProtectedEdit.protected_source_write(
                file_path,
                request=m.Infra.ProtectedSourceWriteRequest(
                    workspace=file_path.parent,
                    updated_source=rewritten,
                    keep_backup=True,
                    gates=gates,
                ),
            )

    @staticmethod
    def _rewrite_project_alias_imports_in_file(
        *, rope_project: t.Infra.RopeProject, file_path: Path, current_project: str
    ) -> None:
        """Rewrite ENFORCE-080 imports using the project alias migrator."""
        resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
            rope_project, file_path
        )
        if resource is None:
            return
        original_source = resource.read()
        backup_path = file_path.with_suffix(
            file_path.suffix + c.Infra.SAFE_EXECUTION_BAK_SUFFIX
        )
        if not backup_path.exists():
            backup_path.write_text(original_source, encoding=c.Cli.ENCODING_DEFAULT)
        transformer = FlextInfraRefactorProjectAliasMigrator(
            current_project=current_project
        )
        updated, changes = transformer.transform(rope_project, resource)
        if changes:
            cleanup_result = FlextInfraUtilitiesRopeImports.normalize_imports(
                rope_project, file_paths=(file_path,)
            )
            if cleanup_result.failure:
                msg = cleanup_result.error or "rope import cleanup failed"
                raise RuntimeError(msg)
        _ = updated

    @staticmethod
    def _rewrite_compat_import_aliases_in_file(
        *,
        rope_project: t.Infra.RopeProject,
        file_path: Path,
        violations: t.SequenceOf[m.Infra.CompatibilityAliasViolation],
        gates: t.StrSequence | None,
    ) -> None:
        """Rewrite non-canonical facade imports using Rope rename (file-local)."""
        _ = gates
        resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
            rope_project, file_path
        )
        if resource is None:
            return
        source = resource.read()

        def _names(source_text: str) -> set[str]:
            return {
                tok.string
                for tok in tokenize.generate_tokens(StringIO(source_text).readline)
                if tok.type == tokenize.NAME
            }

        names_in_file = _names(source)
        changed = False
        for violation in violations:
            long_name = violation.alias_name
            canonical_alias = violation.target_name
            if canonical_alias in names_in_file and canonical_alias != long_name:
                # Avoid shadowing an existing name in the file.
                continue
            offset = source.find(long_name)
            if offset < 0:
                continue
            try:
                changes = FlextInfraUtilitiesRopeRuntime.rename_changes(
                    rope_project,
                    resource,
                    offset,
                    canonical_alias,
                    resources=(resource,),
                )
            except (
                *FlextInfraUtilitiesRopeRuntime.rope_runtime_errors(),
                *FlextInfraUtilitiesRopeRuntime.rope_syntax_errors(),
                TypeError,
                ValueError,
            ):
                continue
            rope_project.do(changes)
            changed = True
            source = resource.read()
            names_in_file = _names(source)
        if changed:
            cleanup_result = FlextInfraUtilitiesRopeImports.normalize_imports(
                rope_project, file_paths=(file_path,)
            )
            if cleanup_result.failure:
                msg = cleanup_result.error or "rope import cleanup failed"
                raise RuntimeError(msg)

    @staticmethod
    def _move_named_blocks(
        *,
        project_root: Path,
        source_file: Path,
        target_filename: str,
        names: t.Infra.StrSet,
        header_prefix: str,
        gates: t.StrSequence | None,
    ) -> t.Triple[Path, Path, t.VariadicTuple[str]] | None:
        """Move named blocks."""
        source = source_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        lines = source.splitlines()
        blocks: t.MutableSequenceOf[str] = []
        ranges: t.MutableSequenceOf[t.IntPair] = []
        moved: t.MutableSequenceOf[str] = []
        for name in sorted(names):
            found = FlextInfraUtilitiesRefactorNamespaceCommon.find_top_level_block(
                lines=lines, header=f"{header_prefix}{name}"
            )
            if found is None:
                continue
            start, end = found
            blocks.append("\n".join(lines[start:end]))
            ranges.append((start, end))
            moved.append(name)
        if not blocks:
            return None
        target_file = FlextInfraUtilitiesRefactorNamespaceCommon.canonical_target_file(
            project_root=project_root, source_file=source_file, filename=target_filename
        )
        required_imports = (
            FlextInfraUtilitiesRefactorNamespaceMoves._collect_required_import_lines(
                source=source, blocks=blocks
            )
        )
        target_source = (
            target_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            if target_file.exists()
            else f"{c.Infra.FUTURE_ANNOTATIONS}\n"
        )
        target_lines = target_source.splitlines()
        target_lines = FlextInfraUtilitiesRefactorNamespaceCommon.insert_import_lines(
            lines=target_lines, imports=required_imports
        )
        updated_target = "\n".join(target_lines).rstrip()
        for block in blocks:
            if block.splitlines()[0] not in updated_target:
                updated_target += f"\n\n{block}"
        filtered_lines = list(lines)
        for start, end in sorted(ranges, reverse=True):
            del filtered_lines[start:end]

        ok, reports = FlextInfraUtilitiesProtectedEdit.protected_source_writes(
            {
                target_file: updated_target.rstrip() + "\n",
                source_file: "\n".join(filtered_lines).rstrip() + "\n",
            },
            request=m.Infra.ProtectedSourceWritesRequest(
                workspace=project_root, keep_backup=True, gates=gates
            ),
        )
        if not ok:
            msg = "named block move failed validation: " + "; ".join(reports)
            raise RuntimeError(msg)
        return (source_file, target_file, tuple(moved))

    @staticmethod
    def _collect_required_import_lines(
        *, source: str, blocks: t.StrSequence
    ) -> t.StrSequence:
        """Collect required import lines using rope-parsed module bodies."""
        source_pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if source_pymodule is None:
            return ()
        source_lines = source.splitlines()
        import_map: dict[str, str] = {}
        for node in getattr(source_pymodule.get_ast(), "body", []) or []:
            kind = FlextInfraUtilitiesRopeAnalysis.node_kind(node)
            if kind not in {"Import", "ImportFrom"}:
                continue
            lineno = getattr(node, "lineno", 1)
            end_lineno = getattr(node, "end_lineno", None) or lineno
            import_line = "\n".join(source_lines[lineno - 1 : end_lineno]).strip()
            for alias in getattr(node, "names", []) or []:
                alias_name = getattr(alias, "name", "")
                alias_as = getattr(alias, "asname", None)
                if kind == "Import":
                    bound_name = alias_as or alias_name.split(".", 1)[0]
                else:
                    bound_name = alias_as or alias_name
                if bound_name:
                    import_map[bound_name] = import_line
        required_imports: t.MutableSequenceOf[str] = []
        seen_imports: t.Infra.StrSet = set()
        for block in blocks:
            block_pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(block)
            if block_pymodule is None:
                continue
            for sub in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(
                block_pymodule.get_ast()
            ):
                if FlextInfraUtilitiesRopeAnalysis.node_kind(sub) != "Name":
                    continue
                import_line = import_map.get(getattr(sub, "id", ""))
                if import_line is None or import_line in seen_imports:
                    continue
                required_imports.append(import_line)
                seen_imports.add(import_line)
        return required_imports

    @staticmethod
    def _drop_moved_alias_exports(*, source: str, alias_names: t.Infra.StrSet) -> str:
        """Remove moved aliases from a literal module ``__all__`` assignment."""
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return source
        lines = source.splitlines()
        for node in getattr(pymodule.get_ast(), "body", ()) or ():
            if c.Infra.DUNDER_ALL not in (
                FlextInfraUtilitiesRopeAnalysis.assignment_target_names(node)
            ):
                continue
            exports = FlextInfraUtilitiesRopeAnalysis.literal_string_sequence(
                getattr(node, "value", None)
            )
            if not exports:
                return source
            kept_exports = tuple(name for name in exports if name not in alias_names)
            if kept_exports == tuple(exports):
                return source
            line_range = FlextInfraUtilitiesRopeAnalysis.line_col_range(node)
            if line_range is None:
                return source
            start_line, _, end_line, _ = line_range
            rendered_exports = ", ".join(f'"{name}"' for name in kept_exports)
            replacement = f"__all__: list[str] = [{rendered_exports}]"
            return "\n".join((*lines[: start_line - 1], replacement, *lines[end_line:]))
        return source

    @staticmethod
    def _move_typing_alias_lines(
        *,
        project_root: Path,
        source_file: Path,
        alias_names: t.Infra.StrSet,
        gates: t.StrSequence | None,
    ) -> None:
        """Move typing alias lines."""
        source = source_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        lines = source.splitlines()
        moved_lines: t.MutableSequenceOf[str] = []
        moved_line_numbers: t.MutableSequenceOf[int] = []
        kept_lines: t.MutableSequenceOf[str] = []
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            typing_match = c.Infra.TYPING_FACTORY_ASSIGN_RE.match(stripped)
            typing_name = typing_match.group(1) if typing_match is not None else ""
            legacy_alias_match = c.Infra.LEGACY_TYPEALIAS_RE.match(stripped)
            legacy_alias_name = (
                legacy_alias_match.group(1) if legacy_alias_match is not None else ""
            )
            should_move = any(
                stripped.startswith((f"type {name} =", f"{name}: TypeAlias ="))
                or typing_name == name
                for name in alias_names
            )
            if should_move:
                moved_lines.append(
                    f"type {legacy_alias_name} = {legacy_alias_match.group(2)}"
                    if legacy_alias_match is not None
                    else line
                )
                moved_line_numbers.append(line_number)
            else:
                kept_lines.append(line)
        if not moved_lines:
            return
        kept_source = "\n".join(kept_lines)
        kept_source = (
            FlextInfraUtilitiesRefactorNamespaceMoves._drop_moved_alias_exports(
                source=kept_source, alias_names=alias_names
            )
        )
        kept_lines = kept_source.splitlines()
        required_imports = (
            FlextInfraUtilitiesRefactorNamespaceMoves._collect_required_import_lines(
                source=source, blocks=moved_lines
            )
        )
        orphaned_imports = (
            FlextInfraUtilitiesRefactorNamespaceMoves._collect_orphaned_import_lines(
                source=source, kept_source=kept_source, max_line=min(moved_line_numbers)
            )
        )
        target_file = FlextInfraUtilitiesRefactorNamespaceCommon.canonical_target_file(
            project_root=project_root,
            source_file=source_file,
            filename=c.Infra.TYPINGS_PY,
        )
        target_source = (
            target_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            if target_file.exists()
            else f"{c.Infra.FUTURE_ANNOTATIONS}\n"
        )
        fallback_runtime_imports = FlextInfraUtilitiesRefactorNamespaceMoves._collect_missing_runtime_alias_imports(
            target_source=target_source, blocks=moved_lines
        )
        target_lines = target_source.splitlines()
        missing_imports = [
            filtered
            for import_line in [
                *required_imports,
                *orphaned_imports,
                *fallback_runtime_imports,
            ]
            if import_line not in target_lines
            if (
                filtered
                := FlextInfraUtilitiesRefactorNamespaceMoves._strip_self_bound_aliases(
                    import_line=import_line, target_source=target_source
                )
            )
        ]
        target_lines = FlextInfraUtilitiesRefactorNamespaceCommon.insert_import_lines(
            lines=target_lines, imports=missing_imports
        )
        updated_target = "\n".join(target_lines).rstrip()
        for moved_line in moved_lines:
            if moved_line not in target_lines:
                updated_target += f"\n\n{moved_line}"
        source_imports = (
            FlextInfraUtilitiesRefactorNamespaceMoves._typing_alias_source_imports(
                project_root=project_root,
                target_file=target_file,
                kept_source=kept_source,
                alias_names=alias_names,
            )
        )
        updated_source_lines = (
            FlextInfraUtilitiesRefactorNamespaceCommon.insert_import_lines(
                lines=kept_lines, imports=source_imports
            )
            if source_imports
            else kept_lines
        )

        ok, reports = FlextInfraUtilitiesProtectedEdit.protected_source_writes(
            {
                target_file: updated_target + "\n",
                source_file: "\n".join(updated_source_lines).rstrip() + "\n",
            },
            request=m.Infra.ProtectedSourceWritesRequest(
                workspace=project_root, keep_backup=True, gates=gates
            ),
        )
        if not ok:
            msg = "typing alias move failed validation: " + "; ".join(reports)
            raise RuntimeError(msg)

    @staticmethod
    def _typing_alias_source_imports(
        *,
        project_root: Path,
        target_file: Path,
        kept_source: str,
        alias_names: t.Infra.StrSet,
    ) -> t.StrSequence:
        """Typing alias source imports."""
        source_pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(
            kept_source
        )
        if source_pymodule is None:
            return ()
        referenced_aliases = sorted({
            getattr(node, "id", "")
            for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(
                source_pymodule.get_ast()
            )
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) == "Name"
            and getattr(node, "id", "") in alias_names
        })
        referenced_aliases = [name for name in referenced_aliases if name]
        if not referenced_aliases:
            return ()
        src_root = project_root / c.Infra.DEFAULT_SRC_DIR
        try:
            module_name = ".".join(
                target_file.relative_to(src_root).with_suffix("").parts
            )
        except ValueError:
            return ()
        import_line = f"from {module_name} import {', '.join(referenced_aliases)}"
        return [import_line] if import_line not in kept_source.splitlines() else ()

    @staticmethod
    def _strip_self_bound_aliases(*, import_line: str, target_source: str) -> str:
        """Drop names already bound locally in the move target from an import.

        A facade-root module (``typings.py``, ``constants.py``, ...) binds its
        canonical alias directly (``t = <Project>Types``). Injecting
        ``from <pkg> import t`` there is a self-package import that shadows the
        binding and raises F811. Keep only the names the target does not already
        own; return ``""`` when nothing remains so the whole line is dropped.
        """
        prefix, separator, names_part = import_line.partition(" import ")
        if not separator:
            return import_line
        kept = [
            f"{name} as {bound}" if name != bound else name
            for name, bound in FlextInfraUtilitiesRopeSource.parse_import_names(
                names_part
            )
            if not _header.alias_locally_bound(target_source, bound)
        ]
        if not kept:
            return ""
        return f"{prefix} import {', '.join(kept)}"

    @staticmethod
    def _collect_missing_runtime_alias_imports(
        *, target_source: str, blocks: t.StrSequence
    ) -> t.StrSequence:
        """Collect missing runtime alias imports."""
        moved_source = "\n".join(blocks)
        moved_pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(
            moved_source
        )
        runtime_aliases = u.runtime_alias_names(c.Infra.PKG_INFRA_UNDERSCORE)
        moved_aliases: set[str] = set()
        if moved_pymodule is not None:
            for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(
                moved_pymodule.get_ast()
            ):
                if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "Name":
                    continue
                node_id = getattr(node, "id", "")
                if node_id in runtime_aliases:
                    moved_aliases.add(node_id)
        if not moved_aliases:
            return ()
        imported_aliases: t.Infra.StrSet = set()
        for match in c.Infra.FROM_IMPORT_RE.finditer(target_source):
            imported_aliases.update(
                bound
                for _, bound in FlextInfraUtilitiesRopeSource.parse_import_names(
                    match.group(2)
                )
            )
        for match in c.Infra.FROM_IMPORT_BLOCK_RE.finditer(target_source):
            imported_aliases.update(
                bound
                for _, bound in FlextInfraUtilitiesRopeSource.parse_import_names(
                    match.group(2)
                )
            )
        missing_aliases = sorted(moved_aliases - imported_aliases)
        if not missing_aliases:
            return ()
        return [
            f"from {c.Infra.PKG_CORE_UNDERSCORE} import {', '.join(missing_aliases)}"
        ]

    @staticmethod
    def _collect_orphaned_import_lines(
        *, source: str, kept_source: str, max_line: int
    ) -> t.StrSequence:
        """Collect orphaned import lines via rope-parsed bodies."""
        source_pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if source_pymodule is None:
            return ()
        source_lines = source.splitlines()
        kept_pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(kept_source)
        kept_names: set[str] = set()
        if kept_pymodule is not None:
            for sub in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(
                kept_pymodule.get_ast()
            ):
                if FlextInfraUtilitiesRopeAnalysis.node_kind(sub) == "Name":
                    name = getattr(sub, "id", "")
                    if name:
                        kept_names.add(name)
        import_lines: t.MutableSequenceOf[str] = []
        for node in getattr(source_pymodule.get_ast(), "body", []) or []:
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) not in {
                "Import",
                "ImportFrom",
            }:
                continue
            lineno = getattr(node, "lineno", 0)
            if lineno >= max_line:
                continue
            bound_names = {
                getattr(alias, "asname", None)
                or getattr(alias, "name", "").split(".", maxsplit=1)[0]
                for alias in getattr(node, "names", []) or []
            }
            if bound_names & kept_names:
                continue
            end_lineno = getattr(node, "end_lineno", None) or lineno
            import_lines.append(
                "\n".join(source_lines[lineno - 1 : end_lineno]).strip()
            )
        return import_lines

    @staticmethod
    def _rewrite_moved_imports(
        *,
        project_root: Path,
        py_files: t.SequenceOf[Path],
        moves: t.SequenceOf[t.Triple[Path, Path, t.VariadicTuple[str]]],
    ) -> None:
        """Rewrite moved imports."""
        with FlextInfraUtilitiesRopeCore.open_project(project_root) as rope_project:
            mappings: t.MutableSequenceOf[t.Triple[str, str, t.VariadicTuple[str]]] = []
            for source, target, names in moves:
                source_resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, source
                )
                target_resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, target
                )
                if source_resource is None or target_resource is None:
                    continue
                try:
                    source_module = FlextInfraUtilitiesRopeCore.get_pymodule(
                        rope_project, source_resource
                    ).get_name()
                    target_module = FlextInfraUtilitiesRopeCore.get_pymodule(
                        rope_project, target_resource
                    ).get_name()
                except (
                    *FlextInfraUtilitiesRopeRuntime.rope_runtime_errors(),
                    *FlextInfraUtilitiesRopeRuntime.rope_syntax_errors(),
                    TypeError,
                ):
                    continue
                if source_module and target_module:
                    mappings.append((source_module, target_module, names))
            for py_file in py_files:
                resolved_py_file = py_file.resolve()
                if not resolved_py_file.is_relative_to(project_root.resolve()):
                    msg = (
                        "refusing moved-import rewrite outside project: "
                        f"{resolved_py_file}"
                    )
                    raise ValueError(msg)
                resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, resolved_py_file
                )
                if resource is None:
                    continue
                original_source = resolved_py_file.read_text(
                    encoding=c.Cli.ENCODING_DEFAULT
                )
                changed = False
                for source_module, target_module, names in mappings:
                    updated = (
                        FlextInfraUtilitiesRopeImports.relocate_from_import_aliases(
                            rope_project,
                            resource,
                            source_module=source_module,
                            target_module=target_module,
                            aliases=names,
                            apply=True,
                        )
                    )
                    changed = changed or updated is not None
                if changed:
                    cleanup_result = FlextInfraUtilitiesRopeImports.normalize_imports(
                        rope_project, file_paths=(resolved_py_file,)
                    )
                    if cleanup_result.failure:
                        msg = cleanup_result.error or "rope import cleanup failed"
                        raise RuntimeError(msg)
                    backup_path = resolved_py_file.with_suffix(
                        resolved_py_file.suffix + c.Infra.SAFE_EXECUTION_BAK_SUFFIX
                    )
                    if not backup_path.exists():
                        backup_path.write_text(
                            original_source, encoding=c.Cli.ENCODING_DEFAULT
                        )


__all__: list[str] = ["FlextInfraUtilitiesRefactorNamespaceMoves"]
