"""Rope-backed import and rename operations."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from flext_cli import u

from flext_infra import c, m, p, r, t
from flext_infra._utilities import (
    FlextInfraUtilitiesRopeCore,
    FlextInfraUtilitiesRopeRuntime,
)
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra.transformers import FlextInfraRefactorProjectAliasMigrator


class FlextInfraUtilitiesRopeImports:
    """Rope-backed import organization and rename helpers."""

    @staticmethod
    def import_statements(
        module_imports: t.Infra.RopeModuleImports,
    ) -> t.SequenceOf[t.Infra.RopeImportStatement]:
        """Return validated Rope import statements from one module import collection."""
        return tuple(module_imports.imports)

    @staticmethod
    def import_statement_module_name(
        import_statement: t.Infra.RopeImportStatement,
    ) -> str | None:
        """Return the absolute module name represented by one Rope import statement."""
        import_info = import_statement.import_info
        if not FlextInfraUtilitiesRopeRuntime.is_from_import(import_info):
            return None
        module_name = import_info.module_name
        return module_name or None

    @staticmethod
    def import_statement_names_and_aliases(
        import_statement: t.Infra.RopeImportStatement,
    ) -> t.SequenceOf[tuple[str, str | None]]:
        """Return validated imported-name pairs from one Rope import statement."""
        import_info = import_statement.import_info
        if not (
            FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
            or FlextInfraUtilitiesRopeRuntime.is_normal_import(import_info)
        ):
            return ()
        return tuple(import_info.names_and_aliases)

    @classmethod
    def imported_module_paths(
        cls, module_imports: t.Infra.RopeModuleImports, *, current_package: str = ""
    ) -> t.StrSequence:
        """Return runtime import targets represented by a Rope module import set."""
        imported_paths: list[str] = []
        for import_statement in cls.import_statements(module_imports):
            declared_name = (
                getattr(import_statement.import_info, "module_name", "") or ""
            )
            level = getattr(import_statement.import_info, "level", 0) or 0
            module_name = cls.import_statement_module_name(import_statement)
            names_and_aliases = cls.import_statement_names_and_aliases(import_statement)
            if current_package and level > 0:
                resolved = FlextInfraUtilitiesRopeAnalysis.resolve_import_module(
                    current_package=current_package,
                    module_name=declared_name,
                    level=level,
                )
                if not resolved:
                    resolved = current_package
                if resolved != declared_name:
                    module_name = resolved
            if module_name is not None:
                imported_paths.append(module_name)
                imported_paths.extend(
                    f"{module_name}.{name}" for name, _alias in names_and_aliases
                )
                continue
            imported_paths.extend(name for name, _alias in names_and_aliases)
        return tuple(imported_paths)

    @staticmethod
    def find_occurrences(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        offset: int,
        *,
        resources: t.SequenceOf[t.Infra.RopeResource] | None = None,
        in_hierarchy: bool = False,
    ) -> t.SequenceOf[t.Infra.RopeLocation]:
        """Find all occurrences of the symbol at offset across the project."""
        try:
            return FlextInfraUtilitiesRopeRuntime.runtime_find_occurrences(
                rope_project,
                resource,
                offset,
                resources=resources,
                in_hierarchy=in_hierarchy,
            )
        except (
            *FlextInfraUtilitiesRopeRuntime.rope_runtime_errors(),
            TypeError,
            RecursionError,
        ) as exc:
            msg = (
                "rope find_occurrences failed for "
                f"{resource.path}@{offset}: {type(exc).__name__}: {exc!s}"
            )
            raise RuntimeError(msg) from exc

    @staticmethod
    def location_file_path(location: t.Infra.RopeLocation) -> Path | None:
        """Resolve one Rope occurrence back to an absolute file path."""
        resource = getattr(location, "resource", None)
        real_path = getattr(resource, "real_path", None)
        if isinstance(real_path, str) and real_path:
            return Path(real_path).resolve()
        path = getattr(resource, "path", None)
        if isinstance(path, str) and path:
            return Path(path)
        return None

    @staticmethod
    def indexed_search_resources(
        rope_workspace: p.AttributeProbe,
        *,
        resource: t.Infra.RopeResource,
        name: str,
        definition_path: Path,
        dependent_import_targets: t.StrSequence = (),
    ) -> tuple[t.Infra.RopeResource, ...] | None:
        """Build the minimal Rope resource set for semantic occurrence searches.

        The workspace name index already narrows the candidate module set to
        files that contain ``name`` textually. This helper converts that cheap
        index into concrete Rope resources so callers can still rely on Rope's
        semantic identity checks without scanning the full project.
        """
        name_index_getter = getattr(rope_workspace, "name_index", None)
        resource_getter = getattr(rope_workspace, "resource", None)
        if name_index_getter is None or resource_getter is None:
            return None
        occurrences = name_index_getter().get(name, ())
        resolved_definition = definition_path.resolve()
        dependent_paths: frozenset[str] | None = None
        import_dependents_getter = getattr(rope_workspace, "import_dependents", None)
        if dependent_import_targets and callable(import_dependents_getter):
            dependent_candidates: set[Path] = {resolved_definition}
            for import_target in dependent_import_targets:
                dependent_paths_raw = import_dependents_getter(import_target)
                if not isinstance(dependent_paths_raw, tuple):
                    msg = (
                        "rope import_dependents returned non-tuple for "
                        f"{import_target}: {type(dependent_paths_raw).__name__}"
                    )
                    raise TypeError(msg)
                for path in dependent_paths_raw:
                    if not isinstance(path, Path):
                        msg = (
                            "rope import_dependents returned invalid path for "
                            f"{import_target}: {type(path).__name__}"
                        )
                        raise TypeError(msg)
                    dependent_candidates.add(path.resolve())
            dependent_paths = frozenset(str(path) for path in dependent_candidates)
        seen_paths = {str(resolved_definition)}
        resources: list[t.Infra.RopeResource] = [resource]
        for path, _surface, _lines in occurrences:
            resolved_path = path.resolve()
            if resolved_path == resolved_definition or path.name == c.Infra.INIT_PY:
                continue
            cache_key = str(resolved_path)
            if dependent_paths is not None and cache_key not in dependent_paths:
                continue
            if cache_key in seen_paths:
                continue
            candidate_resource = resource_getter(resolved_path)
            if candidate_resource is None:
                msg = (
                    "rope search resource unavailable for indexed path "
                    f"{resolved_path} while resolving '{name}'"
                )
                raise RuntimeError(msg)
            seen_paths.add(cache_key)
            resources.append(candidate_resource)
        return tuple(resources)

    @staticmethod
    def organize_imports(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        apply: bool,
    ) -> p.Result[bool]:
        """Organize imports for one rope resource using rope's import tools.

        ``r.ok(True)`` when rope produced a non-empty change set; the
        change is applied when ``apply`` is set. ``r.ok(False)`` when
        rope produced no changes (already organised). ``r.fail(reason)``
        when rope raised a refactoring/resource/type error.
        """
        try:
            original_source = resource.read()
            organizer = FlextInfraUtilitiesRopeRuntime.import_organizer(rope_project)
            changes = organizer.organize_imports(resource)
        except (
            *FlextInfraUtilitiesRopeRuntime.rope_syntax_errors(),
            *FlextInfraUtilitiesRopeRuntime.rope_runtime_errors(),
            TypeError,
        ) as exc:
            return r[bool].fail(f"rope organize_imports raised: {exc!s}")
        if changes is None:
            return r[bool].ok(False)
        change_list_raw = getattr(changes, "changes", None)
        if not isinstance(change_list_raw, list):
            return r[bool].fail(
                "unexpected rope organize_imports result type: "
                f"{type(changes).__name__}"
            )
        change_list = tuple(change_list_raw)
        if not change_list:
            return r[bool].ok(False)
        changed = any(
            getattr(change, "new_contents", None) is not None
            and getattr(change, "new_contents", None) != original_source
            for change in change_list
        )
        if changed and apply:
            rope_project.do(changes)
        return r[bool].ok(changed)

    @classmethod
    def normalize_imports(
        cls,
        rope_project: t.Infra.RopeProject,
        *,
        file_paths: t.SequenceOf[Path],
        preserve_canonical_aliases: bool = False,
    ) -> p.Result[bool]:
        """Apply one centralized Rope+Ruff import cleanup for touched files.

        Runs Rope's import organizer per file first, then lets Ruff remove
        orphaned imports and normalize import ordering/formatting once across the
        touched path set. Returns whether an import cleanup changed any file.

        When ``preserve_canonical_aliases`` is set, runtime-alias imports from
        ``flext_core`` / ``flext_infra`` (e.g. ``from flext_core import c, m``)
        are restored after Ruff only when still referenced semantically, including
        forward references that Ruff cannot see inside string annotations.
        """
        existing_paths = tuple(path.resolve() for path in file_paths if path.is_file())
        if not existing_paths:
            return r[bool].ok(False)
        canonical_imports: dict[Path, list[tuple[str, tuple[str, ...]]]] = {}
        if preserve_canonical_aliases:
            try:
                canonical_imports = cls._collect_canonical_alias_imports(
                    rope_project, existing_paths
                )
            except ValueError as exc:
                return r[bool].fail(str(exc))
        rope_changed = False
        for file_path in existing_paths:
            resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                rope_project, file_path
            )
            if resource is None:
                continue
            organize_result = cls.organize_imports(rope_project, resource, apply=True)
            if organize_result.failure:
                return r[bool].fail(
                    organize_result.error or "rope organize_imports failed"
                )
            rope_changed = rope_changed or organize_result.unwrap_or(False)
        normalized_paths = tuple(str(path) for path in existing_paths)
        check_result = u.Cli.run_raw(
            ["ruff", "check", "--fix", "--select", "I,F401", *normalized_paths],
            timeout=c.Infra.TIMEOUT_SHORT,
        )
        if check_result.failure:
            return r[bool].fail(check_result.error or "ruff check --fix failed")
        if preserve_canonical_aliases:
            restore_result = cls._ensure_canonical_alias_imports(
                rope_project, canonical_imports
            )
            if restore_result.failure:
                return r[bool].fail(
                    restore_result.error or "canonical alias restore failed"
                )
            rope_changed = rope_changed or restore_result.unwrap_or(False)
        format_result = u.Cli.run_raw(
            ["ruff", "format", *normalized_paths], timeout=c.Infra.TIMEOUT_SHORT
        )
        if format_result.failure:
            return r[bool].fail(format_result.error or "ruff format failed")
        return r[bool].ok(rope_changed)

    @classmethod
    def _collect_canonical_alias_imports(
        cls, rope_project: t.Infra.RopeProject, file_paths: t.SequenceOf[Path]
    ) -> dict[Path, list[tuple[str, tuple[str, ...]]]]:
        """Collect canonical runtime-alias imports eligible for semantic restore."""
        runtime_aliases = u.runtime_alias_names(c.Infra.PKG_INFRA_UNDERSCORE)
        canonical_modules = frozenset({
            c.Infra.PKG_CORE_UNDERSCORE,
            c.Infra.PKG_INFRA_UNDERSCORE,
        })
        collected: dict[Path, list[tuple[str, tuple[str, ...]]]] = {}
        for file_path in file_paths:
            resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                rope_project, file_path
            )
            if resource is None:
                continue
            module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
                rope_project, resource
            )
            if module_imports is None:
                continue
            entries: list[tuple[str, tuple[str, ...]]] = []
            for import_stmt in cls.import_statements(module_imports):
                import_info = import_stmt.import_info
                if (
                    not FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                    or import_info.level != 0
                ):
                    continue
                if import_info.module_name not in canonical_modules:
                    continue
                plain_names = [
                    (name, alias)
                    for name, alias in import_info.names_and_aliases
                    if alias is None
                ]
                if not plain_names or not all(
                    name in runtime_aliases for name, _alias in plain_names
                ):
                    continue
                referenced_result = cls._referenced_runtime_aliases(
                    resource.read(), tuple(name for name, _alias in plain_names)
                )
                if referenced_result.failure:
                    msg = referenced_result.error or "alias reference scan failed"
                    raise ValueError(msg)
                referenced_aliases = referenced_result.unwrap_or(frozenset())
                alias_names = tuple(
                    name for name, _alias in plain_names if name in referenced_aliases
                )
                if not alias_names:
                    continue
                entries.append((import_info.module_name, alias_names))
            if entries:
                collected[file_path] = entries
        return collected

    @staticmethod
    def _referenced_runtime_aliases(
        source: str, aliases: t.SequenceOf[str]
    ) -> p.Result[frozenset[str]]:
        """Return runtime aliases referenced after Ruff cleanup."""
        alias_set = frozenset(aliases)
        referenced: set[str] = set()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return r[frozenset[str]].fail(
                f"source is not parseable after import cleanup: {exc!s}"
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in alias_set:
                referenced.add(node.id)
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value
                referenced.update(
                    alias for alias in alias_set if f"{alias}." in literal
                )
        return r[frozenset[str]].ok(frozenset(referenced))

    @classmethod
    def _ensure_canonical_alias_imports(
        cls,
        rope_project: t.Infra.RopeProject,
        collected: dict[Path, list[tuple[str, tuple[str, ...]]]],
    ) -> p.Result[bool]:
        """Re-add canonical runtime-alias imports removed by Ruff F401 cleanup."""
        changed_any = False
        for file_path, entries in collected.items():
            resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                rope_project, file_path
            )
            if resource is None:
                continue
            module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
                rope_project, resource
            )
            if module_imports is None:
                continue
            referenced_aliases_result = cls._referenced_runtime_aliases(
                resource.read(),
                tuple(alias for _module_name, aliases in entries for alias in aliases),
            )
            if referenced_aliases_result.failure:
                return r[bool].fail(
                    referenced_aliases_result.error or "alias reference scan failed"
                )
            referenced_aliases = referenced_aliases_result.unwrap_or(frozenset())
            current: dict[str, set[str]] = defaultdict(set)
            for import_stmt in cls.import_statements(module_imports):
                import_info = import_stmt.import_info
                if (
                    not FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                    or import_info.level != 0
                ):
                    continue
                current[import_info.module_name].update(
                    name
                    for name, alias in import_info.names_and_aliases
                    if alias is None
                )
            changed = False
            for module_name, alias_names in entries:
                missing = sorted(
                    (frozenset(alias_names) & referenced_aliases)
                    - current.get(module_name, set())
                )
                if not missing:
                    continue
                existing = current.get(module_name, set())
                if existing:
                    # Merge into existing from-import via rope mutation.
                    for import_stmt in cls.import_statements(module_imports):
                        import_info = import_stmt.import_info
                        if (
                            FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                            and import_info.level == 0
                            and import_info.module_name == module_name
                        ):
                            merged = list(import_info.names_and_aliases)
                            present = {name for name, alias in merged if alias is None}
                            merged.extend(
                                (name, None)
                                for name in sorted(missing)
                                if name not in present
                            )
                            import_stmt.import_info = (
                                FlextInfraUtilitiesRopeRuntime.from_import(
                                    module_name, 0, merged
                                )
                            )
                            changed = True
                            break
                else:
                    module_imports.add_import(
                        FlextInfraUtilitiesRopeRuntime.from_import(
                            module_name, 0, [(name, None) for name in missing]
                        )
                    )
                    changed = True
            if changed:
                module_imports.remove_duplicates()
                module_imports.sort_imports()
                updated_source = module_imports.get_changed_source()
                if updated_source != resource.read():
                    resource.write(updated_source)
                    changed_any = True
        return r[bool].ok(changed_any)

    @staticmethod
    def get_absolute_from_imports(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.SequenceOf[t.Infra.RopeFromImport]:
        """Return all absolute ``from x import ...`` descriptors in a module."""
        module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
            rope_project, resource
        )
        if module_imports is None:
            return ()
        import_statements = FlextInfraUtilitiesRopeImports.import_statements(
            module_imports
        )
        absolute_from_imports: list[t.Infra.RopeFromImport] = []
        for import_stmt in import_statements:
            import_info = import_stmt.import_info
            if (
                FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                and import_info.level == 0
            ):
                absolute_from_imports.append(import_info)
        return tuple(absolute_from_imports)

    @classmethod
    def relocate_from_import_aliases(
        cls,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        source_module: str,
        target_module: str,
        aliases: t.StrSequence,
        apply: bool,
    ) -> str | None:
        """Move unaliased names from one absolute import to another using Rope."""
        aliases_to_move = frozenset(aliases)
        if not aliases_to_move:
            return None
        module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
            rope_project, resource
        )
        if module_imports is None:
            return None
        original_source: str = resource.read()
        target_import_stmt, moved_aliases = cls._strip_aliases_from_source_imports(
            module_imports,
            source_module=source_module,
            target_module=target_module,
            aliases_to_move=aliases_to_move,
        )
        if not moved_aliases:
            return None
        merged_target_pairs = cls._merge_aliases_into_target(
            module_imports,
            target_import_stmt=target_import_stmt,
            target_module=target_module,
            moved_aliases=moved_aliases,
        )
        updated_source: str = module_imports.get_changed_source()
        if merged_target_pairs and cls._uses_parenthesized_from_import(
            source=original_source, module_name=target_module
        ):
            updated_source = cls._format_parenthesized_from_import(
                source=updated_source,
                module_name=target_module,
                names_and_aliases=merged_target_pairs,
            )
        if updated_source == original_source:
            return None
        if apply:
            resource.write(updated_source)
        return updated_source

    @staticmethod
    def _strip_aliases_from_source_imports(
        module_imports: t.Infra.RopeModuleImports,
        *,
        source_module: str,
        target_module: str,
        aliases_to_move: frozenset[str],
    ) -> tuple[t.Infra.RopeImportStatement | None, t.Infra.StrSet]:
        """Remove ``aliases_to_move`` from each ``from source_module`` statement.

        Returns ``(target_import_stmt_or_None, moved_aliases_set)``. Mutates
        the source-module statements in place via ``import_info`` reassignment;
        the caller still owns the merge into the target.
        """
        target_import_stmt: t.Infra.RopeImportStatement | None = None
        moved_aliases: t.Infra.StrSet = set()
        import_statements = module_imports.imports
        if not import_statements:
            return target_import_stmt, moved_aliases
        for import_stmt in import_statements:
            import_info = import_stmt.import_info
            if not (
                FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                and import_info.level == 0
            ):
                continue
            if import_info.module_name == target_module:
                target_import_stmt = import_stmt
            if import_info.module_name != source_module:
                continue
            kept_pairs: list[tuple[str, str | None]] = []
            for name, alias in import_info.names_and_aliases:
                if alias is None and name in aliases_to_move:
                    moved_aliases.add(name)
                    continue
                kept_pairs.append((name, alias))
            if len(kept_pairs) == len(import_info.names_and_aliases):
                continue
            import_stmt.import_info = FlextInfraUtilitiesRopeRuntime.from_import(
                source_module, 0, kept_pairs
            )
        return target_import_stmt, moved_aliases

    @staticmethod
    def _merge_aliases_into_target(
        module_imports: t.Infra.RopeModuleImports,
        *,
        target_import_stmt: t.Infra.RopeImportStatement | None,
        target_module: str,
        moved_aliases: t.Infra.StrSet,
    ) -> t.SequenceOf[tuple[str, str | None]]:
        """Merge ``moved_aliases`` into the target import; create one if missing."""
        sorted_moved = sorted(moved_aliases)
        if target_import_stmt is None:
            module_imports.add_import(
                FlextInfraUtilitiesRopeRuntime.from_import(
                    target_module, 0, [(name, None) for name in sorted_moved]
                )
            )
            module_imports.sort_imports()
            return tuple((name, None) for name in sorted_moved)
        import_info = target_import_stmt.import_info
        if not (
            FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
            and import_info.level == 0
            and import_info.module_name == target_module
        ):
            msg = (
                "rope target import mismatch for "
                f"{target_module}: {type(import_info).__name__}"
            )
            raise RuntimeError(msg)
        merged_pairs = list(import_info.names_and_aliases)
        existing_plain_names = {name for name, alias in merged_pairs if alias is None}
        merged_pairs.extend(
            (name, None) for name in sorted_moved if name not in existing_plain_names
        )
        target_import_stmt.import_info = FlextInfraUtilitiesRopeRuntime.from_import(
            target_module, 0, list(merged_pairs)
        )
        return tuple(merged_pairs)

    @staticmethod
    def _uses_parenthesized_from_import(*, source: str, module_name: str) -> bool:
        """Check whether source uses a parenthesized from import."""
        pattern = c.Infra.compile_from_module_paren_open(module_name)
        return any(pattern.match(line) for line in source.splitlines())

    @staticmethod
    def _format_parenthesized_from_import(
        *,
        source: str,
        module_name: str,
        names_and_aliases: t.SequenceOf[tuple[str, str | None]],
    ) -> str:
        """Format parenthesized from import."""
        entries = [
            (f"{name} as {alias}" if alias else name)
            for name, alias in names_and_aliases
        ]
        replacement = "\n".join([
            f"from {module_name} import (",
            *[f"    {entry}," for entry in entries],
            ")",
        ])
        pattern = c.Infra.compile_from_module_import_line(module_name)
        rewritten_source: str = pattern.sub(replacement, source, count=1)
        if rewritten_source == source:
            return source
        return rewritten_source

    @classmethod
    def collapse_submodule_alias_imports(
        cls,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        package_name: str,
        aliases: t.StrSequence | None = None,
        apply: bool,
    ) -> str | None:
        """Hoist ``from package.sub import alias`` into ``from package import alias``."""
        effective_aliases = aliases or tuple(
            u.runtime_alias_names(c.Infra.PKG_INFRA_UNDERSCORE)
        )
        requested_aliases = frozenset(
            alias for alias in effective_aliases if len(alias) == 1 and alias.islower()
        )
        result: str | None = None
        if requested_aliases:
            module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
                rope_project, resource
            )
            if module_imports is not None:
                moved_aliases: t.Infra.StrSet = set()
                package_prefix = f"{package_name}."
                import_statements = cls.import_statements(module_imports)
                for import_stmt in import_statements:
                    import_info = import_stmt.import_info
                    from_import = (
                        import_info
                        if FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                        and import_info.level == 0
                        else None
                    )
                    if from_import is None or not from_import.module_name.startswith(
                        package_prefix
                    ):
                        continue
                    kept_pairs: list[tuple[str, str | None]] = []
                    for name, alias in from_import.names_and_aliases:
                        if alias is None and name in requested_aliases:
                            moved_aliases.add(name)
                            continue
                        kept_pairs.append((name, alias))
                    if len(kept_pairs) == len(from_import.names_and_aliases):
                        continue
                    import_stmt.import_info = (
                        FlextInfraUtilitiesRopeRuntime.from_import(
                            from_import.module_name, 0, kept_pairs
                        )
                    )
                if moved_aliases:
                    module_imports.add_import(
                        FlextInfraUtilitiesRopeRuntime.from_import(
                            package_name,
                            0,
                            [(name, None) for name in sorted(moved_aliases)],
                        )
                    )
                    module_imports.remove_duplicates()
                    module_imports.sort_imports()
                    updated_source: str = module_imports.get_changed_source()
                    if updated_source != resource.read():
                        if apply:
                            resource.write(updated_source)
                        result = updated_source
        return result

    @staticmethod
    def add_import(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        from_module: str,
        names: t.StrSequence,
        *,
        apply: bool = True,
    ) -> str | None:
        """Add ``from <module> import <names>`` using rope's ImportOrganizer."""
        module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
            rope_project, resource
        )
        if module_imports is None:
            return None
        module_imports.add_import(
            FlextInfraUtilitiesRopeRuntime.from_import(
                from_module, 0, [(name, None) for name in sorted(names)]
            )
        )
        module_imports.remove_duplicates()
        module_imports.sort_imports()
        updated: str = module_imports.get_changed_source()
        if updated == resource.read():
            return None
        if apply:
            resource.write(updated)
        return updated

    @staticmethod
    def remove_import_names(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        from_module: str,
        names: t.StrSequence,
        *,
        apply: bool = True,
    ) -> str | None:
        """Remove specific names from ``from <module> import ...``."""
        names_to_remove = frozenset(names)
        module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
            rope_project, resource
        )
        if module_imports is None:
            return None
        changed = False
        import_statements = FlextInfraUtilitiesRopeImports.import_statements(
            module_imports
        )
        for import_stmt in import_statements:
            import_info = import_stmt.import_info
            from_import = (
                import_info
                if FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                and import_info.level == 0
                and import_info.module_name == from_module
                else None
            )
            if from_import is None:
                continue
            kept = [
                (name, alias)
                for name, alias in from_import.names_and_aliases
                if name not in names_to_remove
            ]
            if len(kept) < len(from_import.names_and_aliases):
                import_stmt.import_info = FlextInfraUtilitiesRopeRuntime.from_import(
                    from_module, 0, kept
                )
                changed = True
        if not changed:
            return None
        module_imports.remove_duplicates()
        module_imports.sort_imports()
        updated: str = module_imports.get_changed_source()
        if updated == resource.read():
            return None
        if apply:
            resource.write(updated)
        return updated

    @staticmethod
    def rewrite_private_import_bypass_violations(
        rope_project: t.Infra.RopeProject,
        violations: t.SequenceOf[m.Infra.PrivateImportBypassViolation],
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation],
        *,
        apply: bool = True,
    ) -> None:
        """Rewrite private-module imports to their canonical facade equivalents.

        Only auto-fixable violations (``symbol_exported=True``) are rewritten;
        callers already filter the violation list before invoking this helper.
        """
        _ = parse_failures
        removals: t.MappingKV[tuple[Path, str], set[str]] = defaultdict(set)
        additions: t.MappingKV[tuple[Path, str], set[str]] = defaultdict(set)
        for violation in violations:
            file_path = Path(violation.file)
            removals[file_path, violation.private_module].add(violation.imported_symbol)
            additions[file_path, violation.suggested_facade].add(
                violation.imported_symbol
            )
        for (file_path, private_module), names in removals.items():
            resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                rope_project, file_path
            )
            if resource is None:
                continue
            module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
                rope_project, resource
            )
            if module_imports is None:
                continue
            names_to_remove = frozenset(names)
            changed = False
            stmts_to_drop: list[t.Infra.RopeImportStatement] = []
            for import_stmt in FlextInfraUtilitiesRopeImports.import_statements(
                module_imports
            ):
                import_info = import_stmt.import_info
                if (
                    not FlextInfraUtilitiesRopeRuntime.is_from_import(import_info)
                    or import_info.module_name != private_module
                ):
                    continue
                kept = [
                    (name, alias)
                    for name, alias in import_info.names_and_aliases
                    if name not in names_to_remove
                ]
                if len(kept) == len(import_info.names_and_aliases):
                    continue
                if kept:
                    import_stmt.import_info = (
                        FlextInfraUtilitiesRopeRuntime.from_import(
                            private_module, 0, kept
                        )
                    )
                else:
                    stmts_to_drop.append(import_stmt)
                changed = True
            imports_list = module_imports.imports
            for import_stmt in stmts_to_drop:
                imports_list.remove(import_stmt)
            if not changed:
                continue
            module_imports.remove_duplicates()
            module_imports.sort_imports()
            updated = module_imports.get_changed_source()
            if apply and updated != resource.read():
                resource.write(updated)
        for (file_path, facade_module), names in additions.items():
            resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                rope_project, file_path
            )
            if resource is None:
                continue
            FlextInfraUtilitiesRopeImports.add_import(
                rope_project, resource, facade_module, sorted(names), apply=apply
            )

    @staticmethod
    def rewrite_foreign_canonical_alias_violations(
        rope_project: t.Infra.RopeProject,
        violations: t.SequenceOf[m.Infra.CompatibilityAliasViolation],
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation],
    ) -> None:
        """Rewrite foreign canonical alias imports to local project facades.

        Uses the ENFORCE-080 project alias migrator to move owned aliases from
        ``flext_core`` to the project's local facade modules.
        """
        _ = parse_failures
        file_paths: set[Path] = {Path(v.file) for v in violations}
        for file_path in file_paths:
            resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                rope_project, file_path
            )
            if resource is None:
                continue
            transformer = FlextInfraRefactorProjectAliasMigrator(file_path=file_path)
            updated, changes = transformer.transform(rope_project, resource)
            if changes:
                cleanup_result = FlextInfraUtilitiesRopeImports.normalize_imports(
                    rope_project, file_paths=(file_path,)
                )
                if cleanup_result.failure:
                    msg = cleanup_result.error or "rope import cleanup failed"
                    raise RuntimeError(msg)
            _ = updated


__all__: list[str] = ["FlextInfraUtilitiesRopeImports"]
