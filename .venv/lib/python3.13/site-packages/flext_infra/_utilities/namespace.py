"""Canonical codegen namespace utilities shared by census and auto-fix."""

from __future__ import annotations

import ast
import operator
from collections.abc import MutableMapping
from pathlib import Path
from typing import ClassVar, Final

from flext_cli import r, u
from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra._utilities.rope_source import FlextInfraUtilitiesRopeSource
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesCodegenNamespace:
    """Canonical namespace helpers for codegen discovery, parsing, and fixes."""

    _governance_cache: ClassVar[
        MutableMapping[str, m.Infra.ConstantsGovernanceConfig]
    ] = {}
    _governance_file: Final[Path] = (
        Path(__file__).parent.parent / "rules" / "constants-governance.yml"
    )
    # flext-perf.1 (agent: codex): cache __all__ AST extraction by path+mtime
    # so the 4-5 redundant _declared_exports calls per policy() hit memory
    # instead of re-reading + re-parsing the same file from disk each time.
    _declared_exports_cache: ClassVar[dict[str, tuple[int, t.StrSequence]]] = {}

    @classmethod
    def _is_rule_fixable(cls, rule_id: str, module: str) -> bool:
        """Is rule fixable."""
        cached = cls._governance_cache.get("settings")
        if cached is None:
            raw = u.Cli.yaml_load_mapping(cls._governance_file)
            cached = m.Infra.ConstantsGovernanceConfig.model_validate(raw)
            cls._governance_cache["settings"] = cached
        for rule in cached.rules:
            if rule.id != rule_id:
                continue
            if not rule.fixable:
                return False
            if rule.fixable_exclusion is None:
                return True
            return not module.endswith(rule.fixable_exclusion)
        return False

    @classmethod
    def matches_root_namespace_file(cls, file_name: str) -> bool:
        """Return whether *file_name* is a governed root-namespace facade file."""
        return file_name.endswith(c.Infra.EXT_PYTHON) and not file_name.startswith("_")

    @staticmethod
    def runtime_singleton_export(file_name: str) -> str | None:
        """Return the sole public singleton owned by a sanctioned runtime module."""
        return next(
            (
                Path(module_file).stem.removeprefix("_")
                for module_file, *_ in c.Infra.RUNTIME_MODULES
                if file_name == module_file
            ),
            None,
        )

    @classmethod
    def surface_name(cls, package_name: str) -> str:
        """Return the configured surface name for one import package."""
        parts = tuple(part for part in package_name.split(".") if part)
        if not parts:
            return ""
        return parts[0] if parts[0] in c.Infra.NON_PUBLIC_LAZY_ROOTS else "src"

    @classmethod
    def matches_project_namespace_package(cls, package_name: str) -> bool:
        """Return whether a package behaves like a project namespace root.

        Project package roots under ``src/`` qualify, as do namespace packages under
        governed wrapper surfaces like ``tests/``, ``examples/``, and ``scripts/``.
        Ordinary nested runtime subpackages such as ``<pkg>.services`` do not.
        """
        parts = tuple(part for part in package_name.split(".") if part)
        if not parts:
            return False
        return len(parts) == 1

    @staticmethod
    def ordered_namespace_exports(
        *, package_dir: Path, package_name: str, export_names: t.StrSequence
    ) -> t.StrSequence:
        """Order root-package exports with alias hierarchy preserved."""
        _ = (package_dir, package_name)
        ordered_unique = tuple(dict.fromkeys(export_names))
        export_set = set(ordered_unique)
        preferred_aliases = tuple(
            alias for alias in c.Infra.PUBLIC_ROOT_ALIAS_ORDER if alias in export_set
        )
        declared_aliases = tuple(
            name
            for name in ordered_unique
            if (
                name.islower()
                and len(name) <= c.Infra.MAX_ALIAS_LENGTH
                and name not in preferred_aliases
            )
        )
        ordered_aliases = (*preferred_aliases, *declared_aliases)
        alias_set = set(ordered_aliases)
        other_exports = tuple(name for name in ordered_unique if name not in alias_set)
        return (*other_exports, *ordered_aliases)

    @staticmethod
    def package_alias(*, package_name: str) -> str:
        """Derive the canonical root API alias from the import package."""
        root = package_name.split(".", maxsplit=1)[0]
        if root.startswith(c.Infra.PKG_PREFIX_UNDERSCORE):
            return root.removeprefix(c.Infra.PKG_PREFIX_UNDERSCORE)
        return root

    @classmethod
    def _runtime_aliases(cls, package_dir: Path) -> t.StrSequence:
        """Return the runtime aliases actually published by one package root."""
        return tuple(
            alias
            for module_file in sorted(package_dir.glob("*.py"))
            for alias in cls._declared_exports(module_file)
            if (
                alias.isidentifier()
                and alias.islower()
                and len(alias) <= c.Infra.MAX_ALIAS_LENGTH
            )
        )

    @staticmethod
    def _literal_exports(
        tree: ast.Module, value: ast.expr, file_path: Path
    ) -> t.StrSequence:
        """Resolve a literal ``__all__`` or the former generated tuple alias."""
        assignments: dict[str, ast.expr] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                assignments.update(
                    (target.id, node.value)
                    for target in node.targets
                    if isinstance(target, ast.Name)
                )
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                assignments[node.target.id] = node.value
        resolved = value
        match value:
            case ast.Call(
                func=ast.Name(id="tuple"), args=[ast.Name(id=source_name)], keywords=[]
            ):
                if source_name not in assignments:
                    msg = f"{file_path}: __all__ references missing {source_name}"
                    raise ValueError(msg)
                resolved = assignments[source_name]
            case _:
                pass
        try:
            literal = ast.literal_eval(resolved)
        except (ValueError, SyntaxError) as exc:
            msg = f"{file_path}: invalid __all__: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(literal, (list, tuple)) or not all(
            isinstance(item, str) for item in literal
        ):
            msg = f"{file_path}: __all__ must contain only strings"
            raise ValueError(msg)
        return tuple(literal)

    @classmethod
    def _declared_exports(cls, file_path: Path) -> t.StrSequence:
        if not file_path.is_file():
            return ()
        resolved = file_path.resolve()
        try:
            mtime_ns = resolved.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        cache_key = str(resolved)
        cached = cls._declared_exports_cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]
        tree = ast.parse(file_path.read_text(encoding=c.Infra.ENCODING_DEFAULT))
        result: t.StrSequence = ()
        for node in tree.body:
            match node:
                case ast.Assign(targets=targets, value=value) if any(
                    isinstance(target, ast.Name) and target.id == c.Infra.DUNDER_ALL
                    for target in targets
                ):
                    result = cls._literal_exports(tree, value, file_path)
                    break
                case ast.AnnAssign(target=ast.Name(id=name), value=value) if (
                    name == c.Infra.DUNDER_ALL and value is not None
                ):
                    result = cls._literal_exports(tree, value, file_path)
                    break
                case _:
                    continue
        cls._declared_exports_cache[cache_key] = (mtime_ns, result)
        return result

    @classmethod
    def layout(
        cls, project_root: Path, *, project: p.Infra.ProjectInfo | None = None
    ) -> m.Infra.RopeProjectLayout | None:
        """Return the canonical project layout contract for one project root."""
        resolved_root = project_root.resolve()
        package_name = (
            project.package_name
            if project is not None and project.package_name
            else FlextInfraUtilitiesDiscovery.package_name(resolved_root)
        )
        if not package_name:
            return None
        if project is not None and project.name:
            project_name = project.name
        elif (resolved_root / c.Infra.PYPROJECT_FILENAME).is_file():
            project_name = FlextInfraUtilitiesDocsScope.project_name_from_payload(
                resolved_root,
                FlextInfraUtilitiesDocsScope.project_payload(resolved_root),
            )
        else:
            project_name = resolved_root.name
        class_name_source = (
            project_name
            if project_name != resolved_root.name
            else package_name.split(".", maxsplit=1)[0].replace("_", "-")
        )
        src_dir = resolved_root / c.Infra.DEFAULT_SRC_DIR
        package_dir = src_dir / Path(*package_name.split("."))
        return m.Infra.RopeProjectLayout(
            project_root=resolved_root,
            project_name=project_name,
            package_name=package_name,
            package_alias=cls.package_alias(package_name=package_name),
            class_stem=u.derive_class_stem(class_name_source),
            src_dir=src_dir,
            package_dir=package_dir,
            init_path=package_dir / c.Infra.INIT_PY,
            runtime_aliases=cls._runtime_aliases(package_dir),
        )

    @classmethod
    def _resolve_family(
        cls, file_path: Path
    ) -> tuple[str | None, str | None, str | None, t.StrSequence]:
        """Return (family_alias, expected_family, expected_alias, family_tokens)."""
        family_alias = next(
            (
                alias
                for alias, directory in c.Infra.FAMILY_DIRECTORIES.items()
                if file_path.parent.name == directory
            ),
            None,
        )
        declared_exports = cls._declared_exports(file_path)
        expected_alias = next(
            (
                name
                for name in declared_exports
                if name.islower() and len(name) <= c.Infra.MAX_ALIAS_LENGTH
            ),
            family_alias,
        )
        expected_family = next(
            (name for name in declared_exports if name[:1].isupper()),
            c.Infra.FAMILY_SUFFIXES.get(family_alias)
            if family_alias is not None
            else None,
        )
        family_tokens: t.StrSequence = (expected_family,) if expected_family else ()
        return family_alias, expected_family, expected_alias, family_tokens

    @classmethod
    def _resolve_module_flags(
        cls,
        file_path: Path,
        resolved_rel_path: Path,
        package_parts: t.StrSequence,
        family_alias: str | None,
        expected_alias: str | None,
        expected_family: str | None,
    ) -> tuple[bool, bool, bool, bool, bool, bool, bool, bool, str | None]:
        """Return all is_* booleans and resolved surface_name.

        Returns: (is_fixture_module, is_family_module, is_family_package,
                  is_services_module, is_services_package, is_namespace_file,
                  is_root_namespace, is_governed_namespace, resolved expected_alias)
        """
        package_depth = len(package_parts)
        is_fixture_module = file_path.parent.name == "_fixtures"
        family_dir_values = set(c.Infra.FAMILY_DIRECTORIES.values())
        is_family_module = any(
            part in family_dir_values for part in resolved_rel_path.parts
        )
        is_family_package = family_alias is not None or any(
            part in family_dir_values for part in package_parts
        )
        is_services_module = "services" in resolved_rel_path.parts
        is_services_package = "services" in package_parts
        runtime_singleton_export = (
            cls.runtime_singleton_export(resolved_rel_path.name)
            if len(resolved_rel_path.parts) == 1 and package_depth <= 1
            else None
        )
        is_namespace_file = bool(cls._declared_exports(file_path))
        is_governed_namespace = (
            expected_alias is not None or expected_family is not None
        )
        is_root_namespace = (
            is_namespace_file
            and len(resolved_rel_path.parts) == 1
            and package_depth <= 1
        )
        resolved_alias = expected_alias or runtime_singleton_export
        if (
            resolved_alias is None
            and is_root_namespace
            and resolved_rel_path.name == c.Infra.API_PY
        ):
            resolved_alias = cls.package_alias(
                package_name=".".join(package_parts) if package_parts else ""
            )
        return (
            is_fixture_module,
            is_family_module,
            is_family_package,
            is_services_module,
            is_services_package,
            is_namespace_file,
            is_root_namespace,
            is_governed_namespace,
            resolved_alias,
        )

    @classmethod
    def _resolve_project_prefix(cls, file_path: Path) -> str:
        """Derive the class-stem prefix for one file inside a project."""
        project_root = FlextInfraUtilitiesDiscovery.project_root(file_path)
        if project_root is None:
            return ""
        layout = cls.layout(project_root)
        if layout is None:
            return ""
        class_stem: str = layout.class_stem
        try:
            rel_parts = file_path.relative_to(project_root).parts
        except ValueError:
            return class_stem
        surface_prefix = (
            rel_parts[0].title()
            if rel_parts and rel_parts[0] in c.Infra.NON_PUBLIC_LAZY_ROOTS
            else ""
        )
        return f"{surface_prefix}{class_stem}" if surface_prefix else class_stem

    @classmethod
    def policy(
        cls, file_path: Path, *, rel_path: Path | None = None, current_pkg: str = ""
    ) -> m.Infra.NamespaceModulePolicy:
        """Return the derived Pydantic policy for one governed module."""
        package_name = current_pkg or FlextInfraUtilitiesDiscovery.package_name(
            file_path
        )
        resolved_rel_path = rel_path or Path(file_path.name)
        package_parts = tuple(part for part in package_name.split(".") if part)

        family_alias, expected_family, expected_alias, family_tokens = (
            cls._resolve_family(file_path)
        )
        (
            is_fixture_module,
            is_family_module,
            is_family_package,
            is_services_module,
            is_services_package,
            is_namespace_file,
            is_root_namespace,
            is_governed_namespace,
            expected_alias,
        ) = cls._resolve_module_flags(
            file_path,
            resolved_rel_path,
            package_parts,
            family_alias,
            expected_alias,
            expected_family,
        )

        surface_name = package_parts[0] if package_parts else ""
        if surface_name not in c.Infra.NON_PUBLIC_LAZY_ROOTS:
            surface_name = "src"

        enforce_contract = (
            is_fixture_module
            or is_family_module
            or is_services_module
            or is_governed_namespace
            or is_root_namespace
        )
        export_symbols = (
            surface_name == "src"
            or is_fixture_module
            or is_family_module
            or is_family_package
            or is_services_module
            or is_services_package
            or is_namespace_file
            or is_root_namespace
        )
        is_private_module = file_path.stem.startswith("_")
        include_in_lazy_init = not file_path.stem[:1].isdigit() and (
            not is_private_module
            or is_fixture_module
            or is_family_package
            or is_root_namespace
        )
        type_checking_imports = tuple(
            name
            for name in cls._declared_exports(file_path)
            if (
                name.isidentifier()
                and name.islower()
                and len(name) <= c.Infra.MAX_ALIAS_LENGTH
            )
        )
        return m.Infra.NamespaceModulePolicy(
            enforce_contract=enforce_contract,
            export_symbols=export_symbols,
            include_in_lazy_init=include_in_lazy_init,
            project_prefix=cls._resolve_project_prefix(file_path),
            expected_alias=expected_alias,
            expected_family=expected_family,
            family_tokens=family_tokens,
            accepted_suffixes=((expected_family,) if expected_family else ()),
            allow_main_export="main" in cls._declared_exports(file_path),
            allow_type_alias=(
                file_path.name == c.Infra.TYPINGS_PY
                or file_path.parent.name == c.Infra.FAMILY_DIRECTORIES["t"]
            ),
            is_fixture_module=is_fixture_module,
            type_checking_imports=type_checking_imports,
        )

    @classmethod
    def projects(
        cls,
        repository_root: Path,
        *,
        projects: t.SequenceOf[m.Infra.ProjectInfo] | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.ProjectInfo]]:
        """Discover only projects that participate in codegen automation."""
        if projects is None:
            projects_result = FlextInfraUtilitiesDocsScope.discover_projects(
                repository_root
            )
            if not projects_result.success:
                return r[t.SequenceOf[m.Infra.ProjectInfo]].fail(
                    projects_result.error or "project discovery failed"
                )
            discovered = projects_result.unwrap()
        else:
            discovered = projects
        selected = tuple(
            project
            for project in discovered
            if (project.path / c.Infra.PYPROJECT_FILENAME).exists()
        )
        return r[t.SequenceOf[m.Infra.ProjectInfo]].ok(selected)

    @classmethod
    def parse_namespace_validation(
        cls, validation: p.Result[m.Infra.ValidationReport]
    ) -> p.Result[tuple[m.Infra.CensusViolation, ...]]:
        """Convert validator output into typed census violations."""
        if validation.failure:
            return r[tuple[m.Infra.CensusViolation, ...]].fail(
                validation.error or "namespace validation failed"
            )
        report = validation.unwrap()
        parsed: list[m.Infra.CensusViolation] = []
        for violation in report.violations:
            match = c.Infra.VIOLATION_PATTERN.match(violation)
            if match is None:
                continue
            rule = match.group("rule")
            module = match.group("module")
            parsed.append(
                m.Infra.CensusViolation(
                    module=module,
                    rule=rule,
                    line=int(match.group("line")),
                    message=match.group("message"),
                    fixable=cls._is_rule_fixable(rule, module),
                )
            )
        return r[tuple[m.Infra.CensusViolation, ...]].ok(tuple(parsed))

    @classmethod
    def normalize_canonical_facades(
        cls, *, pkg_dir: Path, ctx: m.Infra.FixContext
    ) -> None:
        """Normalize canonical facade base classes for codegen auto-fix."""
        for file_name, base_import, base_name in (
            (
                c.Infra.CONSTANTS_PY,
                "from flext_core import FlextConstants as Constants",
                "Constants",
            ),
            (c.Infra.TYPINGS_PY, "from flext_core import FlextTypes as Types", "Types"),
        ):
            cls._normalize_facade_base(
                file_path=pkg_dir / file_name,
                base_import=base_import,
                base_name=base_name,
                ctx=ctx,
            )

    @classmethod
    def _normalize_facade_base(
        cls,
        *,
        file_path: Path,
        base_import: str,
        base_name: str,
        ctx: m.Infra.FixContext,
    ) -> None:
        """Add the canonical flext-core base to a *baseless* facade class only.

        A facade that already declares a base inherits from its proper owning
        project — flext-core directly, or an intermediate such as flext-web
        (``FlextApiConstants(FlextWebConstants)``). Those are left untouched:
        the pass never rebases an already-parented facade onto flext-core, so
        it is idempotent and correct across project boundaries.
        """
        if not file_path.is_file():
            return
        with FlextInfraUtilitiesRopeCore.open_project(file_path.parent) as rope_project:
            resource: t.Infra.RopeResource | None = (
                FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, file_path
                )
            )
            if resource is None:
                return
            source = resource.read()
            class_infos = sorted(
                FlextInfraUtilitiesRopeAnalysis.get_class_info(rope_project, resource),
                key=operator.attrgetter("line"),
            )
            if not class_infos:
                return
            class_info = class_infos[0]
            if class_info.bases:
                return
            lines = source.splitlines()
            header_idx = class_info.line - 1
            if not 0 <= header_idx < len(lines):
                return
            rewritten_header = cls._normalize_class_header(
                line=lines[header_idx], class_name=class_info.name, base_name=base_name
            )
            if rewritten_header == lines[header_idx]:
                return
            lines[header_idx] = rewritten_header
            updated = "\n".join(lines)
            if source.endswith("\n"):
                updated += "\n"
            if base_import not in updated:
                updated = cls._insert_import_line(
                    source=updated, import_line=base_import
                )
            if updated == source:
                return
            resource.write(updated)
            ctx.files_modified.add(str(file_path))
            ctx.fix(
                module=str(file_path),
                rule="NAMESPACE",
                line=1,
                message=(f"normalized {class_info.name} to inherit from {base_name}"),
            )

    @staticmethod
    def _normalize_class_header(*, line: str, class_name: str, base_name: str) -> str:
        """Normalize class header."""
        stripped = line.strip()
        prefix = f"class {class_name}"
        if not stripped.startswith(prefix) or not stripped.endswith(":"):
            return line
        indent = line[: len(line) - len(line.lstrip())]
        return f"{indent}class {class_name}({base_name}):"

    @classmethod
    def _insert_import_line(cls, *, source: str, import_line: str) -> str:
        """Insert an import after the module docstring and future imports.

        Uses the docstring-aware position helper so the import can never land
        inside a multi-line module docstring (the cause of the F821/"unexpected
        indent" corruption).
        """
        lines = source.splitlines()
        if import_line in lines:
            return source
        insert_at = (
            FlextInfraUtilitiesRopeSource.index_after_docstring_and_future_imports(
                lines
            )
        )
        before = lines[:insert_at]
        after = lines[insert_at:]
        inserted: t.MutableSequenceOf[str] = list(before)
        if inserted and inserted[-1]:
            inserted.append("")
        inserted.append(import_line)
        if after and after[0]:
            inserted.append("")
        inserted.extend(after)
        return "\n".join(inserted) + ("\n" if source.endswith("\n") else "")

    @classmethod
    def classify_violation_outcomes(
        cls,
        *,
        project_path: Path,
        initial_violations: t.SequenceOf[m.Infra.CensusViolation],
        remaining_violations: t.SequenceOf[m.Infra.CensusViolation],
    ) -> tuple[
        tuple[m.Infra.CensusViolation, ...], tuple[m.Infra.CensusViolation, ...]
    ]:
        """Split initial violations into fixed and still-skipped groups."""
        if not initial_violations:
            return ((), ())
        source_cache: MutableMapping[str, t.StrSequence] = {}
        remaining_keys = frozenset(
            cls._build_violation_key(
                violation=violation,
                project_path=project_path,
                source_cache=source_cache,
            )
            for violation in remaining_violations
        )
        fixed: t.MutableSequenceOf[m.Infra.CensusViolation] = []
        skipped: t.MutableSequenceOf[m.Infra.CensusViolation] = []
        for violation in initial_violations:
            key = cls._build_violation_key(
                violation=violation,
                project_path=project_path,
                source_cache=source_cache,
            )
            if violation.fixable and key not in remaining_keys:
                fixed.append(violation)
                continue
            skipped.append(violation)
        return (tuple(fixed), tuple(skipped))

    @staticmethod
    def _read_source_lines(project_path: Path, module: str) -> t.StrSequence:
        """Read source lines."""
        module_path = project_path / module
        if not module_path.is_file():
            return ()
        return module_path.read_text(encoding=c.Cli.ENCODING_DEFAULT).splitlines()

    @classmethod
    def _build_violation_key(
        cls,
        *,
        violation: m.Infra.CensusViolation,
        project_path: Path,
        source_cache: MutableMapping[str, t.StrSequence],
    ) -> m.Infra.ViolationKey:
        """Build violation key."""
        if violation.module not in source_cache:
            source_cache[violation.module] = cls._read_source_lines(
                project_path, violation.module
            )
        return m.Infra.ViolationKey.from_violation(
            violation, source_cache[violation.module]
        )


__all__: list[str] = ["FlextInfraUtilitiesCodegenNamespace"]
