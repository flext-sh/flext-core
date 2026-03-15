"""Resolve and validate facade-family MRO chains."""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_infra import c, m, t, u
from flext_infra.refactor.transformers.mro_reference_rewriter import (
    FlextInfraRefactorMROReferenceRewriter,
)


class FlextInfraRefactorMROResolver:
    """MRO resolver for c/t/p/m/u facade families."""

    CONSTANT_PATTERN: re.Pattern[str] = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
    TYPE_CANDIDATE_PATTERN: re.Pattern[str] = re.compile(r"^_?[A-Za-z][A-Za-z0-9_]*$")

    @classmethod
    def resolve(
        cls,
        *,
        family_classes: Mapping[t.Infra.FacadeFamily, type],
        expected_base_chains: Mapping[
            t.Infra.FacadeFamily,
            Sequence[t.Infra.ExpectedBase],
        ],
    ) -> tuple[m.Infra.Refactor.FamilyMROResolution, ...]:
        """Resolve and validate MRO for all facade families."""
        resolutions: list[m.Infra.Refactor.FamilyMROResolution] = []
        for family in ("c", "t", "p", "m", "u"):
            facade_class = family_classes[family]
            expected_chain = expected_base_chains[family]
            resolutions.append(
                cls._resolve_family(
                    family=family,
                    facade_class=facade_class,
                    expected_chain=expected_chain,
                ),
            )
        return tuple(resolutions)

    @classmethod
    def resolve_from_classification(
        cls,
        *,
        family_classes: Mapping[t.Infra.FacadeFamily, type],
        classification: m.Infra.Refactor.ProjectClassification,
    ) -> tuple[m.Infra.Refactor.FamilyMROResolution, ...]:
        """Resolve MRO using family chains produced by project classifier."""
        expected_base_chains = cls._normalize_classifier_chains(
            family_chains=classification.family_chains,
        )
        return cls.resolve(
            family_classes=family_classes,
            expected_base_chains=expected_base_chains,
        )

    @classmethod
    def _resolve_family(
        cls,
        *,
        family: t.Infra.FacadeFamily,
        facade_class: type,
        expected_chain: Sequence[t.Infra.ExpectedBase],
    ) -> m.Infra.Refactor.FamilyMROResolution:
        expected_names = cls._normalize_expected_chain(expected_chain=expected_chain)
        cls._validate_base_policy(
            family=family,
            facade_class=facade_class,
            expected_names=expected_names,
        )
        resolved_mro = tuple(entry.__name__ for entry in inspect.getmro(facade_class))
        accessible_namespaces = cls._collect_accessible_namespaces(
            family=family,
            facade_class=facade_class,
        )
        cls._validate_expected_accessibility(
            family=family,
            expected_names=expected_names,
            accessible_namespaces=accessible_namespaces,
        )
        return m.Infra.Refactor.FamilyMROResolution(
            family=family,
            expected_bases=expected_names,
            resolved_mro=resolved_mro,
            accessible_namespaces=accessible_namespaces,
        )

    @classmethod
    def _normalize_classifier_chains(
        cls,
        *,
        family_chains: Mapping[str, Sequence[str]],
    ) -> dict[t.Infra.FacadeFamily, tuple[str, ...]]:
        normalized: dict[t.Infra.FacadeFamily, tuple[str, ...]] = {}
        for family in ("c", "t", "p", "m", "u"):
            raw_chain = family_chains.get(family)
            if raw_chain is None:
                msg = f"Missing expected family chain for {family!r}."
                raise ValueError(msg)
            normalized[family] = tuple(raw_chain)
        return normalized

    @classmethod
    def _normalize_expected_chain(
        cls,
        *,
        expected_chain: Sequence[t.Infra.ExpectedBase],
    ) -> tuple[str, ...]:
        expected_names: list[str] = []
        for base in expected_chain:
            if isinstance(base, str):
                expected_names.append(base)
                continue
            expected_names.append(base.__name__)
        return tuple(expected_names)

    @classmethod
    def _validate_base_policy(
        cls,
        *,
        family: t.Infra.FacadeFamily,
        facade_class: type,
        expected_names: tuple[str, ...],
    ) -> None:
        direct_base_names = tuple(base.__name__ for base in facade_class.__bases__)
        if len(direct_base_names) < len(expected_names):
            msg = f"family={family} has fewer direct bases than expected: expected={expected_names!r} direct={direct_base_names!r}"
            raise ValueError(msg)
        if direct_base_names[: len(expected_names)] != expected_names:
            msg = f"family={family} direct base order violates policy: expected={expected_names!r} direct={direct_base_names!r}"
            raise ValueError(msg)
        mro_types = inspect.getmro(facade_class)
        mro_names = tuple(entry.__name__ for entry in mro_types)
        mro_index = {name: index for index, name in enumerate(mro_names)}
        missing = tuple(name for name in expected_names if name not in mro_index)
        if missing:
            msg = f"family={family} missing expected bases in MRO: missing={missing!r} mro={mro_names!r}"
            raise ValueError(msg)
        previous_index = -1
        for base_name in expected_names:
            current_index = mro_index[base_name]
            if current_index <= previous_index:
                msg = f"family={family} MRO order is not C3-coherent for expected chain: expected={expected_names!r} mro={mro_names!r}"
                raise ValueError(msg)
            previous_index = current_index

    @classmethod
    def _validate_expected_accessibility(
        cls,
        *,
        family: t.Infra.FacadeFamily,
        expected_names: tuple[str, ...],
        accessible_namespaces: tuple[str, ...],
    ) -> None:
        missing_namespaces: list[str] = []
        for base_name in expected_names:
            namespace = cls._namespace_from_class_name(
                class_name=base_name,
                family=family,
            )
            if namespace is None:
                continue
            if namespace in accessible_namespaces:
                continue
            missing_namespaces.append(namespace)
        if missing_namespaces:
            msg = f"family={family} expected namespaces are not accessible: missing={tuple(missing_namespaces)!r} accessible={accessible_namespaces!r}"
            raise ValueError(msg)

    @classmethod
    def _collect_accessible_namespaces(
        cls,
        *,
        family: t.Infra.FacadeFamily,
        facade_class: type,
    ) -> tuple[str, ...]:
        namespace_order: list[str] = []
        for current in inspect.getmro(facade_class):
            if current is object:
                continue
            class_namespace = cls._namespace_from_class_name(
                class_name=current.__name__,
                family=family,
            )
            if class_namespace is not None:
                cls._append_unique(namespace_order, class_namespace)
            for member_name, member in vars(current).items():
                if member_name.startswith("_"):
                    continue
                if not isinstance(member, type):
                    continue
                cls._append_unique(namespace_order, member_name)
        return tuple(namespace_order)

    @classmethod
    def _namespace_from_class_name(
        cls,
        *,
        class_name: str,
        family: t.Infra.FacadeFamily,
    ) -> str | None:
        suffix = c.Infra.Refactor.FAMILY_SUFFIXES[family]
        if not class_name.endswith(suffix):
            return None
        root = class_name[: -len(suffix)]
        root = root.removeprefix("Flext")
        if not root:
            return None
        return root

    @staticmethod
    def _append_unique(namespaces: list[str], candidate: str) -> None:
        if candidate not in namespaces:
            namespaces.append(candidate)


class FlextInfraRefactorMROMigrationScanner:
    """Discover constants.py files with loose Final declarations."""

    @classmethod
    def scan_workspace(
        cls,
        *,
        workspace_root: Path,
        target: str,
    ) -> tuple[list[m.Infra.Refactor.MROScanReport], int]:
        """Scan workspace and return candidate files with counts."""
        if target not in c.Infra.Refactor.MRO_TARGETS:
            return ([], 0)
        results: list[m.Infra.Refactor.MROScanReport] = []
        scanned = 0
        target_specs = cls._target_specs(target=target)
        for project_root in cls._project_roots(workspace_root=workspace_root):
            for target_spec in target_specs:
                for file_path in cls._iter_target_files(
                    project_root=project_root,
                    target_spec=target_spec,
                ):
                    scanned += 1
                    result = cls.scan_file(
                        file_path=file_path,
                        project_root=project_root,
                        target_spec=target_spec,
                    )
                    if result is None or len(result.candidates) == 0:
                        continue
                    results.append(result)
        return (results, scanned)

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_root: Path,
        target_spec: m.Infra.Refactor.MROTargetSpec,
    ) -> m.Infra.Refactor.MROScanReport | None:
        """Scan a constants module for module-level Final constants."""
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return None
        constants_class = cls._facade_class_name(tree=tree, target_spec=target_spec)
        if not constants_class:
            return None
        module = cls._module_path(file_path=file_path, project_root=project_root)
        candidates: list[m.Infra.Refactor.MROSymbolCandidate] = []
        for stmt in tree.body:
            candidate = cls._candidate_from_statement(
                stmt=stmt,
                target_spec=target_spec,
            )
            if candidate is not None:
                candidates.append(candidate)
        return m.Infra.Refactor.MROScanReport(
            file=str(file_path),
            module=module,
            constants_class=constants_class,
            facade_alias=target_spec.family_alias,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _candidate_from_statement(
        *,
        stmt: ast.stmt,
        target_spec: m.Infra.Refactor.MROTargetSpec,
    ) -> m.Infra.Refactor.MROSymbolCandidate | None:
        if target_spec.family_alias == "t":
            return (
                FlextInfraRefactorMROMigrationScanner._typing_candidate_from_statement(
                    stmt=stmt,
                )
            )
        if target_spec.family_alias == "p":
            return FlextInfraRefactorMROMigrationScanner._protocol_candidate_from_statement(
                stmt=stmt,
            )
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name):
                return None
            if not FlextInfraRefactorMROResolver.CONSTANT_PATTERN.match(stmt.target.id):
                return None
            if not FlextInfraRefactorMROMigrationScanner._is_final_annotation(
                annotation=stmt.annotation,
            ):
                return None
            return m.Infra.Refactor.MROSymbolCandidate(
                symbol=stmt.target.id,
                line=stmt.lineno,
                kind="constant",
                class_name="",
                facade_name="",
            )
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                return None
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                return None
            if not FlextInfraRefactorMROResolver.CONSTANT_PATTERN.match(target.id):
                return None
            return m.Infra.Refactor.MROSymbolCandidate(
                symbol=target.id,
                line=stmt.lineno,
                kind="constant",
                class_name="",
                facade_name="",
            )
        return None

    @staticmethod
    def _project_roots(*, workspace_root: Path) -> list[Path]:
        return u.Infra.discover_project_roots(workspace_root=workspace_root)

    @staticmethod
    def _target_specs(*, target: str) -> tuple[m.Infra.Refactor.MROTargetSpec, ...]:
        ref_c: type[c.Infra.Refactor] = c.Infra.Refactor
        constants_spec = m.Infra.Refactor.MROTargetSpec(
            family_alias="c",
            file_names=ref_c.MRO_CONSTANTS_FILE_NAMES,
            package_directory=ref_c.MRO_CONSTANTS_DIRECTORY,
            class_suffix=ref_c.CONSTANTS_CLASS_SUFFIX,
        )
        typings_spec = m.Infra.Refactor.MROTargetSpec(
            family_alias="t",
            file_names=ref_c.MRO_TYPINGS_FILE_NAMES,
            package_directory=ref_c.MRO_TYPINGS_DIRECTORY,
            class_suffix="Types",
        )
        protocols_spec = m.Infra.Refactor.MROTargetSpec(
            family_alias="p",
            file_names=ref_c.MRO_PROTOCOLS_FILE_NAMES,
            package_directory=ref_c.MRO_PROTOCOLS_DIRECTORY,
            class_suffix="Protocols",
        )
        models_spec = m.Infra.Refactor.MROTargetSpec(
            family_alias="m",
            file_names=ref_c.MRO_MODELS_FILE_NAMES,
            package_directory=ref_c.MRO_MODELS_DIRECTORY,
            class_suffix="Models",
        )
        utilities_spec = m.Infra.Refactor.MROTargetSpec(
            family_alias="u",
            file_names=ref_c.MRO_UTILITIES_FILE_NAMES,
            package_directory=ref_c.MRO_UTILITIES_DIRECTORY,
            class_suffix="Utilities",
        )
        if target == "constants":
            return (constants_spec,)
        if target == "typings":
            return (typings_spec,)
        if target == "protocols":
            return (protocols_spec,)
        if target == "models":
            return (models_spec,)
        if target == "utilities":
            return (utilities_spec,)
        return (
            constants_spec,
            typings_spec,
            protocols_spec,
            models_spec,
            utilities_spec,
        )

    @staticmethod
    def _iter_constants_files(*, project_root: Path) -> list[Path]:
        ref_c: type[c.Infra.Refactor] = c.Infra.Refactor
        constants_spec = m.Infra.Refactor.MROTargetSpec(
            family_alias="c",
            file_names=ref_c.MRO_CONSTANTS_FILE_NAMES,
            package_directory=ref_c.MRO_CONSTANTS_DIRECTORY,
            class_suffix=ref_c.CONSTANTS_CLASS_SUFFIX,
        )
        return FlextInfraRefactorMROMigrationScanner._iter_target_files(
            project_root=project_root,
            target_spec=constants_spec,
        )

    @staticmethod
    def _iter_target_files(
        *,
        project_root: Path,
        target_spec: m.Infra.Refactor.MROTargetSpec,
    ) -> list[Path]:
        ref_c: type[c.Infra.Refactor] = c.Infra.Refactor
        candidates: set[Path] = set()
        for directory_name in ref_c.MRO_SCAN_DIRECTORIES:
            root: Path = project_root / directory_name
            if not root.is_dir():
                continue
            for file_path in u.Infra.iter_directory_python_files(root):
                if file_path.name in target_spec.file_names:
                    candidates.add(file_path)
                    continue
                if target_spec.package_directory in file_path.parts:
                    candidates.add(file_path)
        return sorted(candidates)

    @staticmethod
    def _module_path(*, file_path: Path, project_root: Path) -> str:
        return u.Infra.module_path(
            file_path=file_path,
            project_root=project_root,
        )

    @staticmethod
    def _facade_class_name(
        *,
        tree: ast.Module,
        target_spec: m.Infra.Refactor.MROTargetSpec,
    ) -> str:
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if (
                not isinstance(target, ast.Name)
                or target.id != target_spec.family_alias
            ):
                continue
            if not isinstance(stmt.value, ast.Name):
                continue
            class_name = stmt.value.id
            if class_name.endswith(target_spec.class_suffix):
                return class_name
        for stmt in tree.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            if stmt.name.endswith(target_spec.class_suffix):
                return stmt.name
        return ""

    @staticmethod
    def _typing_candidate_from_statement(
        *,
        stmt: ast.stmt,
    ) -> m.Infra.Refactor.MROSymbolCandidate | None:
        if isinstance(stmt, ast.TypeAlias):
            symbol = stmt.name.id
            if (
                FlextInfraRefactorMROResolver.TYPE_CANDIDATE_PATTERN.match(symbol)
                is None
            ):
                return None
            return m.Infra.Refactor.MROSymbolCandidate(
                symbol=symbol,
                line=stmt.lineno,
                kind="typealias",
                class_name="",
                facade_name="",
            )
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name):
                return None
            symbol = stmt.target.id
            if (
                FlextInfraRefactorMROResolver.TYPE_CANDIDATE_PATTERN.match(symbol)
                is None
            ):
                return None
            if not FlextInfraRefactorMROMigrationScanner._is_type_alias_annotation(
                annotation=stmt.annotation,
            ):
                return None
            return m.Infra.Refactor.MROSymbolCandidate(
                symbol=symbol,
                line=stmt.lineno,
                kind="typealias",
                class_name="",
                facade_name="",
            )
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return None
            symbol = stmt.targets[0].id
            if (
                FlextInfraRefactorMROResolver.TYPE_CANDIDATE_PATTERN.match(symbol)
                is None
            ):
                return None
            if not FlextInfraRefactorMROMigrationScanner._is_typing_factory_call(
                expr=stmt.value,
            ):
                return None
            return m.Infra.Refactor.MROSymbolCandidate(
                symbol=symbol,
                line=stmt.lineno,
                kind="typevar",
                class_name="",
                facade_name="",
            )
        return None

    @staticmethod
    def _protocol_candidate_from_statement(
        *,
        stmt: ast.stmt,
    ) -> m.Infra.Refactor.MROSymbolCandidate | None:
        if not isinstance(stmt, ast.ClassDef):
            return None
        has_protocol_base = False
        for base_expr in stmt.bases:
            if isinstance(base_expr, ast.Name) and base_expr.id == "Protocol":
                has_protocol_base = True
                break
            if isinstance(base_expr, ast.Attribute) and base_expr.attr == "Protocol":
                has_protocol_base = True
                break
            if isinstance(base_expr, ast.Subscript):
                root_expr = base_expr.value
                if isinstance(root_expr, ast.Name) and root_expr.id == "Protocol":
                    has_protocol_base = True
                    break
                if (
                    isinstance(root_expr, ast.Attribute)
                    and root_expr.attr == "Protocol"
                ):
                    has_protocol_base = True
                    break
        if not has_protocol_base:
            return None
        return m.Infra.Refactor.MROSymbolCandidate(
            symbol=stmt.name,
            line=stmt.lineno,
            kind="protocol",
            class_name="",
            facade_name="",
        )

    @staticmethod
    def _is_final_annotation(*, annotation: ast.expr) -> bool:
        final_name = c.Infra.Refactor.FINAL_ANNOTATION_NAME
        if isinstance(annotation, ast.Name):
            return annotation.id == final_name
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == final_name
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            if isinstance(base, ast.Name):
                return base.id == final_name
            if isinstance(base, ast.Attribute):
                return base.attr == final_name
        return False

    @staticmethod
    def _is_type_alias_annotation(*, annotation: ast.expr) -> bool:
        alias_name = "TypeAlias"
        if isinstance(annotation, ast.Name):
            return annotation.id == alias_name
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == alias_name
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            if isinstance(base, ast.Name):
                return base.id == alias_name
            if isinstance(base, ast.Attribute):
                return base.attr == alias_name
        return False

    @staticmethod
    def _is_typing_factory_call(*, expr: ast.expr) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        func = expr.func
        if isinstance(func, ast.Name):
            return func.id in {"TypeVar", "ParamSpec", "TypeVarTuple", "NewType"}
        if isinstance(func, ast.Attribute):
            return func.attr in {"TypeVar", "ParamSpec", "TypeVarTuple", "NewType"}
        return False


class FlextInfraRefactorMROImportRewriter:
    """Rewrite imports and references to use the local facade alias."""

    @classmethod
    def rewrite_workspace(
        cls,
        *,
        workspace_root: Path,
        moved_index: dict[str, dict[str, str]],
        module_facade_aliases: dict[str, str],
        apply: bool,
    ) -> list[m.Infra.Refactor.MRORewriteResult]:
        """Rewrite references across all project Python files."""
        results: list[m.Infra.Refactor.MRORewriteResult] = []
        for file_path in cls._iter_workspace_python_files(
            workspace_root=workspace_root,
        ):
            rewritten = cls.rewrite_file(
                file_path=file_path,
                moved_index=moved_index,
                module_facade_aliases=module_facade_aliases,
                apply=apply,
            )
            if rewritten is not None and rewritten.replacements > 0:
                results.append(rewritten)
        return results

    @staticmethod
    def rewrite_file(
        *,
        file_path: Path,
        moved_index: dict[str, dict[str, str]],
        module_facade_aliases: dict[str, str],
        apply: bool,
    ) -> m.Infra.Refactor.MRORewriteResult | None:
        """Rewrite one file according to moved constant symbol mappings."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except OSError:
            return None
        # NOTE: source text needed below for rendered-diff/write decisions.
        tree = u.Infra.parse_ast_from_source(source)
        if tree is None:
            return None
        imported_symbols: dict[str, m.Infra.Refactor.MROImportRewrite] = {}
        module_aliases: dict[str, str] = {}
        facade_aliases: dict[str, str] = {}
        module_facade_alias: dict[str, str] = {}
        facade_imports_needed: set[str] = set()
        facade_import_objects: dict[str, m.Infra.Refactor.MROImportRewrite] = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                module_name = stmt.module
                if module_name is None or module_name not in moved_index:
                    continue
                if any(alias.name == "*" for alias in stmt.names):
                    continue
                kept_names: list[ast.alias] = []
                for alias in stmt.names:
                    default_facade_alias = module_facade_aliases.get(
                        module_name,
                        c.Infra.Refactor.DEFAULT_FACADE_ALIAS,
                    )
                    if alias.name == default_facade_alias:
                        facade_local_name = default_facade_alias
                        facade_aliases[facade_local_name] = module_name
                        module_facade_alias[module_name] = facade_local_name
                        facade_import = m.Infra.Refactor.MROImportRewrite(
                            module=module_name,
                            import_name=default_facade_alias,
                            as_name=None,
                            symbol="",
                            facade_name=facade_local_name,
                        )
                        facade_key = f"{facade_import.module}:{facade_import.import_name}:{facade_import.as_name or ''}"
                        facade_imports_needed.add(facade_key)
                        facade_import_objects[facade_key] = facade_import
                        if alias.asname is None or alias.asname == default_facade_alias:
                            kept_names.append(ast.alias(name=default_facade_alias))
                        continue
                    symbol_map = moved_index[module_name]
                    new_symbol = symbol_map.get(alias.name)
                    if new_symbol is None:
                        kept_names.append(alias)
                        continue
                    imported_symbols[alias.asname or alias.name] = (
                        m.Infra.Refactor.MROImportRewrite(
                            module=module_name,
                            import_name=default_facade_alias,
                            as_name=None,
                            symbol=new_symbol,
                            facade_name=module_facade_alias.get(
                                module_name,
                                default_facade_alias,
                            ),
                        )
                    )
                    facade_import = m.Infra.Refactor.MROImportRewrite(
                        module=module_name,
                        import_name=default_facade_alias,
                        as_name=None
                        if module_name not in module_facade_alias
                        else (
                            None
                            if module_facade_alias[module_name] == default_facade_alias
                            else module_facade_alias[module_name]
                        ),
                        symbol="",
                        facade_name=module_facade_alias.get(
                            module_name,
                            default_facade_alias,
                        ),
                    )
                    facade_key = f"{facade_import.module}:{facade_import.import_name}:{facade_import.as_name or ''}"
                    facade_imports_needed.add(facade_key)
                    facade_import_objects[facade_key] = facade_import
                stmt.names = kept_names
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name in moved_index:
                        module_aliases[alias.asname or alias.name] = alias.name
        rewriter = FlextInfraRefactorMROReferenceRewriter(
            imported_symbols=imported_symbols,
            module_aliases=module_aliases,
            module_facades=facade_aliases,
            moved_index=moved_index,
        )
        rewritten = rewriter.visit(tree)
        if not isinstance(rewritten, ast.Module):
            return None
        rewritten.body = [
            stmt
            for stmt in rewritten.body
            if not (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module in moved_index
                and (len(stmt.names) == 0)
            )
        ]
        existing_imports: set[str] = set()
        for stmt in rewritten.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module is not None:
                for alias in stmt.names:
                    if alias.name != "*":
                        key = f"{stmt.module}:{alias.name}:{alias.asname or ''}"
                        existing_imports.add(key)
        imports_to_add = sorted(facade_imports_needed - existing_imports)
        if len(imports_to_add) > 0:
            insert_at = FlextInfraRefactorMROImportRewriter._import_insertion_index(
                module=rewritten,
            )
            for offset, facade_key in enumerate(imports_to_add):
                facade_import = facade_import_objects[facade_key]
                rewritten.body.insert(
                    insert_at + offset,
                    ast.ImportFrom(
                        module=facade_import.module,
                        names=[
                            ast.alias(
                                name=facade_import.import_name,
                                asname=facade_import.as_name,
                            ),
                        ],
                        level=0,
                    ),
                )
        if rewriter.replacements == 0 and len(imports_to_add) == 0:
            return None
        rendered = ast.unparse(ast.fix_missing_locations(rewritten))
        if apply and rendered != source:
            _ = file_path.write_text(f"{rendered}\n", encoding=c.Infra.Encoding.DEFAULT)
        return m.Infra.Refactor.MRORewriteResult(
            file=str(file_path),
            replacements=rewriter.replacements,
        )

    @staticmethod
    def _iter_workspace_python_files(*, workspace_root: Path) -> list[Path]:
        result = u.Infra.iter_python_files(workspace_root=workspace_root)
        return result.fold(
            on_failure=lambda _: [],
            on_success=lambda v: v,
        )

    @staticmethod
    def _import_insertion_index(*, module: ast.Module) -> int:
        insert_at = 0
        for index, stmt in enumerate(module.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                insert_at = index + 1
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                insert_at = index + 1
        return insert_at


__all__ = [
    "FlextInfraRefactorMROImportRewriter",
    "FlextInfraRefactorMROMigrationScanner",
    "FlextInfraRefactorMROResolver",
]
