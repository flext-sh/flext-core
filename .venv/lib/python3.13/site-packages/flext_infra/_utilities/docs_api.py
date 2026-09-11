"""Public API extraction helpers for code-driven documentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraUtilitiesDocsApi:
    """Extract public package metadata, exports, modules, and docstring issues."""

    _ALIAS_EXPORTS: t.StrSequence = (
        "c",
        "d",
        "e",
        "h",
        "m",
        "p",
        "r",
        "s",
        "t",
        "u",
        "x",
    )

    @staticmethod
    def _string_values(value: t.Infra.InfraValue | None) -> t.StrSequence:
        """Normalize one infra sequence payload into strings."""
        try:
            items = t.Infra.INFRA_SEQ_ADAPTER.validate_python(value)
        except c.ValidationError:
            return []
        return [str(item) for item in items]

    @staticmethod
    def _string_mapping(value: t.Infra.InfraValue | None) -> t.StrMapping:
        """Normalize one infra mapping payload into string keys and values."""
        try:
            items = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(value)
        except c.ValidationError:
            return {}
        normalized_items: dict[str, str] = {
            key: str(entry) for key, entry in items.items()
        }
        return normalized_items

    @staticmethod
    def _module_file(project_root: Path, module_name: str) -> Path:
        """Resolve a Python module path to its file path."""
        parts = module_name.split(".")
        src_dir: str = c.Infra.DEFAULT_SRC_DIR
        init_py: str = c.Infra.INIT_PY
        ext_python: str = c.Infra.EXT_PYTHON
        base = project_root / src_dir / parts[0]
        if len(parts) == 1:
            return base / init_py
        module_file = base.joinpath(*parts[1:]).with_suffix(ext_python)
        if module_file.exists():
            return module_file
        package_init = base.joinpath(*parts[1:]) / init_py
        return package_init if package_init.exists() else module_file

    @classmethod
    def _assignment_strings(cls, source: str, name: str) -> t.StrSequence:
        """Collect literal string values from one module-level assignment."""
        return FlextInfraUtilitiesRopeAnalysis.module_assignment_strings_source(
            source, name
        )

    @classmethod
    def _imported_symbol_binding(
        cls, source: str, *, current_module: str, symbol_name: str, package_module: bool
    ) -> tuple[str, str]:
        """Return the source module and original name for one imported symbol."""
        return FlextInfraUtilitiesRopeAnalysis.imported_symbol_binding_source(
            source,
            current_module=current_module,
            symbol_name=symbol_name,
            package_module=package_module,
        )

    @classmethod
    def _resolve_assignment_strings(
        cls,
        project_root: Path,
        *,
        module_name: str,
        symbol_name: str,
        visited: frozenset[str] = frozenset(),
    ) -> t.StrSequence:
        """Resolve a literal string assignment through imported symbols."""
        key = f"{module_name}:{symbol_name}"
        if key in visited:
            return []
        module_file = cls._module_file(project_root, module_name)
        if not module_file.exists():
            return []
        source = module_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        values = cls._assignment_strings(source, symbol_name)
        if values:
            return values
        # flext-o6h5 (agent: kimi): resolve through import aliases to the
        # ORIGINAL symbol — lazy __unit__ contracts bind PUBLIC_EXPORTS as
        # _PUBLIC_EXPORTS; resolving the alias in the target module yielded
        # an empty contract (root cause of the flext-infra validate FAIL).
        imported_module, original_name = cls._imported_symbol_binding(
            source,
            current_module=module_name,
            symbol_name=symbol_name,
            package_module=module_file.name == c.Infra.INIT_PY,
        )
        if not imported_module:
            return []
        return cls._resolve_assignment_strings(
            project_root,
            module_name=imported_module,
            symbol_name=original_name,
            visited=visited | frozenset({key}),
        )

    @staticmethod
    def _resolve_lazy_module_name(package_name: str, module_name: str) -> str:
        """Resolve a lazy import module string against the root package."""
        if module_name.startswith("."):
            return f"{package_name}{module_name}"
        return module_name

    @classmethod
    def _resolve_lazy_import_targets(
        cls,
        project_root: Path,
        *,
        root_package: str,
        module_name: str,
        symbol_name: str,
        visited: frozenset[str] = frozenset(),
    ) -> t.StrMapping:
        """Resolve one lazy-import map symbol into export target modules."""
        key = f"{module_name}:{symbol_name}"
        if key in visited:
            return {}
        module_file = cls._module_file(project_root, module_name)
        if not module_file.exists():
            return {}
        source = module_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        entries, refs = (
            FlextInfraUtilitiesRopeAnalysis.module_mapping_assignment_source(
                source, symbol_name
            )
        )
        next_visited = visited | frozenset({key})
        if not entries and not refs:
            imported_module, imported_symbol = cls._imported_symbol_binding(
                source,
                current_module=module_name,
                symbol_name=symbol_name,
                package_module=module_file.name == c.Infra.INIT_PY,
            )
            if not imported_module:
                return {}
            return cls._resolve_lazy_import_targets(
                project_root,
                root_package=root_package,
                module_name=imported_module,
                symbol_name=imported_symbol or symbol_name,
                visited=next_visited,
            )
        targets: dict[str, str] = {}
        for target_module, export_names in entries:
            resolved_module = cls._resolve_lazy_module_name(root_package, target_module)
            for export_name in export_names:
                targets[export_name] = resolved_module
        for ref_name in refs:
            local_targets = cls._resolve_lazy_import_targets(
                project_root,
                root_package=root_package,
                module_name=module_name,
                symbol_name=ref_name,
                visited=next_visited,
            )
            if local_targets:
                targets.update(local_targets)
                continue
            imported_module, imported_symbol = cls._imported_symbol_binding(
                source,
                current_module=module_name,
                symbol_name=ref_name,
                package_module=module_file.name == c.Infra.INIT_PY,
            )
            if imported_module:
                targets.update(
                    cls._resolve_lazy_import_targets(
                        project_root,
                        root_package=root_package,
                        module_name=imported_module,
                        symbol_name=imported_symbol or ref_name,
                        visited=next_visited,
                    )
                )
        return targets

    @classmethod
    def _lazy_export_target_map(
        cls,
        project_root: Path,
        *,
        package_name: str,
        source: str,
        exports: t.StrSequence,
    ) -> t.StrMapping:
        """Return target modules declared by the package lazy import map."""
        lazy_imports_name = FlextInfraUtilitiesRopeAnalysis.lazy_imports_name_source(
            source
        )
        if not lazy_imports_name:
            return {}
        target_map = cls._resolve_lazy_import_targets(
            project_root,
            root_package=package_name,
            module_name=package_name,
            symbol_name=lazy_imports_name,
        )
        export_names = frozenset(exports)
        return {
            export_name: module_name
            for export_name, module_name in target_map.items()
            if export_name in export_names
        }

    @classmethod
    def _public_export_strings(
        cls, project_root: Path, package_name: str, source: str
    ) -> t.StrSequence:
        """Return runtime public exports from lazy-loader or ``__all__`` contracts."""
        literal_values, export_name = (
            FlextInfraUtilitiesRopeAnalysis.lazy_public_exports_source(source)
        )
        if literal_values:
            return literal_values
        if export_name:
            local_values = cls._assignment_strings(source, export_name)
            if local_values:
                return local_values
            # flext-o6h5 (agent: kimi): alias-aware binding — see
            # _resolve_assignment_strings for the root-cause note.
            imported_module, original_name = cls._imported_symbol_binding(
                source,
                current_module=package_name,
                symbol_name=export_name,
                package_module=True,
            )
            if imported_module:
                return cls._resolve_assignment_strings(
                    project_root, module_name=imported_module, symbol_name=original_name
                )
        return cls._assignment_strings(source, "__all__")

    @staticmethod
    def _export_target_map(
        source: str, package_name: str, exports: t.StrSequence
    ) -> t.StrMapping:
        """Resolve exported symbols to their defining import modules when possible."""
        return FlextInfraUtilitiesRopeAnalysis.export_target_modules_source(
            source, package_name, exports
        )

    @staticmethod
    def _has_module_docstring(source: str) -> bool:
        """Return whether source starts with a module docstring."""
        return FlextInfraUtilitiesRopeAnalysis.module_has_docstring_source(source)

    @staticmethod
    def _assignment_docstrings(source: str) -> set[str]:
        """Return assignment names followed by a literal docstring expression."""
        return set(FlextInfraUtilitiesRopeAnalysis.assignment_docstrings_source(source))

    @staticmethod
    def _has_symbol_docstring(source: str, symbol_name: str) -> bool:
        """Return whether one exported class/function starts with a docstring."""
        if FlextInfraUtilitiesRopeAnalysis.symbol_has_docstring_source(
            source, symbol_name
        ):
            return True
        return symbol_name in FlextInfraUtilitiesDocsApi._assignment_docstrings(source)

    @classmethod
    def _has_flext_docstring(
        cls,
        project_root: Path,
        *,
        module_name: str,
        source: str,
        symbol_name: str,
        visited: frozenset[str],
    ) -> bool:
        """Return whether one class inherits documentation through its FLEXT chain."""
        if not FlextInfraUtilitiesRopeAnalysis.class_declared_source(
            source, symbol_name
        ):
            return False
        for base_name in FlextInfraUtilitiesRopeAnalysis.class_bases_source(
            source, symbol_name
        ):
            base_symbol = base_name.split(".")[-1]
            if not base_symbol or base_symbol in {"ABC", "Generic", "Protocol", "Self"}:
                continue
            imported_module, imported_symbol = cls._imported_symbol_binding(
                source,
                current_module=module_name,
                symbol_name=base_symbol,
                package_module=cls._module_file(project_root, module_name).name
                == c.Infra.INIT_PY,
            )
            if imported_module:
                if cls._has_exported_symbol_docstring(
                    project_root,
                    module_name=imported_module,
                    symbol_name=imported_symbol or base_symbol,
                    visited=visited,
                ):
                    return True
                continue
            if cls._has_exported_symbol_docstring(
                project_root,
                module_name=module_name,
                symbol_name=base_symbol,
                visited=visited,
            ):
                return True
        return False

    @classmethod
    def _has_exported_symbol_docstring(
        cls,
        project_root: Path,
        *,
        module_name: str,
        symbol_name: str,
        visited: frozenset[str] = frozenset(),
    ) -> bool:
        """Return whether an exported symbol or its import target is documented."""
        key = f"{module_name}:{symbol_name}"
        if key in visited:
            return False
        module_file = cls._module_file(project_root, module_name)
        if not module_file.exists():
            return False
        source = module_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        if cls._has_symbol_docstring(source, symbol_name):
            return True
        if cls._has_flext_docstring(
            project_root,
            module_name=module_name,
            source=source,
            symbol_name=symbol_name,
            visited=visited | frozenset({key}),
        ):
            return True
        imported_module, imported_symbol = cls._imported_symbol_binding(
            source,
            current_module=module_name,
            symbol_name=symbol_name,
            package_module=module_file.name == c.Infra.INIT_PY,
        )
        if not imported_module:
            return False
        return cls._has_exported_symbol_docstring(
            project_root,
            module_name=imported_module,
            symbol_name=imported_symbol or symbol_name,
            visited=visited | frozenset({key}),
        )

    @staticmethod
    def _rope_public_symbols(
        project_root: Path, target_map: t.StrMapping
    ) -> t.StrSequence:
        """Use Rope to verify which exported symbols resolve in real modules."""
        with FlextInfraUtilitiesRopeCore.open_project(project_root) as rope_project:
            symbols: t.MutableSequenceOf[str] = []
            for export_name, module_name in target_map.items():
                module_file = FlextInfraUtilitiesDocsApi._module_file(
                    project_root, module_name
                )
                if not module_file.exists():
                    continue
                resource = FlextInfraUtilitiesRopeCore.get_resource_from_path(
                    rope_project, module_file
                )
                if resource is None:
                    continue
                pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(
                    rope_project, resource
                )
                if export_name in pymodule.get_attributes():
                    symbols.append(export_name)
            return tuple(dict.fromkeys(symbols))

    @staticmethod
    def public_contract(project_root: Path, package_name: str) -> t.JsonMapping:
        """Build the public API contract from pyproject, exports, and Rope validation."""
        # flext-j47u: retain flext-core's validated metadata object; no shadow DTO.
        metadata_result = u.read_project_metadata(project_root)
        if metadata_result.failure:
            msg = (
                metadata_result.error or f"project metadata unavailable: {project_root}"
            )
            raise ValueError(msg)
        metadata = metadata_result.value
        project = metadata.project
        docs = metadata.flext.docs
        site_title = docs.site_title or project.name
        site_url = project.urls.documentation or project.urls.homepage
        repo_url = project.urls.repository or project.urls.homepage
        if not package_name:
            result: t.JsonMapping = t.Infra.INFRA_MAPPING_ADAPTER.validate_python({
                "package_name": "",
                "description": project.description,
                "doc_summary": "",
                "classifiers": list(project.classifiers),
                "version": project.version,
                "site_title": site_title,
                "site_url": site_url,
                "repo_url": repo_url,
                "exports": [],
                "aliases": [],
                "facades": [],
                "public_symbols": [],
                "target_map": {},
                "modules": [],
                "exclude_docs": list(docs.exclude_docs),
            })
            return result
        init_path = (
            project_root / c.Infra.DEFAULT_SRC_DIR / package_name / c.Infra.INIT_PY
        )
        if not init_path.exists():
            return FlextInfraUtilitiesDocsApi.public_contract(project_root, "")
        source = init_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        all_exports = list(
            FlextInfraUtilitiesDocsApi._public_export_strings(
                project_root, package_name, source
            )
        )
        target_map = dict(
            FlextInfraUtilitiesDocsApi._export_target_map(
                source, package_name, all_exports
            )
        )
        target_map.update(
            FlextInfraUtilitiesDocsApi._lazy_export_target_map(
                project_root,
                package_name=package_name,
                source=source,
                exports=all_exports,
            )
        )
        aliases, module_exports, symbol_exports = (
            FlextInfraUtilitiesDocsApi._classify_exports(all_exports, target_map)
        )
        modules = FlextInfraUtilitiesDocsApi._resolve_modules(
            package_name=package_name, target_map=target_map
        )
        rope_symbols = frozenset(
            FlextInfraUtilitiesDocsApi._rope_public_symbols(project_root, target_map)
        )
        public_symbols = [
            name for name in symbol_exports if name in rope_symbols
        ] or symbol_exports
        facades = [
            name for name in public_symbols if name.startswith(metadata.class_stem)
        ]
        contract: t.JsonMapping = t.Infra.INFRA_MAPPING_ADAPTER.validate_python({
            "package_name": package_name,
            "description": project.description,
            "doc_summary": FlextInfraUtilitiesRopeAnalysis.module_docstring_summary_source(
                source
            ),
            "classifiers": list(project.classifiers),
            "keywords": list(project.keywords),
            "version": project.version,
            "site_title": site_title,
            "site_url": site_url,
            "repo_url": repo_url,
            "exports": all_exports,
            "aliases": aliases,
            "facades": facades,
            "module_exports": module_exports,
            "public_symbols": public_symbols,
            "target_map": dict(target_map),
            "modules": modules,
            "exclude_docs": list(docs.exclude_docs),
        })
        return contract

    @staticmethod
    def _classify_exports(
        all_exports: t.StrSequence, target_map: t.StrMapping
    ) -> tuple[list[str], list[str], list[str]]:
        """Split ``__all__`` entries into ``(aliases, module_exports, symbol_exports)``."""
        aliases = [
            name
            for name in all_exports
            if name in FlextInfraUtilitiesDocsApi._ALIAS_EXPORTS
        ]
        module_exports = [
            name
            for name in all_exports
            if not name.startswith("_")
            and name not in FlextInfraUtilitiesDocsApi._ALIAS_EXPORTS
            and target_map.get(name, "").split(".")[-1] == name
            and name[:1].islower()
        ]
        symbol_exports = [
            name
            for name in all_exports
            if name not in FlextInfraUtilitiesDocsApi._ALIAS_EXPORTS
            and not name.startswith("_")
            and name not in module_exports
        ]
        return aliases, module_exports, symbol_exports

    @staticmethod
    def _resolve_modules(*, package_name: str, target_map: t.StrMapping) -> list[str]:
        """Compute doc-eligible modules from the resolved public export graph."""
        return sorted({
            module
            for module in target_map.values()
            if module.startswith(package_name)
            and "._" not in module
            and not module.endswith(".__version__")
        })

    @staticmethod
    def _iter_docstring_checks(
        project_root: Path, contract: t.JsonMapping
    ) -> t.SequenceOf[tuple[str, str, bool]]:
        """Evaluate every public docstring target once (SSOT).

        Yields ``(rel_file, missing_message, documented)`` for the package
        module, each public module, and each exported symbol — shared by
        ``docstring_issues`` (undocumented targets) and
        ``docstring_coverage`` (aggregate metric) so the target set and the
        docstring predicates can never drift apart.
        """
        package_name = str(contract.get("package_name", ""))
        module_list = FlextInfraUtilitiesDocsApi._string_values(
            contract.get("modules", [])
        )
        target_map = FlextInfraUtilitiesDocsApi._string_mapping(
            contract.get("target_map", {})
        )
        module_exports = set(
            FlextInfraUtilitiesDocsApi._string_values(
                contract.get("module_exports", [])
            )
        )
        results: t.MutableSequenceOf[tuple[str, str, bool]] = []
        module_docstring_checks = [
            (module_name, f"public module `{module_name}` is missing a docstring")
            for module_name in module_list
        ]
        if package_name:
            module_docstring_checks.insert(
                0, (package_name, "package module is missing a docstring")
            )
        for module_name, message in module_docstring_checks:
            module_file = FlextInfraUtilitiesDocsApi._module_file(
                project_root, module_name
            )
            if not module_file.exists():
                continue
            source = module_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            results.append((
                module_file.relative_to(project_root).as_posix(),
                message,
                FlextInfraUtilitiesDocsApi._has_module_docstring(source),
            ))
        export_docstring_checks = [
            (export_name, module_name)
            for export_name, module_name in target_map.items()
            if export_name not in FlextInfraUtilitiesDocsApi._ALIAS_EXPORTS
            and not export_name.startswith("__")
            and export_name not in module_exports
        ]
        for export_name, module_name in export_docstring_checks:
            module_file = FlextInfraUtilitiesDocsApi._module_file(
                project_root, module_name
            )
            if not module_file.exists():
                continue
            results.append((
                module_file.relative_to(project_root).as_posix(),
                f"exported symbol `{export_name}` is missing a docstring",
                FlextInfraUtilitiesDocsApi._has_exported_symbol_docstring(
                    project_root, module_name=module_name, symbol_name=export_name
                ),
            ))
        return tuple(results)

    @staticmethod
    def docstring_issues(
        project_root: Path, contract: t.JsonMapping
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Return audit issues for public modules and exports missing docstrings."""
        return [
            m.Infra.AuditIssue(
                file=rel_file,
                issue_type="missing_docstring",
                severity="medium",
                message=message,
            )
            for rel_file, message, documented in (
                FlextInfraUtilitiesDocsApi._iter_docstring_checks(
                    project_root, contract
                )
            )
            if not documented
        ]

    @staticmethod
    def docstring_coverage(
        project_root: Path, contract: t.JsonMapping
    ) -> m.Infra.DocstringCoverage:
        """Aggregate docstring coverage over every public target.

        ``percent`` is computed here (behavior lives in ``u``); the model is a
        declaration-only payload.
        """
        checks = FlextInfraUtilitiesDocsApi._iter_docstring_checks(
            project_root, contract
        )
        checked = len(checks)
        documented = sum(1 for *_head, documented in checks if documented)
        percent = 100.0 if checked == 0 else round(100.0 * documented / checked, 1)
        return m.Infra.DocstringCoverage(
            checked=checked, documented=documented, percent=percent
        )


__all__: list[str] = ["FlextInfraUtilitiesDocsApi"]
