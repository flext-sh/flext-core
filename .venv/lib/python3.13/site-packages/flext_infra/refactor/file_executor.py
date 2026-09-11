"""Direct file-rule execution for the refactor service."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m, t, u
from flext_infra.transformers.class_nesting import (
    FlextInfraRefactorClassNestingTransformer,
)
from flext_infra.transformers.helper_consolidation import (
    FlextInfraHelperConsolidationTransformer,
)
from flext_infra.transformers.nested_class_propagation import (
    FlextInfraNestedClassPropagationTransformer,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class FlextInfraClassNestingPostCheckGate:
    """Run post-transform validation gates for direct class-nesting execution."""

    def validate(
        self, result: m.Infra.Result, expected: t.JsonMapping
    ) -> t.Pair[bool, t.StrSequence]:
        """Validate post-check expectations against one transformed file result."""
        if not result.success:
            return (False, [result.error] if result.error else ["transform_failed"])
        if not result.modified:
            return (True, list[str]())
        file_path = result.file_path
        errors: t.MutableSequenceOf[str] = []
        post_checks = u.Infra.string_list(expected.get(c.Infra.RK_POST_CHECKS))
        quality_gates = u.Infra.string_list(expected.get(c.Infra.RK_QUALITY_GATES))
        source_symbol = str(expected.get(c.Infra.RK_SOURCE_SYMBOL, ""))
        expected_chain = u.Infra.string_list(
            expected.get(c.Infra.RK_EXPECTED_BASE_CHAIN)
        )
        if c.Infra.RK_IMPORTS_RESOLVE in post_checks:
            errors.extend(self._validate_imports(file_path))
        if source_symbol and expected_chain and c.Infra.RK_FLEXT_VALID in post_checks:
            errors.extend(
                self._validate_flext(file_path, source_symbol, expected_chain)
            )
        if c.Infra.RK_LSP_DIAGNOSTICS_CLEAN in quality_gates:
            errors.extend(self._validate_types(file_path))
        return (not errors, errors)

    def _validate_imports(self, file_path: Path) -> t.StrSequence:
        """Validate imports."""
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return [f"parse_error:{file_path}:parse_failed"]
        source = read.value
        return [
            f"line_{lineno}:invalid_import_from"
            for lineno, line in enumerate(source.splitlines(), start=1)
            if c.Infra.BARE_IMPORT_FROM_RE.match(line)
        ]

    def _validate_flext(
        self, file_path: Path, class_name: str, expected_bases: t.StrSequence
    ) -> t.StrSequence:
        """Validate flext."""
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return [f"flext_parse_error:{file_path}:parse_failed"]
        actual_clean = list(u.Infra.parse_class_bases(read.value, class_name))
        if not actual_clean:
            return [f"class_not_found:{class_name}"]
        expected_prefix = list(expected_bases)[: len(actual_clean)]
        if actual_clean != expected_prefix:
            return [
                f"flext_mismatch:{class_name}:expected={expected_prefix}:actual={actual_clean}"
            ]
        return list[str]()

    @staticmethod
    def _validate_types(file_path: Path) -> t.StrSequence:
        """Validate types."""
        result = u.Cli.capture([sys.executable, "-m", "py_compile", str(file_path)])
        return (
            [f"lsp_diagnostics_clean_failed:{result.error or ''}"]
            if result.failure
            else list[str]()
        )


class FlextInfraRefactorFileExecutor:
    """Execute declarative Rope-backed file rules directly from kind + settings."""

    _class_nesting_config: t.JsonMapping | None
    _class_nesting_policy_by_family: t.MappingKV[str, m.Infra.ClassNestingPolicy] | None
    _class_nesting_gate: FlextInfraClassNestingPostCheckGate | None

    def _apply_file_rule_selection(
        self,
        kind: c.Infra.RefactorFileRuleKind,
        settings: t.MappingKV[str, t.Infra.InfraValue],
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        dry_run: bool = False,
    ) -> m.Infra.Result:
        """Apply file rule selection."""
        _ = (kind, settings)
        return self._apply_class_nesting(rope_project, resource, dry_run=dry_run)

    def _apply_class_nesting(
        self,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        dry_run: bool = False,
    ) -> m.Infra.Result:
        """Apply class nesting."""
        root_real_path = getattr(getattr(rope_project, "root", None), "real_path", None)
        project_root = Path(root_real_path) if isinstance(root_real_path, str) else None
        file_path = (
            project_root / resource.path
            if project_root is not None
            else Path(resource.real_path)
        )
        try:
            return self._apply_class_nesting_checked(
                resource, file_path, dry_run=dry_run
            )
        except c.EXC_BROAD_IO_TYPE as exc:
            return m.Infra.Result(
                file_path=file_path,
                success=False,
                modified=False,
                error=str(exc),
                changes=[],
                refactored_code=None,
            )

    def _apply_class_nesting_checked(
        self, resource: t.Infra.RopeResource, file_path: Path, *, dry_run: bool
    ) -> m.Infra.Result:
        """Apply class nesting after the public error boundary."""
        source = resource.read()
        config = self._load_class_nesting_config()
        threshold = self._class_nesting_threshold(config)
        class_map = self._class_nesting_symbol_map(
            config,
            file_path,
            threshold,
            c.Infra.RK_CLASS_NESTING,
            c.Infra.RK_LOOSE_NAME,
        )
        helper_map = self._class_nesting_symbol_map(
            config,
            file_path,
            threshold,
            c.Infra.RK_HELPER_CONSOLIDATION,
            c.Infra.RK_HELPER_NAME,
        )
        violations = self._class_nesting_precheck(config, file_path, threshold)
        if violations:
            return m.Infra.Result(
                file_path=file_path,
                success=False,
                modified=False,
                error="precheck_failed",
                changes=violations,
                refactored_code=None,
            )
        changes: t.MutableSequenceOf[str] = []
        updated = self._apply_class_nesting_transforms(
            source, class_map, helper_map, changes
        )
        modified = updated != source
        if modified and not dry_run:
            postcheck_result = self._run_class_nesting_postcheck(
                file_path=file_path, updated=updated, changes=changes
            )
            if postcheck_result is not None:
                return postcheck_result
            resource.write(updated)
        return m.Infra.Result(
            file_path=file_path,
            success=True,
            modified=modified,
            changes=changes,
            refactored_code=updated,
        )

    def _run_class_nesting_postcheck(
        self, *, file_path: Path, updated: str, changes: t.StrSequence
    ) -> m.Infra.Result | None:
        """Run postchecks for a modified class-nesting result."""
        expected_base_chain: t.JsonValueList = []
        post_checks: t.JsonValueList = [c.Infra.RK_IMPORTS_RESOLVE]
        quality_gates: t.JsonValueList = [c.Infra.RK_LSP_DIAGNOSTICS_CLEAN]
        payload_values: t.JsonMapping = {
            c.Infra.RK_SOURCE_SYMBOL: "",
            c.Infra.RK_EXPECTED_BASE_CHAIN: expected_base_chain,
            c.Infra.RK_POST_CHECKS: post_checks,
            c.Infra.RK_QUALITY_GATES: quality_gates,
        }
        payload = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(payload_values)
        gate = self._class_nesting_gate or FlextInfraClassNestingPostCheckGate()
        self._class_nesting_gate = gate
        ok, errs = gate.validate(
            m.Infra.Result(
                file_path=file_path,
                success=True,
                modified=True,
                changes=changes,
                refactored_code=updated,
            ),
            payload,
        )
        if ok:
            return None
        return m.Infra.Result(
            file_path=file_path,
            success=False,
            modified=False,
            error="postcheck_failed",
            changes=errs,
            refactored_code=None,
        )

    def _load_class_nesting_config(self) -> t.JsonMapping:
        """Load class nesting config."""
        if self._class_nesting_config is not None:
            return self._class_nesting_config
        rules_dir = Path(__file__).resolve().parent.parent / c.Infra.RK_RULES
        loaded = u.Cli.yaml_load_mapping(
            rules_dir / c.Infra.CLASS_NESTING_MAPPINGS_FILENAME
        )
        settings: MutableMapping[str, t.Infra.InfraValue] = {}
        threshold = loaded.get(c.Infra.RK_CONFIDENCE_THRESHOLD)
        if isinstance(threshold, str):
            settings[c.Infra.RK_CONFIDENCE_THRESHOLD] = threshold
        for key in c.Infra.NESTING_SECTION_KEYS:
            raw = loaded.get(key)
            if isinstance(raw, list):
                settings[key] = [
                    dict(entry)
                    for entry in self._coerce_class_nesting_entries(
                        u.Cli.json_as_mapping_list(raw)
                    )
                ]
        self._class_nesting_config = settings
        return settings

    def _class_nesting_policy(self) -> t.MappingKV[str, m.Infra.ClassNestingPolicy]:
        """Class nesting policy."""
        if self._class_nesting_policy_by_family is None:
            rules_dir = Path(__file__).resolve().parent.parent / c.Infra.RK_RULES
            self._class_nesting_policy_by_family = (
                u.Infra.class_nesting_policy_by_family(
                    rules_dir / c.Infra.CLASS_NESTING_POLICY_FILENAME
                )
            )
        return self._class_nesting_policy_by_family

    def _class_nesting_precheck(
        self, config: t.JsonMapping, file_path: Path, threshold: str
    ) -> t.MutableSequenceOf[str]:
        """Class nesting precheck."""
        entries: t.MutableSequenceOf[t.StrMapping] = []
        for key in c.Infra.NESTING_SECTION_KEYS:
            entries.extend(
                self._filter_class_nesting_entries(
                    u.Infra.entry_list(config.get(key)), file_path, threshold
                )
            )
        violations: t.MutableSequenceOf[str] = []
        for entry in entries:
            ok, violation = u.Infra.validate_class_nesting_entry(
                entry, policy_by_family=self._class_nesting_policy()
            )
            if not ok and violation is not None:
                violation_parts: list[str] = [
                    violation[c.Infra.RK_RULE_ID],
                    violation[c.Infra.RK_SOURCE_SYMBOL],
                    violation[c.Infra.RK_VIOLATION_TYPE],
                    violation[c.Infra.RK_SUGGESTED_FIX],
                ]
                violations.append("|".join(violation_parts))
        return violations

    def _class_nesting_symbol_map(
        self,
        config: t.JsonMapping,
        file_path: Path,
        threshold: str,
        section: str,
        name_key: str,
    ) -> t.MutableStrMapping:
        """Class nesting symbol map."""
        result: t.MutableStrMapping = {}
        for entry in self._filter_class_nesting_entries(
            u.Infra.entry_list(config.get(section)), file_path, threshold
        ):
            name = entry.get(name_key)
            target = entry.get(c.Infra.RK_TARGET_NAMESPACE)
            if isinstance(name, str) and isinstance(target, str):
                result[name] = target
        return result

    def _filter_class_nesting_entries(
        self, raw: t.SequenceOf[t.StrMapping], file_path: Path, threshold: str
    ) -> t.SequenceOf[t.StrMapping]:
        """Filter class nesting entries."""
        module_path = u.Infra.normalize_module_path(file_path)
        result: t.MutableSequenceOf[t.StrMapping] = []
        for entry in raw:
            current_file = entry.get(c.Infra.RK_CURRENT_FILE)
            if not isinstance(current_file, str):
                continue
            current_module = u.Infra.normalize_module_path(Path(current_file))
            if current_module != module_path and not module_path.endswith(
                f"/{current_module}"
            ):
                continue
            confidence = entry.get(c.Infra.RK_CONFIDENCE, c.Infra.SeverityLevel.LOW)
            if c.Infra.CONFIDENCE_RANKS.get(
                u.norm_str(confidence, case="lower"), 0
            ) < c.Infra.CONFIDENCE_RANKS.get(threshold, 0):
                continue
            result.append(entry)
        return result

    def _apply_class_nesting_transforms(
        self,
        source: str,
        class_map: t.MutableStrMapping,
        helper_map: t.MutableStrMapping,
        changes: t.MutableSequenceOf[str],
    ) -> str:
        """Apply class nesting transforms."""
        updated = source
        if class_map:
            nesting = FlextInfraRefactorClassNestingTransformer(class_map, {}, {})
            updated, class_changes = nesting.apply_to_source(updated)
            changes.extend(class_changes)
            propagation_map: t.MutableStrMapping = {
                name: f"{target}.{name}" for name, target in class_map.items()
            }
            propagation = FlextInfraNestedClassPropagationTransformer(propagation_map)
            updated, propagation_changes = propagation.apply_to_source(updated)
            changes.extend(propagation_changes)
        if helper_map:
            consolidation = FlextInfraHelperConsolidationTransformer(helper_map)
            updated, helper_changes = consolidation.apply_to_source(updated)
            changes.extend(helper_changes)
        return updated

    @staticmethod
    def _coerce_class_nesting_entries(
        entries: t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]],
    ) -> t.SequenceOf[t.StrMapping]:
        """Coerce class nesting entries."""
        result: t.MutableSequenceOf[t.StrMapping] = []
        for typed in entries:
            current_file = typed.get(c.Infra.RK_CURRENT_FILE)
            if not isinstance(current_file, str):
                continue
            entry: t.MutableStrMapping = {c.Infra.RK_CURRENT_FILE: current_file}
            for key in c.Infra.NESTING_COERCE_KEYS:
                value = typed.get(key)
                if isinstance(value, str):
                    entry[key] = value
            result.append(entry)
        return result

    @staticmethod
    def _class_nesting_threshold(settings: t.JsonMapping) -> str:
        """Class nesting threshold."""
        raw = settings.get(c.Infra.RK_CONFIDENCE_THRESHOLD, c.Infra.SeverityLevel.LOW)
        if not isinstance(raw, str):
            msg = "confidence_threshold must be a string"
            raise TypeError(msg)
        candidate: str = u.norm_str(raw, case="lower")
        if candidate not in c.Infra.CONFIDENCE_RANKS:
            msg = f"unsupported confidence_threshold: {raw}"
            raise ValueError(msg)
        return candidate


__all__: list[str] = [
    "FlextInfraClassNestingPostCheckGate",
    "FlextInfraRefactorFileExecutor",
]
