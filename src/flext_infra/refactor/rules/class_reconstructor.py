"""Rule that reorders class methods based on configured method order."""

from __future__ import annotations

from pathlib import Path
from typing import override

import libcst as cst
from libcst.metadata import MetadataWrapper
from pydantic import JsonValue, TypeAdapter, ValidationError

from flext_infra import c, m, t, u
from flext_infra.refactor._base_rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.class_reconstructor import (
    FlextInfraRefactorClassReconstructor,
)
from flext_infra.refactor.transformers.nested_class_propagation import (
    NestedClassPropagationTransformer,
)


class PreCheckGate:
    """Gate that validates class nesting entries against a YAML policy matrix."""

    def __init__(self, policy_path: Path | None = None) -> None:
        """Initialize with optional custom policy path."""
        self._policy_path = policy_path or Path(__file__).with_name(
            "class-policy-v2.yml"
        )
        self._schema_path = self._policy_path.with_name("class-policy-v2.schema.json")
        self._policy_by_family: dict[str, m.Infra.Refactor.ClassNestingPolicy] = (
            self._load_policy()
        )

    def validate_entry(
        self,
        entry: t.Infra.StrMap,
    ) -> tuple[bool, t.Infra.StrMap | None]:
        """Validate a single class-nesting entry against the loaded policy."""
        source_symbol = entry.get(c.Infra.ReportKeys.LOOSE_NAME, "")
        helper_symbol = entry.get("helper_name", "")
        symbol = source_symbol or helper_symbol
        target_namespace = entry.get(c.Infra.ReportKeys.TARGET_NAMESPACE, "")
        current_file = entry.get(c.Infra.ReportKeys.CURRENT_FILE, "")
        if not symbol or not target_namespace or (not current_file):
            return (True, None)
        module_family = u.Infra.module_family_from_path(current_file)
        if module_family == "other_private":
            return (True, None)
        policy = self._policy_by_family.get(module_family)
        if policy is None:
            return (
                False,
                {
                    c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                    c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                    c.Infra.ReportKeys.VIOLATION_TYPE: "unknown_module_family",
                    c.Infra.ReportKeys.SUGGESTED_FIX: f"declare explicit policy for {module_family}",
                },
            )
        operation = (
            c.Infra.ReportKeys.HELPER_CONSOLIDATION
            if helper_symbol
            else c.Infra.ReportKeys.CLASS_NESTING
        )
        if operation not in policy.allowed_operations:
            return (
                False,
                {
                    c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                    c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                    c.Infra.ReportKeys.VIOLATION_TYPE: "operation_not_allowed",
                    c.Infra.ReportKeys.SUGGESTED_FIX: f"allow {operation} in policy for {module_family}",
                },
            )
        if operation in policy.forbidden_operations:
            return (
                False,
                {
                    c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                    c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                    c.Infra.ReportKeys.VIOLATION_TYPE: "operation_forbidden",
                    c.Infra.ReportKeys.SUGGESTED_FIX: f"remove {operation} from forbidden_operations for {module_family}",
                },
            )
        if any(
            self._target_matches(target_namespace, pattern)
            for pattern in policy.forbidden_targets
        ):
            return (
                False,
                {
                    c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                    c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                    c.Infra.ReportKeys.VIOLATION_TYPE: "forbidden_target",
                    c.Infra.ReportKeys.SUGGESTED_FIX: f"choose allowed target for family {module_family}",
                },
            )
        return (True, None)

    def _load_policy(self) -> dict[str, m.Infra.Refactor.ClassNestingPolicy]:
        try:
            loaded = u.Infra.safe_load_yaml(self._policy_path)
        except (OSError, TypeError):
            return {}
        loaded_dict: dict[str, t.Infra.InfraValue] = TypeAdapter(
            dict[str, t.Infra.InfraValue]
        ).validate_python(dict(loaded.items()))
        if not self._schema_valid(loaded_dict):
            return {}
        policy_matrix = u.Infra.mapping_list(loaded_dict.get("policy_matrix"))
        by_family: dict[str, m.Infra.Refactor.ClassNestingPolicy] = {}
        for raw in policy_matrix:
            try:
                policy = m.Infra.Refactor.ClassNestingPolicy.model_validate(raw)
            except ValidationError:
                continue
            by_family[policy.family_name] = policy
        return by_family

    def _schema_valid(self, loaded: dict[str, t.Infra.InfraValue]) -> bool:
        schema_result = u.Infra.read_json(self._schema_path)
        if schema_result.is_failure:
            return True
        schema: dict[str, JsonValue] = dict(schema_result.value.items())
        top_required = u.Infra.string_list(schema.get("required"))
        if not u.Infra.has_required_fields(loaded, top_required):
            return False
        definitions_raw: JsonValue | None = schema.get("definitions")
        if not isinstance(definitions_raw, dict):
            return False
        policy_entry_raw: JsonValue | None = definitions_raw.get("policyEntry")
        class_rule_raw: JsonValue | None = definitions_raw.get("classRule")
        if not isinstance(policy_entry_raw, dict):
            return False
        if not isinstance(class_rule_raw, dict):
            return False
        policy_entry_required = u.Infra.string_list(policy_entry_raw.get("required"))
        class_rule_required = u.Infra.string_list(class_rule_raw.get("required"))
        policy_matrix = u.Infra.mapping_list(loaded.get("policy_matrix"))
        for entry in policy_matrix:
            if not u.Infra.has_required_fields(entry, policy_entry_required):
                return False
        rules = u.Infra.mapping_list(loaded.get(c.Infra.ReportKeys.RULES))
        for rule in rules:
            if not u.Infra.has_required_fields(rule, class_rule_required):
                return False
        return True

    def _target_matches(self, target_namespace: str, pattern: str) -> bool:
        if pattern.endswith(".*"):
            return target_namespace.lower().startswith(pattern[:-2].lower())
        return target_namespace == pattern


class FlextInfraRefactorClassNestingReconstructor:
    """Reconstruct class nesting by applying rename mappings and propagation."""

    @staticmethod
    def class_rename_mappings(entries: list[t.Infra.StrMap]) -> dict[str, str]:
        """Build a mapping of loose class names to their nested target names."""
        mappings: dict[str, str] = {}
        for entry in entries:
            loose_name = entry.get(c.Infra.ReportKeys.LOOSE_NAME)
            target_namespace = entry.get(c.Infra.ReportKeys.TARGET_NAMESPACE)
            target_name = entry.get("target_name")
            if not isinstance(loose_name, str):
                continue
            if not isinstance(target_namespace, str):
                continue
            if not isinstance(target_name, str):
                continue
            mappings[loose_name] = f"{target_namespace}.{target_name}"
        return mappings

    @staticmethod
    def apply_nested_class_propagation(
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
        policy_context: t.Infra.PolicyContext,
        class_families: t.Infra.ClassFamilyMap,
    ) -> cst.Module:
        """Apply nested class propagation transforms using the given rename mappings."""
        transformer = NestedClassPropagationTransformer(
            class_renames=mappings,
            policy_context=policy_context,
            class_families=class_families,
        )
        wrapped_tree = MetadataWrapper(tree)
        updated_tree = wrapped_tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied NestedClassPropagationTransformer ({len(mappings)} renames)",
            )
        return updated_tree


class FlextInfraRefactorClassReconstructorRule(FlextInfraRefactorRule):
    """Apply class method ordering reconstruction to matching class nodes."""

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Apply method reordering transformer when order config is available."""
        order_config_raw = self.config.get("method_order") or self.config.get(
            "order",
            [],
        )
        try:
            order_config = TypeAdapter(list[t.Infra.RuleConfig]).validate_python(
                order_config_raw,
            )
        except ValidationError:
            return (tree, [])
        if not order_config:
            return (tree, [])
        transformer = FlextInfraRefactorClassReconstructor(order_config=order_config)
        new_tree = tree.visit(transformer)
        return (new_tree, transformer.changes)


__all__ = [
    "FlextInfraRefactorClassNestingReconstructor",
    "FlextInfraRefactorClassReconstructorRule",
    "PreCheckGate",
]
