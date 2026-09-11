"""Class-nesting policy validation utilities for the refactor subpackage.

Handles policy document loading, schema validation, and class nesting
policy enforcement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_cli import u
from flext_infra import c, m, p, r, t


class FlextInfraUtilitiesRefactorPolicy:
    """Policy document loading and class-nesting policy enforcement."""

    _MODULE_FAMILY_KEYS: t.StrSequence = (
        "models",
        "_utilities",
        "_dispatcher",
        "_decorators",
        "_runtime",
    )

    @staticmethod
    def default_class_policy_path() -> Path:
        """Return the default class-nesting policy YAML path."""
        return Path(__file__).resolve().parent.parent / "rules" / "class-policy-v2.yml"

    @staticmethod
    def load_validated_policy_document(
        policy_path: Path,
    ) -> p.Result[t.MappingKV[str, t.Infra.InfraValue]]:
        """Load and validate a YAML policy document."""
        raw = u.Cli.yaml_load_mapping(policy_path)
        if not raw:
            return r[t.MappingKV[str, t.Infra.InfraValue]].fail(
                f"Failed to load policy {policy_path}"
            )
        validated = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(raw)
        return r[t.MappingKV[str, t.Infra.InfraValue]].ok(validated)

    @staticmethod
    def class_nesting_policy_by_family(
        policy_path: Path | None = None,
    ) -> t.MappingKV[str, m.Infra.ClassNestingPolicy]:
        """Load the class-nesting policy matrix keyed by module family."""
        resolved_path = (
            policy_path
            if policy_path is not None
            else FlextInfraUtilitiesRefactorPolicy.default_class_policy_path()
        )
        loaded = FlextInfraUtilitiesRefactorPolicy.load_validated_policy_document(
            resolved_path
        )
        by_family: dict[str, m.Infra.ClassNestingPolicy] = {}
        if loaded.failure:
            return by_family
        for raw in u.Cli.json_as_mapping_list(loaded.value.get("policy_matrix")):
            policy = m.Infra.ClassNestingPolicy.model_validate(raw)
            by_family[policy.family_name] = policy
        return by_family

    @staticmethod
    def module_family_from_path(path: str) -> str:
        """Resolve module family key from a source file path."""
        normalized = path.replace("\\", "/")
        return next(
            (
                key
                for key in FlextInfraUtilitiesRefactorPolicy._MODULE_FAMILY_KEYS
                if key in normalized
            ),
            "other_private",
        )

    @staticmethod
    def policy_for_symbol(
        *,
        policy_context: t.Infra.PolicyContext | None,
        symbol_families: t.StrMapping | None,
        symbol_name: str,
    ) -> m.Infra.ClassNestingPolicy | None:
        """Resolve one validated policy entry for the symbol family in context."""
        if policy_context is None or symbol_families is None:
            return None
        family = symbol_families.get(symbol_name)
        if family is None:
            return None
        raw = policy_context.get(family)
        if raw is None:
            return None
        try:
            policy: m.Infra.ClassNestingPolicy = (
                m.Infra.ClassNestingPolicy.model_validate(raw)
            )
        except c.ValidationError:
            return None
        else:
            return policy

    @staticmethod
    def target_allowed(
        *, policy: m.Infra.ClassNestingPolicy, target_namespace: str
    ) -> bool:
        """Check whether policy allows writing the symbol to target namespace."""
        allowed_targets = tuple(policy.allowed_targets)
        if allowed_targets and target_namespace not in allowed_targets:
            return False
        return target_namespace not in policy.forbidden_targets

    @staticmethod
    def _class_nesting_target_matches(target_namespace: str, pattern: str) -> bool:
        """Check whether a target namespace matches a forbidden-target pattern."""
        if pattern.endswith(".*"):
            return target_namespace.lower().startswith(pattern[:-2].lower())
        return target_namespace == pattern

    @staticmethod
    def _class_nesting_violation(
        *,
        request: m.Infra.ClassNestingViolationRequest,
        policy_by_family: t.MappingKV[str, m.Infra.ClassNestingPolicy],
    ) -> t.StrMapping | None:
        """Build a policy violation payload when class nesting is forbidden."""
        policy = policy_by_family.get(request.family)
        if policy is None:
            return {
                c.Infra.RK_RULE_ID: f"precheck:{request.symbol}",
                c.Infra.RK_SOURCE_SYMBOL: request.symbol,
                c.Infra.RK_VIOLATION_TYPE: "unknown_module_family",
                c.Infra.RK_SUGGESTED_FIX: (
                    f"declare explicit policy for {request.family}"
                ),
            }
        operation = request.operation
        if operation not in policy.allowed_operations:
            return {
                c.Infra.RK_RULE_ID: f"precheck:{request.symbol}",
                c.Infra.RK_SOURCE_SYMBOL: request.symbol,
                c.Infra.RK_VIOLATION_TYPE: "operation_not_allowed",
                c.Infra.RK_SUGGESTED_FIX: (
                    f"allow {operation} in policy for {request.family}"
                ),
            }
        if operation in policy.forbidden_operations:
            return {
                c.Infra.RK_RULE_ID: f"precheck:{request.symbol}",
                c.Infra.RK_SOURCE_SYMBOL: request.symbol,
                c.Infra.RK_VIOLATION_TYPE: "operation_forbidden",
                c.Infra.RK_SUGGESTED_FIX: (
                    f"remove {operation} from forbidden_operations for {request.family}"
                ),
            }
        if any(
            FlextInfraUtilitiesRefactorPolicy._class_nesting_target_matches(
                request.target_namespace, pattern
            )
            for pattern in policy.forbidden_targets
        ):
            return {
                c.Infra.RK_RULE_ID: f"precheck:{request.symbol}",
                c.Infra.RK_SOURCE_SYMBOL: request.symbol,
                c.Infra.RK_VIOLATION_TYPE: "forbidden_target",
                c.Infra.RK_SUGGESTED_FIX: (
                    f"choose allowed target for family {request.family}"
                ),
            }
        return None

    @staticmethod
    def validate_class_nesting_entry(
        entry: t.StrMapping,
        *,
        policy_by_family: t.MappingKV[str, m.Infra.ClassNestingPolicy] | None = None,
        policy_path: Path | None = None,
    ) -> t.Pair[bool, t.StrMapping | None]:
        """Validate one class/helper nesting entry against the family policy."""
        symbol = entry.get(c.Infra.RK_LOOSE_NAME, "") or entry.get("helper_name", "")
        target_namespace = entry.get(c.Infra.RK_TARGET_NAMESPACE, "")
        current_file = entry.get(c.Infra.RK_CURRENT_FILE, "")
        if not symbol or not target_namespace or not current_file:
            return (True, None)
        family = FlextInfraUtilitiesRefactorPolicy.module_family_from_path(current_file)
        if family == "other_private":
            return (True, None)
        policies = (
            policy_by_family
            if policy_by_family is not None
            else FlextInfraUtilitiesRefactorPolicy.class_nesting_policy_by_family(
                policy_path
            )
        )
        operation = (
            c.Infra.RK_HELPER_CONSOLIDATION
            if bool(entry.get("helper_name", ""))
            else c.Infra.RK_CLASS_NESTING
        )
        request = m.Infra.ClassNestingViolationRequest(
            symbol=symbol,
            family=family,
            target_namespace=target_namespace,
            operation=operation,
        )
        violation = FlextInfraUtilitiesRefactorPolicy._class_nesting_violation(
            request=request, policy_by_family=policies
        )
        return (False, violation) if violation is not None else (True, None)


__all__: list[str] = ["FlextInfraUtilitiesRefactorPolicy"]
