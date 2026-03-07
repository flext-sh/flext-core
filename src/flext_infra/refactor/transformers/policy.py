"""Shared policy helpers for class-nesting transformers."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from flext_infra import m, t


class FlextInfraRefactorTransformerPolicyUtilities:
    """Typed helpers for transformer policy parsing and checks."""

    @staticmethod
    def policy_for_symbol(
        *,
        policy_context: t.Infra.PolicyContext | None,
        symbol_families: Mapping[str, str] | None,
        symbol_name: str,
    ) -> m.Infra.Refactor.ClassNestingPolicy | None:
        """Resolve and validate policy for a symbol based on its family."""
        if policy_context is None or symbol_families is None:
            return None
        family = symbol_families.get(symbol_name)
        if family is None:
            return None
        raw = policy_context.get(family)
        if raw is None:
            return None
        try:
            return m.Infra.Refactor.ClassNestingPolicy.model_validate(raw)
        except ValidationError:
            return None

    @staticmethod
    def target_allowed(
        *,
        policy: m.Infra.Refactor.ClassNestingPolicy,
        target_namespace: str,
    ) -> bool:
        """Check whether policy allows writing the symbol to target namespace."""
        allowed_targets = tuple(policy.allowed_targets)
        if allowed_targets and target_namespace not in allowed_targets:
            return False
        return target_namespace not in policy.forbidden_targets


__all__ = ["FlextInfraRefactorTransformerPolicyUtilities"]
