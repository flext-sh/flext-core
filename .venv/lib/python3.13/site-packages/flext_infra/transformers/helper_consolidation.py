"""Helper consolidation transformer — rope-based implementation."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, override

from flext_infra import c, u
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import m, t


class FlextInfraHelperConsolidationTransformer(FlextInfraRopeTransformer):
    """Move top-level helper functions into target namespace classes."""

    def __init__(
        self,
        helper_mappings: t.StrMapping,
        policy_context: t.Infra.PolicyContext | None = None,
        helper_families: t.StrMapping | None = None,
        on_change: t.Infra.ChangeCallback = None,
    ) -> None:
        """Initialize with helper-to-namespace mappings and optional policy."""
        super().__init__(on_change=on_change)
        self._mappings = helper_mappings
        self._policy_context = policy_context
        self._families = helper_families or {}

    _description = "helper consolidation"

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply helper consolidation to in-memory source without persisting."""
        updated = source
        for name, ns in self._mappings.items():
            func_src = u.Infra.extract_definition(updated, name, kind="function")
            if func_src is None:
                continue
            if not self._policy_ok(name, ns, "enable_helper_consolidation"):
                continue
            if not self._sig_allowed(updated, name):
                continue
            updated = u.Infra.remove_definition(updated, name, kind="function")
            if "@staticmethod" not in func_src:
                func_src = f"@staticmethod\n{func_src}"
            updated = u.Infra.append_to_class_body(
                updated, ns, textwrap.indent(func_src, "    ")
            )
            self._record_change(f"Moved {name} into {ns}")
        updated = self._rewrite_calls(updated)
        return updated, list(self.changes)

    def _rewrite_calls(self, source: str) -> str:
        """Rewrite calls."""
        for name, ns in self._mappings.items():
            if not self._policy_ok(name, ns, "allow_helper_call_rewrite"):
                continue
            pat = c.Infra.compile_helper_call_site(name)
            new = pat.sub(f"{ns}.{name}(", source)
            if new != source:
                self._record_change(f"Rewritten call: {name}() -> {ns}.{name}()")
                source = new
        return source

    def _policy_ok(self, name: str, target_ns: str, attr: str) -> bool:
        """Policy ok."""
        policy = self._policy_for(name)
        if policy is None:
            return True
        if not getattr(policy, attr, True):
            return False
        allowed: bool = u.Infra.target_allowed(
            policy=policy, target_namespace=target_ns
        )
        return allowed

    def _sig_allowed(self, source: str, name: str) -> bool:
        """Sig allowed."""
        policy = self._policy_for(name)
        result: bool
        if policy is None or not policy.require_signature_validation:
            result = True
        else:
            sig_pat = c.Infra.compile_function_signature(name)
            match = sig_pat.search(source)
            if match is None:
                result = True
            else:
                params_str = match.group(1)
                param_names = u.Infra.parse_param_names(params_str)
                if (
                    any(r not in param_names for r in policy.required_parameters)
                    or any(f in param_names for f in policy.forbidden_parameters)
                    or (not policy.allow_vararg and "*args" in params_str)
                ):
                    result = False
                else:
                    result = not (not policy.allow_kwarg and "**" in params_str)
        return result

    def _policy_for(self, name: str) -> m.Infra.ClassNestingPolicy | None:
        """Policy for."""
        return u.Infra.policy_for_symbol(
            policy_context=self._policy_context,
            symbol_families=self._families,
            symbol_name=name,
        )


__all__: list[str] = ["FlextInfraHelperConsolidationTransformer"]
