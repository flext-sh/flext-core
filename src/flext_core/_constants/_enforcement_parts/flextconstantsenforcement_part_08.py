"""Fix-action constants for catalog-driven enforcement automation."""

from __future__ import annotations

from typing import Final

from flext_core._typings.base import FlextTypingBase as t


class FlextConstantsEnforcementFixActions:
    """Fix-action metadata consumed by flext-infra enforcement fixers."""

    ENFORCEMENT_FIX_ACTIONS: Final[t.MappingKV[str, t.JsonMapping]] = {
        "ENFORCE-008": {
            "kind": "transformer",
            "target": "future_import",
            "params": {},
            "safe": True,
        },
        "ENFORCE-016": {
            "kind": "transformer",
            "target": "typing_unifier",
            "params": {"targets": ["dict"]},
            "safe": True,
        },
        "ENFORCE-026": {
            "kind": "transformer",
            "target": "bare_except",
            "params": {},
            "safe": True,
        },
        "ENFORCE-027": {
            "kind": "transformer",
            "target": "print_to_logger",
            "params": {},
            "safe": False,
        },
        "ENFORCE-028": {
            "kind": "transformer",
            "target": "remove_breakpoint",
            "params": {},
            "safe": True,
        },
        "ENFORCE-029": {
            "kind": "transformer",
            "target": "open_encoding",
            "params": {},
            "safe": True,
        },
        "ENFORCE-030": {
            "kind": "transformer",
            "target": "typing_unifier",
            "params": {"targets": ["dict"]},
            "safe": True,
        },
        "ENFORCE-074": {
            "kind": "gate",
            "target": "smells",
            "params": {"smell_tag": "smell_boolean_logic"},
            "safe": True,
        },
    }


__all__: list[str] = ["FlextConstantsEnforcementFixActions"]
