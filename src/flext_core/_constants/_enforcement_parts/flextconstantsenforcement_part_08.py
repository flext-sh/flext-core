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
            "target": "pattern",
            "params": {
                "patterns": [
                    {
                        "regex": r"^(?P<indent>\s*)except\s*:(?P<trail>.*)$",
                        "replacement": r"\g<indent>except Exception:\g<trail>",
                        "change_message": "Rewrote bare except to except Exception",
                        "flags": ["MULTILINE"],
                    },
                ],
            },
            "safe": True,
        },
        "ENFORCE-027": {
            "kind": "transformer",
            "target": "pattern",
            "params": {
                "patterns": [
                    {
                        "regex": r"\bprint\s*\(\s*(?P<args>[^)]*)\s*\)",
                        "replacement": r"u.fetch_logger(__name__).info(\g<args>)",
                        "change_message": "Rewrote print() to u.fetch_logger(__name__).info()",
                    },
                ],
                "required_alias": "u",
            },
            "safe": False,
        },
        "ENFORCE-028": {
            "kind": "transformer",
            "target": "pattern",
            "params": {
                "patterns": [
                    {
                        "regex": r"^[ \t]*breakpoint\s*\(\s*\)\s*[;\n]",
                        "replacement": "\n",
                        "change_message": "Removed debugger statement",
                        "flags": ["MULTILINE"],
                    },
                    {
                        "regex": r"^[ \t]*import\s+pdb\s*;\s*pdb\.set_trace\s*\(\s*\)\s*[;\n]",
                        "replacement": "\n",
                        "change_message": "Removed debugger statement",
                        "flags": ["MULTILINE"],
                    },
                ],
            },
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
        "ENFORCE-031": {
            "kind": "transformer",
            "target": "typing_dict_import",
            "params": {},
            "safe": True,
        },
        "ENFORCE-032": {
            "kind": "transformer",
            "target": "typing_dict_attr",
            "params": {},
            "safe": True,
        },
        "ENFORCE-033": {
            "kind": "transformer",
            "target": "hardcoded_version",
            "params": {},
            "safe": False,
        },
        "ENFORCE-045": {
            "kind": "transformer",
            "target": "import_modernizer",
            "params": {
                "imports_to_remove": ["pydantic"],
                "symbols_to_replace": {
                    "BaseModel": "m.BaseModel",
                    "ConfigDict": "m.ConfigDict",
                    "Field": "u.Field",
                    "PrivateAttr": "u.PrivateAttr",
                    "TypeAdapter": "m.TypeAdapter",
                    "computed_field": "u.computed_field",
                    "field_validator": "u.field_validator",
                    "model_validator": "u.model_validator",
                },
                "runtime_aliases": ["m", "u"],
                "blocked_aliases": [],
            },
            "safe": True,
        },
        "ENFORCE-048": {
            "kind": "transformer",
            "target": "mro_remover",
            "params": {},
            "safe": True,
        },
        "ENFORCE-064": {
            "kind": "rope",
            "target": "rewrite_compatibility_alias",
            "params": {},
            "safe": True,
        },
        "ENFORCE-066": {
            "kind": "rope",
            "target": "rewrite_compatibility_alias",
            "params": {},
            "safe": True,
        },
        "ENFORCE-067": {
            "kind": "rope",
            "target": "one_class_per_module",
            "params": {},
            "safe": False,
        },
        "ENFORCE-068": {
            "kind": "rope",
            "target": "rewrite_private_import_bypass",
            "params": {},
            "safe": True,
        },
        "ENFORCE-069": {
            "kind": "manual",
            "target": "deep_namespace_refactor",
            "params": {},
            "safe": False,
        },
        "ENFORCE-070": {
            "kind": "rope",
            "target": "rewrite_library_abstraction",
            "params": {},
            "safe": True,
        },
        "ENFORCE-074": {
            "kind": "gate",
            "target": "smells",
            "params": {"smell_tag": "smell_boolean_logic"},
            "safe": True,
        },
        "ENFORCE-079": {
            "kind": "rope",
            "target": "classvar_relocation",
            "params": {},
            "safe": True,
        },
        "ENFORCE-080": {
            "kind": "transformer",
            "target": "rewrite_foreign_canonical_alias",
            "params": {},
            "safe": True,
        },
        "ENFORCE-081": {
            "kind": "rope",
            "target": "hoist_inline_import",
            "params": {},
            "safe": True,
        },
        "ENFORCE-082": {
            "kind": "rope",
            "target": "fix_silent_failure_sentinels",
            "params": {},
            "safe": True,
        },
        "ENFORCE-083": {
            "kind": "gate",
            "target": "smells",
            "params": {"smell_tag": "type_ignore"},
            "safe": True,
        },
        "ENFORCE-084": {
            "kind": "gate",
            "target": "smells",
            "params": {"smell_tag": "noqa"},
            "safe": True,
        },
    }


__all__: list[str] = ["FlextConstantsEnforcementFixActions"]
