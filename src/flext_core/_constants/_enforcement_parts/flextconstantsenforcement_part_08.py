"""Fix-action constants for catalog-driven enforcement automation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
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
        "ENFORCE-039": {
            "kind": "transformer",
            "target": "cast_remover",
            "params": {},
            "safe": True,
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
        "ENFORCE-091": {
            "kind": "transformer",
            "target": "pattern",
            "params": {
                "patterns": [
                    {
                        "regex": r"\bList\s*\[",
                        "replacement": "t.SequenceOf[",
                        "change_message": "Rewrote List[...] to t.SequenceOf[...]",
                    },
                ],
                "required_alias": "t",
            },
            "safe": True,
        },
        "ENFORCE-092": {
            "kind": "transformer",
            "target": "pattern",
            "params": {
                "patterns": [
                    {
                        "regex": r"\btyping\s*\.\s*List\s*\[",
                        "replacement": "t.SequenceOf[",
                        "change_message": "Rewrote typing.List[...] to t.SequenceOf[...]",
                    },
                ],
                "required_alias": "t",
            },
            "safe": True,
        },
        "ENFORCE-093": {
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
        "ENFORCE-094": {
            "kind": "transformer",
            "target": "pattern",
            "params": {
                "patterns": [
                    {
                        "regex": r"\bstructlog\s*\.\s*get_logger\s*\(\s*\)",
                        "replacement": "u.fetch_logger(__name__)",
                        "change_message": "Rewrote structlog.get_logger() to u.fetch_logger(__name__)",
                    },
                    {
                        "regex": r"\bstructlog\s*\.\s*get_logger\s*\(\s*['\"](?P<name>[^'\"]*)['\"]\s*\)",
                        "replacement": r'u.fetch_logger("\g<name>")',
                        "change_message": "Rewrote structlog.get_logger(name) to u.fetch_logger(name)",
                    },
                ],
                "required_alias": "u",
            },
            "safe": False,
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
        "ENFORCE-090": {
            "kind": "rope",
            "target": "remove_stub_file",
            "params": {},
            "safe": True,
        },
        "ENFORCE-097": {
            "kind": "manual",
            "target": "extract_magic_literal",
            "params": {},
            "safe": False,
        },
        "ENFORCE-017": {
            "kind": "manual",
            "target": "remove_bypass",
            "params": {},
            "safe": True,
        },
        "ENFORCE-052": {
            "kind": "manual",
            "target": "move_to_type_checking",
            "params": {},
            "safe": True,
        },
        "ENFORCE-095": {
            "kind": "manual",
            "target": "route_oracledb_import",
            "params": {},
            "safe": True,
        },
        "ENFORCE-096": {
            "kind": "manual",
            "target": "route_ldap3_import",
            "params": {},
            "safe": True,
        },
        "ENFORCE-010": {
            "kind": "rope",
            "target": "rewrite_compatibility_alias",
            "params": {},
            "safe": True,
        },
        "ENFORCE-001": {
            "kind": "manual",
            "target": "relocate_loose_object",
            "params": {},
            "safe": False,
        },
        "ENFORCE-002": {
            "kind": "manual",
            "target": "rewrite_noncanonical_import",
            "params": {},
            "safe": False,
        },
        "ENFORCE-003": {
            "kind": "manual",
            "target": "fix_namespace_source",
            "params": {},
            "safe": False,
        },
        "ENFORCE-004": {
            "kind": "manual",
            "target": "fix_internal_import_boundary",
            "params": {},
            "safe": False,
        },
        "ENFORCE-005": {
            "kind": "manual",
            "target": "move_protocol_to_tree",
            "params": {},
            "safe": False,
        },
        "ENFORCE-006": {
            "kind": "manual",
            "target": "break_cyclic_import",
            "params": {},
            "safe": False,
        },
        "ENFORCE-007": {
            "kind": "manual",
            "target": "fix_runtime_alias_rebind",
            "params": {},
            "safe": False,
        },
        "ENFORCE-009": {
            "kind": "manual",
            "target": "move_typing_to_tree",
            "params": {},
            "safe": False,
        },
        "ENFORCE-011": {
            "kind": "manual",
            "target": "relocate_class_to_layer",
            "params": {},
            "safe": False,
        },
        "ENFORCE-012": {
            "kind": "manual",
            "target": "complete_mro_composition",
            "params": {},
            "safe": False,
        },
        "ENFORCE-013": {
            "kind": "manual",
            "target": "fix_parse_failure",
            "params": {},
            "safe": False,
        },
        "ENFORCE-014": {
            "kind": "manual",
            "target": "create_facade_files",
            "params": {},
            "safe": False,
        },
        "ENFORCE-015": {
            "kind": "manual",
            "target": "fix_import_discipline",
            "params": {},
            "safe": False,
        },
        "ENFORCE-018": {
            "kind": "manual",
            "target": "fix_layer_violation",
            "params": {},
            "safe": False,
        },
        "ENFORCE-019": {
            "kind": "manual",
            "target": "fix_test_pattern",
            "params": {},
            "safe": False,
        },
        "ENFORCE-020": {
            "kind": "manual",
            "target": "fix_pyproject_config",
            "params": {},
            "safe": False,
        },
        "ENFORCE-021": {
            "kind": "manual",
            "target": "fix_markdown_block",
            "params": {},
            "safe": False,
        },
        "ENFORCE-022": {
            "kind": "manual",
            "target": "fix_runtime_mro_violation",
            "params": {},
            "safe": False,
        },
        "ENFORCE-023": {
            "kind": "manual",
            "target": "remove_dynamic_any",
            "params": {},
            "safe": False,
        },
        "ENFORCE-024": {
            "kind": "manual",
            "target": "add_suppression_justification",
            "params": {},
            "safe": False,
        },
        "ENFORCE-025": {
            "kind": "manual",
            "target": "convert_relative_import",
            "params": {},
            "safe": False,
        },
        "ENFORCE-040": {
            "kind": "manual",
            "target": "add_suppression_justification",
            "params": {},
            "safe": False,
        },
        "ENFORCE-041": {
            "kind": "manual",
            "target": "resolve_forward_refs",
            "params": {},
            "safe": False,
        },
        "ENFORCE-042": {
            "kind": "manual",
            "target": "inherit_flext_settings",
            "params": {},
            "safe": False,
        },
        "ENFORCE-043": {
            "kind": "manual",
            "target": "inline_or_relocate_wrapper",
            "params": {},
            "safe": False,
        },
        "ENFORCE-044": {
            "kind": "manual",
            "target": "remove_private_attr_probe",
            "params": {},
            "safe": False,
        },
        "ENFORCE-046": {
            "kind": "manual",
            "target": "remove_blacklisted_import",
            "params": {},
            "safe": False,
        },
        "ENFORCE-047": {
            "kind": "manual",
            "target": "fix_facade_base_order",
            "params": {},
            "safe": False,
        },
        "ENFORCE-049": {
            "kind": "manual",
            "target": "reorder_facade_bases",
            "params": {},
            "safe": False,
        },
        "ENFORCE-050": {
            "kind": "manual",
            "target": "add_alias_rebind",
            "params": {},
            "safe": False,
        },
        "ENFORCE-051": {
            "kind": "manual",
            "target": "remove_local_alias_import",
            "params": {},
            "safe": False,
        },
        "ENFORCE-053": {
            "kind": "manual",
            "target": "fix_utility_parent_base",
            "params": {},
            "safe": False,
        },
        "ENFORCE-054": {
            "kind": "manual",
            "target": "remove_core_tests_path",
            "params": {},
            "safe": False,
        },
        "ENFORCE-055": {
            "kind": "manual",
            "target": "fix_wrapper_alias_import",
            "params": {},
            "safe": False,
        },
        "ENFORCE-071": {
            "kind": "manual",
            "target": "decompose_function_parameters",
            "params": {},
            "safe": False,
        },
        "ENFORCE-072": {
            "kind": "manual",
            "target": "reduce_return_statements",
            "params": {},
            "safe": False,
        },
        "ENFORCE-073": {
            "kind": "manual",
            "target": "flatten_nested_control_flow",
            "params": {},
            "safe": False,
        },
        "ENFORCE-075": {
            "kind": "manual",
            "target": "reduce_function_complexity",
            "params": {},
            "safe": False,
        },
        "ENFORCE-076": {
            "kind": "manual",
            "target": "split_complex_module",
            "params": {},
            "safe": False,
        },
        "ENFORCE-077": {
            "kind": "manual",
            "target": "extract_identical_code_block",
            "params": {},
            "safe": False,
        },
        "ENFORCE-078": {
            "kind": "manual",
            "target": "extract_similar_code_abstraction",
            "params": {},
            "safe": False,
        },
    }


__all__: list[str] = ["FlextConstantsEnforcementFixActions"]
