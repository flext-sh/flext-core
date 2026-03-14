# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Refactor package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.refactor.test_infra_refactor_analysis import (
        test_build_impact_map_extracts_rename_entries,
        test_build_impact_map_extracts_signature_entries,
        test_main_analyze_violations_is_read_only,
        test_main_analyze_violations_writes_json_report,
        test_violation_analysis_counts_massive_patterns,
        test_violation_analyzer_skips_non_utf8_files,
    )
    from tests.infra.unit.refactor.test_infra_refactor_class_and_propagation import (
        test_class_reconstructor_reorders_each_contiguous_method_block,
        test_class_reconstructor_reorders_methods_by_config,
        test_class_reconstructor_skips_interleaved_non_method_members,
        test_mro_checker_keeps_external_attribute_base,
        test_mro_redundancy_checker_removes_nested_attribute_inheritance,
        test_signature_propagation_removes_and_adds_keywords,
        test_signature_propagation_renames_call_keyword,
        test_symbol_propagation_keeps_alias_reference_when_asname_used,
        test_symbol_propagation_renames_import_and_local_references,
        test_symbol_propagation_updates_mro_base_references,
    )
    from tests.infra.unit.refactor.test_infra_refactor_engine import (
        test_engine_always_enables_class_nesting_file_rule,
        test_refactor_files_skips_non_python_inputs,
        test_refactor_project_scans_tests_and_scripts_dirs,
        test_rule_dispatch_fails_on_invalid_pattern_rule_config,
        test_rule_dispatch_fails_on_unknown_rule_mapping,
        test_rule_dispatch_keeps_legacy_id_fallback_mapping,
        test_rule_dispatch_prefers_fix_action_metadata,
    )
    from tests.infra.unit.refactor.test_infra_refactor_import_modernizer import (
        test_import_modernizer_adds_c_when_existing_c_is_aliased,
        test_import_modernizer_does_not_rewrite_function_parameter_shadow,
        test_import_modernizer_does_not_rewrite_rebound_local_name_usage,
        test_import_modernizer_partial_import_keeps_unmapped_symbols,
        test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias,
        test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function,
        test_import_modernizer_skips_when_runtime_alias_name_is_blocked,
        test_import_modernizer_updates_aliased_symbol_usage,
        test_lazy_import_rule_hoists_import_to_module_level,
        test_lazy_import_rule_uses_fix_action_for_hoist,
    )
    from tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations import (
        test_ensure_future_annotations_after_docstring,
        test_ensure_future_annotations_moves_existing_import_to_top,
        test_legacy_import_bypass_collapses_to_primary_import,
        test_legacy_rule_uses_fix_action_remove_for_aliases,
        test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias,
        test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias,
        test_legacy_wrapper_function_is_inlined_as_alias,
        test_legacy_wrapper_non_passthrough_is_not_inlined,
    )
    from tests.infra.unit.refactor.test_infra_refactor_pattern_corrections import (
        test_pattern_rule_converts_dict_annotations_to_mapping,
        test_pattern_rule_keeps_dict_param_when_copy_used,
        test_pattern_rule_keeps_dict_param_when_subscript_mutated,
        test_pattern_rule_keeps_type_cast_when_not_nested_object_cast,
        test_pattern_rule_optionally_converts_return_annotations_to_mapping,
        test_pattern_rule_removes_configured_redundant_casts,
        test_pattern_rule_removes_nested_type_object_cast_chain,
        test_pattern_rule_skips_overload_signatures,
    )
    from tests.infra.unit.refactor.test_infra_refactor_safety import (
        EngineSafetyStub,
        test_refactor_project_integrates_safety_manager,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EngineSafetyStub": (
        "tests.infra.unit.refactor.test_infra_refactor_safety",
        "EngineSafetyStub",
    ),
    "test_build_impact_map_extracts_rename_entries": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_build_impact_map_extracts_rename_entries",
    ),
    "test_build_impact_map_extracts_signature_entries": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_build_impact_map_extracts_signature_entries",
    ),
    "test_class_reconstructor_reorders_each_contiguous_method_block": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_class_reconstructor_reorders_each_contiguous_method_block",
    ),
    "test_class_reconstructor_reorders_methods_by_config": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_class_reconstructor_reorders_methods_by_config",
    ),
    "test_class_reconstructor_skips_interleaved_non_method_members": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_class_reconstructor_skips_interleaved_non_method_members",
    ),
    "test_engine_always_enables_class_nesting_file_rule": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_engine_always_enables_class_nesting_file_rule",
    ),
    "test_ensure_future_annotations_after_docstring": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_ensure_future_annotations_after_docstring",
    ),
    "test_ensure_future_annotations_moves_existing_import_to_top": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_ensure_future_annotations_moves_existing_import_to_top",
    ),
    "test_import_modernizer_adds_c_when_existing_c_is_aliased": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_adds_c_when_existing_c_is_aliased",
    ),
    "test_import_modernizer_does_not_rewrite_function_parameter_shadow": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_does_not_rewrite_function_parameter_shadow",
    ),
    "test_import_modernizer_does_not_rewrite_rebound_local_name_usage": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_does_not_rewrite_rebound_local_name_usage",
    ),
    "test_import_modernizer_partial_import_keeps_unmapped_symbols": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_partial_import_keeps_unmapped_symbols",
    ),
    "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias",
    ),
    "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function",
    ),
    "test_import_modernizer_skips_when_runtime_alias_name_is_blocked": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_skips_when_runtime_alias_name_is_blocked",
    ),
    "test_import_modernizer_updates_aliased_symbol_usage": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_updates_aliased_symbol_usage",
    ),
    "test_lazy_import_rule_hoists_import_to_module_level": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_lazy_import_rule_hoists_import_to_module_level",
    ),
    "test_lazy_import_rule_uses_fix_action_for_hoist": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_lazy_import_rule_uses_fix_action_for_hoist",
    ),
    "test_legacy_import_bypass_collapses_to_primary_import": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_import_bypass_collapses_to_primary_import",
    ),
    "test_legacy_rule_uses_fix_action_remove_for_aliases": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_rule_uses_fix_action_remove_for_aliases",
    ),
    "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_function_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_function_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_non_passthrough_is_not_inlined": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_non_passthrough_is_not_inlined",
    ),
    "test_main_analyze_violations_is_read_only": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_main_analyze_violations_is_read_only",
    ),
    "test_main_analyze_violations_writes_json_report": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_main_analyze_violations_writes_json_report",
    ),
    "test_mro_checker_keeps_external_attribute_base": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_mro_checker_keeps_external_attribute_base",
    ),
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    ),
    "test_pattern_rule_converts_dict_annotations_to_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_converts_dict_annotations_to_mapping",
    ),
    "test_pattern_rule_keeps_dict_param_when_copy_used": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_keeps_dict_param_when_copy_used",
    ),
    "test_pattern_rule_keeps_dict_param_when_subscript_mutated": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_keeps_dict_param_when_subscript_mutated",
    ),
    "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast",
    ),
    "test_pattern_rule_optionally_converts_return_annotations_to_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_optionally_converts_return_annotations_to_mapping",
    ),
    "test_pattern_rule_removes_configured_redundant_casts": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_removes_configured_redundant_casts",
    ),
    "test_pattern_rule_removes_nested_type_object_cast_chain": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_removes_nested_type_object_cast_chain",
    ),
    "test_pattern_rule_skips_overload_signatures": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_skips_overload_signatures",
    ),
    "test_refactor_files_skips_non_python_inputs": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_refactor_files_skips_non_python_inputs",
    ),
    "test_refactor_project_integrates_safety_manager": (
        "tests.infra.unit.refactor.test_infra_refactor_safety",
        "test_refactor_project_integrates_safety_manager",
    ),
    "test_refactor_project_scans_tests_and_scripts_dirs": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_refactor_project_scans_tests_and_scripts_dirs",
    ),
    "test_rule_dispatch_fails_on_invalid_pattern_rule_config": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_fails_on_invalid_pattern_rule_config",
    ),
    "test_rule_dispatch_fails_on_unknown_rule_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_fails_on_unknown_rule_mapping",
    ),
    "test_rule_dispatch_keeps_legacy_id_fallback_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_keeps_legacy_id_fallback_mapping",
    ),
    "test_rule_dispatch_prefers_fix_action_metadata": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_prefers_fix_action_metadata",
    ),
    "test_signature_propagation_removes_and_adds_keywords": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_signature_propagation_removes_and_adds_keywords",
    ),
    "test_signature_propagation_renames_call_keyword": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_signature_propagation_renames_call_keyword",
    ),
    "test_symbol_propagation_keeps_alias_reference_when_asname_used": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_symbol_propagation_keeps_alias_reference_when_asname_used",
    ),
    "test_symbol_propagation_renames_import_and_local_references": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_symbol_propagation_renames_import_and_local_references",
    ),
    "test_symbol_propagation_updates_mro_base_references": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_symbol_propagation_updates_mro_base_references",
    ),
    "test_violation_analysis_counts_massive_patterns": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_violation_analysis_counts_massive_patterns",
    ),
    "test_violation_analyzer_skips_non_utf8_files": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_violation_analyzer_skips_non_utf8_files",
    ),
}

__all__ = [
    "EngineSafetyStub",
    "test_build_impact_map_extracts_rename_entries",
    "test_build_impact_map_extracts_signature_entries",
    "test_class_reconstructor_reorders_each_contiguous_method_block",
    "test_class_reconstructor_reorders_methods_by_config",
    "test_class_reconstructor_skips_interleaved_non_method_members",
    "test_engine_always_enables_class_nesting_file_rule",
    "test_ensure_future_annotations_after_docstring",
    "test_ensure_future_annotations_moves_existing_import_to_top",
    "test_import_modernizer_adds_c_when_existing_c_is_aliased",
    "test_import_modernizer_does_not_rewrite_function_parameter_shadow",
    "test_import_modernizer_does_not_rewrite_rebound_local_name_usage",
    "test_import_modernizer_partial_import_keeps_unmapped_symbols",
    "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias",
    "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function",
    "test_import_modernizer_skips_when_runtime_alias_name_is_blocked",
    "test_import_modernizer_updates_aliased_symbol_usage",
    "test_lazy_import_rule_hoists_import_to_module_level",
    "test_lazy_import_rule_uses_fix_action_for_hoist",
    "test_legacy_import_bypass_collapses_to_primary_import",
    "test_legacy_rule_uses_fix_action_remove_for_aliases",
    "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias",
    "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias",
    "test_legacy_wrapper_function_is_inlined_as_alias",
    "test_legacy_wrapper_non_passthrough_is_not_inlined",
    "test_main_analyze_violations_is_read_only",
    "test_main_analyze_violations_writes_json_report",
    "test_mro_checker_keeps_external_attribute_base",
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    "test_pattern_rule_converts_dict_annotations_to_mapping",
    "test_pattern_rule_keeps_dict_param_when_copy_used",
    "test_pattern_rule_keeps_dict_param_when_subscript_mutated",
    "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast",
    "test_pattern_rule_optionally_converts_return_annotations_to_mapping",
    "test_pattern_rule_removes_configured_redundant_casts",
    "test_pattern_rule_removes_nested_type_object_cast_chain",
    "test_pattern_rule_skips_overload_signatures",
    "test_refactor_files_skips_non_python_inputs",
    "test_refactor_project_integrates_safety_manager",
    "test_refactor_project_scans_tests_and_scripts_dirs",
    "test_rule_dispatch_fails_on_invalid_pattern_rule_config",
    "test_rule_dispatch_fails_on_unknown_rule_mapping",
    "test_rule_dispatch_keeps_legacy_id_fallback_mapping",
    "test_rule_dispatch_prefers_fix_action_metadata",
    "test_signature_propagation_removes_and_adds_keywords",
    "test_signature_propagation_renames_call_keyword",
    "test_symbol_propagation_keeps_alias_reference_when_asname_used",
    "test_symbol_propagation_renames_import_and_local_references",
    "test_symbol_propagation_updates_mro_base_references",
    "test_violation_analysis_counts_massive_patterns",
    "test_violation_analyzer_skips_non_utf8_files",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
