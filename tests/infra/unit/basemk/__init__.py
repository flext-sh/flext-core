# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Basemk package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.basemk.engine import (
        test_basemk_cli_generate_to_file,
        test_basemk_cli_generate_to_stdout,
        test_basemk_engine_execute_calls_render_all,
        test_basemk_engine_render_all_handles_template_error,
        test_basemk_engine_render_all_returns_string,
        test_basemk_engine_render_all_with_valid_config,
        test_generator_fails_for_invalid_make_syntax,
        test_generator_renders_with_config_override,
        test_generator_write_saves_output_file,
        test_render_all_generates_large_makefile,
        test_render_all_has_no_scripts_path_references,
    )
    from tests.infra.unit.basemk.generator import (
        test_generator_execute_returns_generated_content,
        test_generator_generate_propagates_render_failure,
        test_generator_generate_with_basemk_config_object,
        test_generator_generate_with_dict_config,
        test_generator_generate_with_invalid_dict_config,
        test_generator_generate_with_none_config_uses_default,
        test_generator_initializes_with_custom_engine,
        test_generator_initializes_with_default_engine,
        test_generator_write_creates_parent_directories,
        test_generator_write_fails_without_output_or_stream,
        test_generator_write_to_file,
        test_generator_write_to_stream,
    )
    from tests.infra.unit.basemk.generator_edge_cases import (
        test_generator_normalize_config_with_basemk_config,
        test_generator_normalize_config_with_dict,
        test_generator_normalize_config_with_invalid_dict,
        test_generator_normalize_config_with_none,
        test_generator_validate_generated_output_handles_oserror,
        test_generator_write_handles_file_permission_error,
        test_generator_write_to_stream_handles_oserror,
    )
    from tests.infra.unit.basemk.init import TestFlextInfraBaseMk
    from tests.infra.unit.basemk.main import (
        test_basemk_build_config_with_none,
        test_basemk_build_config_with_project_name,
        test_basemk_main_calls_sys_exit,
        test_basemk_main_ensures_structlog_configured,
        test_basemk_main_output_to_stdout,
        test_basemk_main_with_generate_command,
        test_basemk_main_with_generation_failure,
        test_basemk_main_with_invalid_command,
        test_basemk_main_with_no_command,
        test_basemk_main_with_none_argv,
        test_basemk_main_with_output_file,
        test_basemk_main_with_project_name,
        test_basemk_main_with_write_failure,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestFlextInfraBaseMk": ("tests.infra.unit.basemk.init", "TestFlextInfraBaseMk"),
    "test_basemk_build_config_with_none": (
        "tests.infra.unit.basemk.main",
        "test_basemk_build_config_with_none",
    ),
    "test_basemk_build_config_with_project_name": (
        "tests.infra.unit.basemk.main",
        "test_basemk_build_config_with_project_name",
    ),
    "test_basemk_cli_generate_to_file": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_cli_generate_to_file",
    ),
    "test_basemk_cli_generate_to_stdout": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_cli_generate_to_stdout",
    ),
    "test_basemk_engine_execute_calls_render_all": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_execute_calls_render_all",
    ),
    "test_basemk_engine_render_all_handles_template_error": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_render_all_handles_template_error",
    ),
    "test_basemk_engine_render_all_returns_string": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_render_all_returns_string",
    ),
    "test_basemk_engine_render_all_with_valid_config": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_render_all_with_valid_config",
    ),
    "test_basemk_main_calls_sys_exit": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_calls_sys_exit",
    ),
    "test_basemk_main_ensures_structlog_configured": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_ensures_structlog_configured",
    ),
    "test_basemk_main_output_to_stdout": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_output_to_stdout",
    ),
    "test_basemk_main_with_generate_command": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_generate_command",
    ),
    "test_basemk_main_with_generation_failure": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_generation_failure",
    ),
    "test_basemk_main_with_invalid_command": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_invalid_command",
    ),
    "test_basemk_main_with_no_command": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_no_command",
    ),
    "test_basemk_main_with_none_argv": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_none_argv",
    ),
    "test_basemk_main_with_output_file": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_output_file",
    ),
    "test_basemk_main_with_project_name": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_project_name",
    ),
    "test_basemk_main_with_write_failure": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_write_failure",
    ),
    "test_generator_execute_returns_generated_content": (
        "tests.infra.unit.basemk.generator",
        "test_generator_execute_returns_generated_content",
    ),
    "test_generator_fails_for_invalid_make_syntax": (
        "tests.infra.unit.basemk.engine",
        "test_generator_fails_for_invalid_make_syntax",
    ),
    "test_generator_generate_propagates_render_failure": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_propagates_render_failure",
    ),
    "test_generator_generate_with_basemk_config_object": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_basemk_config_object",
    ),
    "test_generator_generate_with_dict_config": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_dict_config",
    ),
    "test_generator_generate_with_invalid_dict_config": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_invalid_dict_config",
    ),
    "test_generator_generate_with_none_config_uses_default": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_none_config_uses_default",
    ),
    "test_generator_initializes_with_custom_engine": (
        "tests.infra.unit.basemk.generator",
        "test_generator_initializes_with_custom_engine",
    ),
    "test_generator_initializes_with_default_engine": (
        "tests.infra.unit.basemk.generator",
        "test_generator_initializes_with_default_engine",
    ),
    "test_generator_normalize_config_with_basemk_config": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_basemk_config",
    ),
    "test_generator_normalize_config_with_dict": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_dict",
    ),
    "test_generator_normalize_config_with_invalid_dict": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_invalid_dict",
    ),
    "test_generator_normalize_config_with_none": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_none",
    ),
    "test_generator_renders_with_config_override": (
        "tests.infra.unit.basemk.engine",
        "test_generator_renders_with_config_override",
    ),
    "test_generator_validate_generated_output_handles_oserror": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_validate_generated_output_handles_oserror",
    ),
    "test_generator_write_creates_parent_directories": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_creates_parent_directories",
    ),
    "test_generator_write_fails_without_output_or_stream": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_fails_without_output_or_stream",
    ),
    "test_generator_write_handles_file_permission_error": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_write_handles_file_permission_error",
    ),
    "test_generator_write_saves_output_file": (
        "tests.infra.unit.basemk.engine",
        "test_generator_write_saves_output_file",
    ),
    "test_generator_write_to_file": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_to_file",
    ),
    "test_generator_write_to_stream": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_to_stream",
    ),
    "test_generator_write_to_stream_handles_oserror": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_write_to_stream_handles_oserror",
    ),
    "test_render_all_generates_large_makefile": (
        "tests.infra.unit.basemk.engine",
        "test_render_all_generates_large_makefile",
    ),
    "test_render_all_has_no_scripts_path_references": (
        "tests.infra.unit.basemk.engine",
        "test_render_all_has_no_scripts_path_references",
    ),
}

__all__ = [
    "TestFlextInfraBaseMk",
    "test_basemk_build_config_with_none",
    "test_basemk_build_config_with_project_name",
    "test_basemk_cli_generate_to_file",
    "test_basemk_cli_generate_to_stdout",
    "test_basemk_engine_execute_calls_render_all",
    "test_basemk_engine_render_all_handles_template_error",
    "test_basemk_engine_render_all_returns_string",
    "test_basemk_engine_render_all_with_valid_config",
    "test_basemk_main_calls_sys_exit",
    "test_basemk_main_ensures_structlog_configured",
    "test_basemk_main_output_to_stdout",
    "test_basemk_main_with_generate_command",
    "test_basemk_main_with_generation_failure",
    "test_basemk_main_with_invalid_command",
    "test_basemk_main_with_no_command",
    "test_basemk_main_with_none_argv",
    "test_basemk_main_with_output_file",
    "test_basemk_main_with_project_name",
    "test_basemk_main_with_write_failure",
    "test_generator_execute_returns_generated_content",
    "test_generator_fails_for_invalid_make_syntax",
    "test_generator_generate_propagates_render_failure",
    "test_generator_generate_with_basemk_config_object",
    "test_generator_generate_with_dict_config",
    "test_generator_generate_with_invalid_dict_config",
    "test_generator_generate_with_none_config_uses_default",
    "test_generator_initializes_with_custom_engine",
    "test_generator_initializes_with_default_engine",
    "test_generator_normalize_config_with_basemk_config",
    "test_generator_normalize_config_with_dict",
    "test_generator_normalize_config_with_invalid_dict",
    "test_generator_normalize_config_with_none",
    "test_generator_renders_with_config_override",
    "test_generator_validate_generated_output_handles_oserror",
    "test_generator_write_creates_parent_directories",
    "test_generator_write_fails_without_output_or_stream",
    "test_generator_write_handles_file_permission_error",
    "test_generator_write_saves_output_file",
    "test_generator_write_to_file",
    "test_generator_write_to_stream",
    "test_generator_write_to_stream_handles_oserror",
    "test_render_all_generates_large_makefile",
    "test_render_all_has_no_scripts_path_references",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
