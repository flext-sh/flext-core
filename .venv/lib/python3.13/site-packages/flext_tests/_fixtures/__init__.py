# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Fixtures package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _enforcement_parts as _enforcement_parts
    from ._enforcement_parts.build import build_items
    from ._enforcement_parts.config import SessionConfig, resolve_config
    from ._enforcement_parts.discovery import (
        collected_project_names,
        collected_validator_targets,
        load_infra_report,
        project_name_for_path,
    )
    from ._enforcement_parts.namespace import NamespaceDetectorBuilder
    from ._enforcement_parts.validators import (
        build_tests_validator_items,
        dispatch_infra_detector,
    )
    from .enforcement import (
        EnforcementCollector,
        EnforcementItem,
        EnforcementViolationError,
        active_rules,
        discover_workspace_root,
        split_csv,
    )
    from .markdown_validation import (
        MarkdownCodeBlockCollector,
        MarkdownCodeBlockItem,
        MarkdownValidationError,
    )
    from .project_metadata import project_metadata, project_tool_flext
    from .settings import (
        clean_container,
        reset_settings,
        sample_data,
        settings,
        settings_factory,
        temp_dir,
        temp_file,
        test_context,
        test_runtime,
    )
__all__: tuple[str, ...] = (
    "EnforcementCollector",
    "EnforcementItem",
    "EnforcementViolationError",
    "MarkdownCodeBlockCollector",
    "MarkdownCodeBlockItem",
    "MarkdownValidationError",
    "NamespaceDetectorBuilder",
    "SessionConfig",
    "_enforcement_parts",
    "active_rules",
    "build_items",
    "build_tests_validator_items",
    "clean_container",
    "collected_project_names",
    "collected_validator_targets",
    "discover_workspace_root",
    "dispatch_infra_detector",
    "load_infra_report",
    "project_metadata",
    "project_name_for_path",
    "project_tool_flext",
    "reset_settings",
    "resolve_config",
    "sample_data",
    "settings",
    "settings_factory",
    "split_csv",
    "temp_dir",
    "temp_file",
    "test_context",
    "test_runtime",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._enforcement_parts": ("_enforcement_parts",),
            "._enforcement_parts.build": ("build_items",),
            "._enforcement_parts.config": ("SessionConfig", "resolve_config"),
            "._enforcement_parts.discovery": (
                "collected_project_names",
                "collected_validator_targets",
                "load_infra_report",
                "project_name_for_path",
            ),
            "._enforcement_parts.namespace": ("NamespaceDetectorBuilder",),
            "._enforcement_parts.validators": (
                "build_tests_validator_items",
                "dispatch_infra_detector",
            ),
            ".enforcement": (
                "EnforcementCollector",
                "EnforcementItem",
                "EnforcementViolationError",
                "active_rules",
                "discover_workspace_root",
                "split_csv",
            ),
            ".markdown_validation": (
                "MarkdownCodeBlockCollector",
                "MarkdownCodeBlockItem",
                "MarkdownValidationError",
            ),
            ".project_metadata": ("project_metadata", "project_tool_flext"),
            ".settings": (
                "clean_container",
                "reset_settings",
                "sample_data",
                "settings",
                "settings_factory",
                "temp_dir",
                "temp_file",
                "test_context",
                "test_runtime",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
