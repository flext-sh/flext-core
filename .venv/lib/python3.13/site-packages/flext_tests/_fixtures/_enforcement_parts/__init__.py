# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Fixtures. Enforcement Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .build import build_items
    from .config import (
        SessionConfig,
        active_rules,
        discover_workspace_root,
        resolve_config,
        split_csv,
    )
    from .discovery import (
        collected_project_names,
        collected_validator_targets,
        load_infra_report,
        project_name_for_path,
    )
    from .items import EnforcementCollector, EnforcementItem, EnforcementViolationError
    from .namespace import NamespaceDetectorBuilder
    from .validators import build_tests_validator_items, dispatch_infra_detector
__all__: tuple[str, ...] = (
    "EnforcementCollector",
    "EnforcementItem",
    "EnforcementViolationError",
    "NamespaceDetectorBuilder",
    "SessionConfig",
    "active_rules",
    "build_items",
    "build_tests_validator_items",
    "collected_project_names",
    "collected_validator_targets",
    "discover_workspace_root",
    "dispatch_infra_detector",
    "load_infra_report",
    "project_name_for_path",
    "resolve_config",
    "split_csv",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".build": ("build_items",),
            ".config": (
                "SessionConfig",
                "active_rules",
                "discover_workspace_root",
                "resolve_config",
                "split_csv",
            ),
            ".discovery": (
                "collected_project_names",
                "collected_validator_targets",
                "load_infra_report",
                "project_name_for_path",
            ),
            ".items": (
                "EnforcementCollector",
                "EnforcementItem",
                "EnforcementViolationError",
            ),
            ".namespace": ("NamespaceDetectorBuilder",),
            ".validators": ("build_tests_validator_items", "dispatch_infra_detector"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
