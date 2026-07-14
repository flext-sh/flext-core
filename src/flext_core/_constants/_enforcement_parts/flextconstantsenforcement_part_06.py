"""Namespace-target enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from collections.abc import Mapping

from flext_core._typings.base import FlextTypingBase as t


class FlextConstantsEnforcementTargets:
    """Target sets and external library ownership constants."""

    ENFORCEMENT_RECURSIVE_TAGS: Final[frozenset[str]] = frozenset({"const_mutable"})
    """Tags that must recurse into inner namespace classes during scanning."""

    ENFORCEMENT_NAMESPACE_TARGET_TAGS: Final[frozenset[str]] = frozenset({
        "alias_first_multi_parent",
        "alias_rebound_at_module_end",
        "cast_outside_core",
        "classvar_constant_outside_constants",
        "compatibility_alias_import",
        "cross_project_duplicate",
        "deprecated_typealias_syntax",
        "facade_base_is_alias_or_peer",
        "forbid_deep_namespace",
        "library_abstraction",
        "loc_cap",
        "model_rebuild_call",
        "nested_layer_misplacement",
        "no_concrete_namespace_import",
        "no_core_tests_namespace",
        "no_module_compat_alias",
        "no_private_module_bypass",
        "no_pydantic_consumer_import",
        "one_class_per_module",
        "no_redundant_inner_namespace",
        "no_self_root_import_in_core_files",
        "no_wrapper_root_alias_import",
        "pass_through_wrapper",
        "private_attr_probe",
        "settings_inheritance",
        "sibling_models_type_checking",
        "utilities_explicit_class_when_self_ref",
    })
    """NAMESPACE tags that use simple class-target dispatch (yield qn, (target,))."""

    ENFORCEMENT_CANONICAL_FILES: Final[frozenset[str]] = frozenset({
        "constants.py",
        "models.py",
        "protocols.py",
        "typings.py",
        "utilities.py",
    })
    """The five canonical facade files per project (AGENTS.md §2.2)."""

    ENFORCEMENT_ACCESSOR_RENAMES: Final[Mapping[str, t.StrPair]] = MappingProxyType({
        "is_success_result": (
            "successful_result",
            "Rename result helper to the canonical success helper",
        ),
        "is_failure_result": (
            "failed_result",
            "Rename result helper to the canonical failure helper",
        ),
        "is_success": (
            "success",
            "Rename boolean result predicate to the canonical success field",
        ),
        "is_failure": (
            "failure",
            "Rename boolean result predicate to the canonical failure field",
        ),
        "set_attribute": (
            "update_attribute",
            "Rewrite attribute mutator to the canonical update verb",
        ),
        "get_beartype_conf": (
            "build_beartype_conf",
            "Rewrite beartype settings accessor to the canonical build verb",
        ),
        "get_message_route": (
            "resolve_message_route",
            "Rewrite route accessor to the canonical resolve helper",
        ),
        "set_container_adapter": (
            "container_set_adapter",
            "Rewrite type adapter accessor to the canonical container_* name",
        ),
        "set_str_adapter": (
            "string_set_adapter",
            "Rewrite type adapter accessor to the canonical string_* name",
        ),
        "set_scalar_adapter": (
            "scalar_set_adapter",
            "Rewrite type adapter accessor to the canonical scalar_* name",
        ),
        "get_logger": (
            "fetch_logger",
            "Rewrite logger accessor to the canonical fetch verb",
        ),
        "is_structlog_configured": (
            "structlog_configured",
            "Rewrite structlog predicate to the canonical boolean helper",
        ),
        "get_log_level_from_config": (
            "resolve_log_level_from_config",
            "Rewrite log-level accessor to the canonical resolve helper",
        ),
        "get_version_string": (
            "resolve_version_string",
            "Rewrite version accessor to the canonical resolve helper",
        ),
        "get_version_info": (
            "resolve_version_info",
            "Rewrite version info accessor to the canonical resolve helper",
        ),
        "get_package_info": (
            "resolve_package_info",
            "Rewrite package info accessor to the canonical resolve helper",
        ),
        "is_version_at_least": (
            "version_at_least",
            "Rewrite version predicate to the canonical boolean helper",
        ),
    })
    """SSOT: legacy accessor name → (canonical replacement, human-readable reason).

    All entries target flext-core surface (origin="flext_core") — the data
    necessarily lives here because flext-core owns the names being renamed.
    Refactor verbs in flext-infra read this mapping; adding a new rename =
    one entry here, no parallel list.
    """

    ENFORCEMENT_COMPATIBILITY_ALIAS_RENAMES: Final[Mapping[str, str]] = (
        MappingProxyType({
            # flext-core canonical facade aliases
            "FlextConstants": "c",
            "FlextModels": "m",
            "FlextProtocols": "p",
            "FlextTypes": "t",
            "FlextUtilities": "u",
            "FlextResult": "r",
            # flext-cli canonical facade aliases
            "FlextCliConstants": "c",
            "FlextCliModels": "m",
            "FlextCliProtocols": "p",
            "FlextCliTypes": "t",
            "FlextCliUtilities": "u",
            # flext-infra canonical facade aliases
            "FlextInfraConstants": "c",
            "FlextInfraModels": "m",
            "FlextInfraProtocols": "p",
            "FlextInfraTypes": "t",
            "FlextInfraUtilities": "u",
            # flext-tests canonical facade aliases
            "FlextTestsConstants": "c",
            "FlextTestsModels": "m",
            "FlextTestsProtocols": "p",
            "FlextTestsTypes": "t",
            "FlextTestsUtilities": "u",
        })
    )
    """SSOT: long facade class name → canonical short alias.

    Any ``from <pkg> import <long_name>`` where ``<long_name>`` is present in
    this mapping must be rewritten to ``from <pkg> import <alias>``. The
    detector/rewriter in flext-infra sources this mapping from flext-core;
    adding a new compatibility alias = one entry here, no parallel list.
    """

    ENFORCEMENT_LIBRARY_OWNERS: Final[Mapping[str, str]] = MappingProxyType({
        "pydantic": "flext-core",
        "pydantic_settings": "flext-core",
        "pydantic_core": "flext-core",
        "dependency_injector": "flext-core",
        "returns": "flext-core",
        "structlog": "flext-core",
        "rich": "flext-cli",
        "rope": "flext-infra",
        "orjson": "flext-cli",
        "yaml": "flext-cli",
        "pyyaml": "flext-cli",
        "click": "flext-cli",
        "ldap3": "flext-ldap",
        "singer_sdk": "flext-meltano",
        "sqlalchemy": "flext-db-oracle",
        "oracledb": "flext-db-oracle",
        "grpc": "flext-grpc",
        "fastapi": "flext-web",
        "httpx": "flext-api",
    })
    """SSOT mapping: external library → owning FLEXT abstraction project (§2.7).

    Every consumer accesses these libraries via the owning project's facades
    (``c/m/p/t/u``), never via a bare top-level import. The runtime
    LIBRARY_IMPORT predicate (``m.Enforcement.LibraryImportParams``) and the
    rope-based source-level tier-whitelist validator both source their data
    from this mapping. Adding a new abstracted library = one entry here, no
    parallel list elsewhere.
    """


__all__: list[str] = ["FlextConstantsEnforcementTargets"]
