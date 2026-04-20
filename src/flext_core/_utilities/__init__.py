# AUTO-GENERATED FILE — Regenerate with: make gen
"""Utilities package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".args": ("FlextUtilitiesArgs",),
        ".beartype_conf": ("FlextUtilitiesBeartypeConf",),
        ".beartype_engine": ("FlextUtilitiesBeartypeEngine",),
        ".checker": ("FlextUtilitiesChecker",),
        ".collection": ("FlextUtilitiesCollection",),
        ".context": ("FlextUtilitiesContext",),
        ".context_crud": ("FlextUtilitiesContextCrud",),
        ".context_lifecycle": ("FlextUtilitiesContextLifecycle",),
        ".context_normalization": ("FlextUtilitiesContextNormalization",),
        ".context_scope": ("FlextUtilitiesContextScope",),
        ".context_tracing": ("FlextUtilitiesContextTracing",),
        ".conversion": ("FlextUtilitiesConversion",),
        ".discovery": ("FlextUtilitiesDiscovery",),
        ".domain": ("FlextUtilitiesDomain",),
        ".enforcement": ("FlextUtilitiesEnforcement",),
        ".enum": ("FlextUtilitiesEnum",),
        ".generators": ("FlextUtilitiesGenerators",),
        ".guards": ("FlextUtilitiesGuards",),
        ".guards_ensure": ("FlextUtilitiesGuardsEnsure",),
        ".guards_type": ("FlextUtilitiesGuardsType",),
        ".guards_type_core": ("FlextUtilitiesGuardsTypeCore",),
        ".guards_type_model": ("FlextUtilitiesGuardsTypeModel",),
        ".guards_type_protocol": ("FlextUtilitiesGuardsTypeProtocol",),
        ".handler": ("FlextUtilitiesHandler",),
        ".logging_config": ("FlextUtilitiesLoggingConfig",),
        ".logging_context": ("FlextUtilitiesLoggingContext",),
        ".mapper": ("FlextUtilitiesMapper",),
        ".model": ("FlextUtilitiesModel",),
        ".parser": ("FlextUtilitiesParser",),
        ".project_metadata": ("FlextUtilitiesProjectMetadata",),
        ".pydantic": ("FlextUtilitiesPydantic",),
        ".reliability": ("FlextUtilitiesReliability",),
        ".settings": ("FlextUtilitiesSettings",),
        ".text": ("FlextUtilitiesText",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
