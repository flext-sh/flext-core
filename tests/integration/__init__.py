# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Integration package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    import tests.integration.patterns as _tests_integration_patterns

    patterns = _tests_integration_patterns
    import tests.integration.test_architecture as _tests_integration_test_architecture

    test_architecture = _tests_integration_test_architecture
    import tests.integration.test_config_integration as _tests_integration_test_config_integration

    test_config_integration = _tests_integration_test_config_integration
    import tests.integration.test_integration as _tests_integration_test_integration

    test_integration = _tests_integration_test_integration
    import tests.integration.test_migration_validation as _tests_integration_test_migration_validation

    test_migration_validation = _tests_integration_test_migration_validation
    import tests.integration.test_service as _tests_integration_test_service

    test_service = _tests_integration_test_service
    import tests.integration.test_system as _tests_integration_test_system

    test_system = _tests_integration_test_system
    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.models import FlextModels as m
    from flext_core.protocols import FlextProtocols as p
    from flext_core.result import FlextResult as r
    from flext_core.service import FlextService as s
    from flext_core.typings import FlextTypes as t
    from flext_core.utilities import FlextUtilities as u
_LAZY_IMPORTS = merge_lazy_imports(
    ("tests.integration.patterns",),
    {
        "c": ("flext_core.constants", "FlextConstants"),
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "h": ("flext_core.handlers", "FlextHandlers"),
        "m": ("flext_core.models", "FlextModels"),
        "p": ("flext_core.protocols", "FlextProtocols"),
        "patterns": "tests.integration.patterns",
        "r": ("flext_core.result", "FlextResult"),
        "s": ("flext_core.service", "FlextService"),
        "t": ("flext_core.typings", "FlextTypes"),
        "test_architecture": "tests.integration.test_architecture",
        "test_config_integration": "tests.integration.test_config_integration",
        "test_integration": "tests.integration.test_integration",
        "test_migration_validation": "tests.integration.test_migration_validation",
        "test_service": "tests.integration.test_service",
        "test_system": "tests.integration.test_system",
        "u": ("flext_core.utilities", "FlextUtilities"),
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("logger", None)
_ = _LAZY_IMPORTS.pop("merge_lazy_imports", None)
_ = _LAZY_IMPORTS.pop("output", None)
_ = _LAZY_IMPORTS.pop("output_reporting", None)

__all__ = [
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "patterns",
    "r",
    "s",
    "t",
    "test_architecture",
    "test_config_integration",
    "test_integration",
    "test_migration_validation",
    "test_service",
    "test_system",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
