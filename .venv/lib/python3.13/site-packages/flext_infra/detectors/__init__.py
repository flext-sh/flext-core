# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra.detectors package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .class_placement_detector import FlextInfraClassPlacementDetector
    from .compatibility_alias_detector import FlextInfraCompatibilityAliasDetector
    from .cyclic_import_detector import FlextInfraCyclicImportDetector
    from .deferred_self_reference_detector import (
        FlextInfraDeferredSelfReferenceDetector,
    )
    from .future_annotations_detector import FlextInfraFutureAnnotationsDetector
    from .import_alias_detector import FlextInfraImportAliasDetector
    from .inline_import_detector import FlextInfraInlineImportDetector
    from .internal_import_detector import FlextInfraInternalImportDetector
    from .loose_object_detector import FlextInfraLooseObjectDetector
    from .loose_test_function_detector import FlextInfraLooseTestFunctionDetector
    from .manual_protocol_detector import FlextInfraManualProtocolDetector
    from .manual_typing_alias_detector import FlextInfraManualTypingAliasDetector
    from .namespace_source_detector import FlextInfraNamespaceSourceDetector
    from .private_import_bypass_detector import FlextInfraPrivateImportBypassDetector
    from .runtime_alias_detector import FlextInfraRuntimeAliasDetector
    from .silent_failure_detector import FlextInfraSilentFailureDetector
__all__: tuple[str, ...] = (
    "FlextInfraClassPlacementDetector",
    "FlextInfraCompatibilityAliasDetector",
    "FlextInfraCyclicImportDetector",
    "FlextInfraDeferredSelfReferenceDetector",
    "FlextInfraFutureAnnotationsDetector",
    "FlextInfraImportAliasDetector",
    "FlextInfraInlineImportDetector",
    "FlextInfraInternalImportDetector",
    "FlextInfraLooseObjectDetector",
    "FlextInfraLooseTestFunctionDetector",
    "FlextInfraManualProtocolDetector",
    "FlextInfraManualTypingAliasDetector",
    "FlextInfraNamespaceSourceDetector",
    "FlextInfraPrivateImportBypassDetector",
    "FlextInfraRuntimeAliasDetector",
    "FlextInfraSilentFailureDetector",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".class_placement_detector": ("FlextInfraClassPlacementDetector",),
            ".compatibility_alias_detector": ("FlextInfraCompatibilityAliasDetector",),
            ".cyclic_import_detector": ("FlextInfraCyclicImportDetector",),
            ".deferred_self_reference_detector": (
                "FlextInfraDeferredSelfReferenceDetector",
            ),
            ".future_annotations_detector": ("FlextInfraFutureAnnotationsDetector",),
            ".import_alias_detector": ("FlextInfraImportAliasDetector",),
            ".inline_import_detector": ("FlextInfraInlineImportDetector",),
            ".internal_import_detector": ("FlextInfraInternalImportDetector",),
            ".loose_object_detector": ("FlextInfraLooseObjectDetector",),
            ".loose_test_function_detector": ("FlextInfraLooseTestFunctionDetector",),
            ".manual_protocol_detector": ("FlextInfraManualProtocolDetector",),
            ".manual_typing_alias_detector": ("FlextInfraManualTypingAliasDetector",),
            ".namespace_source_detector": ("FlextInfraNamespaceSourceDetector",),
            ".private_import_bypass_detector": (
                "FlextInfraPrivateImportBypassDetector",
            ),
            ".runtime_alias_detector": ("FlextInfraRuntimeAliasDetector",),
            ".silent_failure_detector": ("FlextInfraSilentFailureDetector",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
