# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _enforcement_data, _enforcement_parts, _errors_parts
    from ._enforcement_parts.flextconstantsenforcement_part_01 import (
        FlextConstantsEnforcementEnums,
    )
    from ._enforcement_parts.flextconstantsenforcement_part_02 import (
        FlextConstantsEnforcementRuntime,
    )
    from ._enforcement_parts.flextconstantsenforcement_part_03 import (
        FlextConstantsEnforcementNamespace,
    )
    from ._enforcement_parts.flextconstantsenforcement_part_04 import (
        FlextConstantsEnforcementRules,
    )
    from ._enforcement_parts.flextconstantsenforcement_part_06 import (
        FlextConstantsEnforcementTargets,
    )
    from ._enforcement_parts.flextconstantsenforcement_part_07 import (
        FlextConstantsEnforcementSmellData,
    )
    from ._errors_parts.flextconstantserrors_part_01 import FlextConstantsErrorsMessages
    from ._errors_parts.flextconstantserrors_part_02 import (
        FlextConstantsErrorsRuntimeExceptions,
    )
    from ._errors_parts.flextconstantserrors_part_03 import (
        FlextConstantsErrorsValidationExceptions,
    )
    from ._errors_parts.flextconstantserrors_part_04 import (
        FlextConstantsErrorsDomainParser,
    )
    from ._errors_parts.flextconstantserrors_part_05 import (
        FlextConstantsErrorsRuntimeSettings,
    )
    from .base import FlextConstantsBase
    from .config import FlextConstantsConfig
    from .cqrs import FlextConstantsCqrs
    from .enforcement import (
        FlextConstantsEnforcement,
        FlextMroViolation,
        FlextSmellViolation,
    )
    from .environment import FlextConstantsEnvironment
    from .errors import FlextConstantsErrors
    from .file import FlextConstantsFile
    from .guards import FlextConstantsGuards
    from .infrastructure import FlextConstantsInfrastructure
    from .loggings import FlextConstantsLogging
    from .mixins import FlextConstantsMixins
    from .project_metadata import FlextConstantsProjectMetadata
    from .pydantic import FlextConstantsPydantic
    from .regex import FlextConstantsRegex
    from .serialization import FlextConstantsSerialization
    from .settings import FlextConstantsSettings
    from .status import FlextConstantsStatus
    from .timeout import FlextConstantsTimeout
    from .validation import FlextConstantsValidation


__all__: tuple[str, ...] = (
    "FlextConstantsBase",
    "FlextConstantsConfig",
    "FlextConstantsCqrs",
    "FlextConstantsEnforcement",
    "FlextConstantsEnforcementEnums",
    "FlextConstantsEnforcementNamespace",
    "FlextConstantsEnforcementRules",
    "FlextConstantsEnforcementRuntime",
    "FlextConstantsEnforcementSmellData",
    "FlextConstantsEnforcementTargets",
    "FlextConstantsEnvironment",
    "FlextConstantsErrors",
    "FlextConstantsErrorsDomainParser",
    "FlextConstantsErrorsMessages",
    "FlextConstantsErrorsRuntimeExceptions",
    "FlextConstantsErrorsRuntimeSettings",
    "FlextConstantsErrorsValidationExceptions",
    "FlextConstantsFile",
    "FlextConstantsGuards",
    "FlextConstantsInfrastructure",
    "FlextConstantsLogging",
    "FlextConstantsMixins",
    "FlextConstantsProjectMetadata",
    "FlextConstantsPydantic",
    "FlextConstantsRegex",
    "FlextConstantsSerialization",
    "FlextConstantsSettings",
    "FlextConstantsStatus",
    "FlextConstantsTimeout",
    "FlextConstantsValidation",
    "FlextMroViolation",
    "FlextSmellViolation",
    "_enforcement_data",
    "_enforcement_parts",
    "_errors_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._enforcement_data": ("_enforcement_data",),
            "._enforcement_parts": ("_enforcement_parts",),
            "._enforcement_parts.flextconstantsenforcement_part_01": (
                "FlextConstantsEnforcementEnums",
            ),
            "._enforcement_parts.flextconstantsenforcement_part_02": (
                "FlextConstantsEnforcementRuntime",
            ),
            "._enforcement_parts.flextconstantsenforcement_part_03": (
                "FlextConstantsEnforcementNamespace",
            ),
            "._enforcement_parts.flextconstantsenforcement_part_04": (
                "FlextConstantsEnforcementRules",
            ),
            "._enforcement_parts.flextconstantsenforcement_part_06": (
                "FlextConstantsEnforcementTargets",
            ),
            "._enforcement_parts.flextconstantsenforcement_part_07": (
                "FlextConstantsEnforcementSmellData",
            ),
            "._errors_parts": ("_errors_parts",),
            "._errors_parts.flextconstantserrors_part_01": (
                "FlextConstantsErrorsMessages",
            ),
            "._errors_parts.flextconstantserrors_part_02": (
                "FlextConstantsErrorsRuntimeExceptions",
            ),
            "._errors_parts.flextconstantserrors_part_03": (
                "FlextConstantsErrorsValidationExceptions",
            ),
            "._errors_parts.flextconstantserrors_part_04": (
                "FlextConstantsErrorsDomainParser",
            ),
            "._errors_parts.flextconstantserrors_part_05": (
                "FlextConstantsErrorsRuntimeSettings",
            ),
            ".base": ("FlextConstantsBase",),
            ".config": ("FlextConstantsConfig",),
            ".cqrs": ("FlextConstantsCqrs",),
            ".enforcement": (
                "FlextConstantsEnforcement",
                "FlextMroViolation",
                "FlextSmellViolation",
            ),
            ".environment": ("FlextConstantsEnvironment",),
            ".errors": ("FlextConstantsErrors",),
            ".file": ("FlextConstantsFile",),
            ".guards": ("FlextConstantsGuards",),
            ".infrastructure": ("FlextConstantsInfrastructure",),
            ".loggings": ("FlextConstantsLogging",),
            ".mixins": ("FlextConstantsMixins",),
            ".project_metadata": ("FlextConstantsProjectMetadata",),
            ".pydantic": ("FlextConstantsPydantic",),
            ".regex": ("FlextConstantsRegex",),
            ".serialization": ("FlextConstantsSerialization",),
            ".settings": ("FlextConstantsSettings",),
            ".status": ("FlextConstantsStatus",),
            ".timeout": ("FlextConstantsTimeout",),
            ".validation": ("FlextConstantsValidation",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
