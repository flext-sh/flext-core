# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core.tests.unit._models.test_base import (
        TestsFlextModelsBase as TestsFlextModelsBase,
    )
    from flext_core.tests.unit._models.test_cqrs import (
        TestsFlextModelsCQRS as TestsFlextModelsCQRS,
    )
    from flext_core.tests.unit._models.test_enforcement_sources import (
        TestsFlextModelsEnforcementSources as TestsFlextModelsEnforcementSources,
    )
    from flext_core.tests.unit._models.test_entity import (
        TestsFlextModelsEntity as TestsFlextModelsEntity,
    )
    from flext_core.tests.unit._models.test_exception_params_core import (
        TestsFlextModelsExceptionParamsCore as TestsFlextModelsExceptionParamsCore,
    )
    from flext_core.tests.unit._models.test_exception_params_operations import (
        TestsFlextModelsExceptionParamsOperations as TestsFlextModelsExceptionParamsOperations,
    )
    from flext_core.tests.unit._models.test_exception_params_resources import (
        TestsFlextModelsExceptionParamsResources as TestsFlextModelsExceptionParamsResources,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_base": ("TestsFlextModelsBase",),
        ".test_cqrs": ("TestsFlextModelsCQRS",),
        ".test_enforcement_sources": ("TestsFlextModelsEnforcementSources",),
        ".test_entity": ("TestsFlextModelsEntity",),
        ".test_exception_params_core": ("TestsFlextModelsExceptionParamsCore",),
        ".test_exception_params_operations": (
            "TestsFlextModelsExceptionParamsOperations",
        ),
        ".test_exception_params_resources": (
            "TestsFlextModelsExceptionParamsResources",
        ),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
