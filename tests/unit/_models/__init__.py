# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Models package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.unit._models import (
        test_base,
        test_cqrs,
        test_entity,
        test_errors,
        test_exception_params,
    )
    from tests.unit._models.test_base import TestFlextModelsBase
    from tests.unit._models.test_cqrs import TestFlextModelsCqrs
    from tests.unit._models.test_entity import TestFlextModelsEntity
    from tests.unit._models.test_errors import TestFlextModelsErrors
    from tests.unit._models.test_exception_params import TestFlextModelsExceptionParams

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
    "TestFlextModelsBase": "tests.unit._models.test_base",
    "TestFlextModelsCqrs": "tests.unit._models.test_cqrs",
    "TestFlextModelsEntity": "tests.unit._models.test_entity",
    "TestFlextModelsErrors": "tests.unit._models.test_errors",
    "TestFlextModelsExceptionParams": "tests.unit._models.test_exception_params",
    "test_base": "tests.unit._models.test_base",
    "test_cqrs": "tests.unit._models.test_cqrs",
    "test_entity": "tests.unit._models.test_entity",
    "test_errors": "tests.unit._models.test_errors",
    "test_exception_params": "tests.unit._models.test_exception_params",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
