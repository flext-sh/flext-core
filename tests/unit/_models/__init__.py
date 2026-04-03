# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Models package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import tests.unit._models.test_base as _tests_unit__models_test_base

    test_base = _tests_unit__models_test_base
    import tests.unit._models.test_cqrs as _tests_unit__models_test_cqrs
    from tests.unit._models.test_base import TestFlextModelsBase

    test_cqrs = _tests_unit__models_test_cqrs
    import tests.unit._models.test_entity as _tests_unit__models_test_entity
    from tests.unit._models.test_cqrs import TestFlextModelsCqrs

    test_entity = _tests_unit__models_test_entity
    import tests.unit._models.test_errors as _tests_unit__models_test_errors
    from tests.unit._models.test_entity import TestFlextModelsEntity

    test_errors = _tests_unit__models_test_errors
    import tests.unit._models.test_exception_params as _tests_unit__models_test_exception_params
    from tests.unit._models.test_errors import TestFlextModelsErrors

    test_exception_params = _tests_unit__models_test_exception_params
    from tests.unit._models.test_exception_params import TestFlextModelsExceptionParams
_LAZY_IMPORTS = {
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

__all__ = [
    "TestFlextModelsBase",
    "TestFlextModelsCqrs",
    "TestFlextModelsEntity",
    "TestFlextModelsErrors",
    "TestFlextModelsExceptionParams",
    "test_base",
    "test_cqrs",
    "test_entity",
    "test_errors",
    "test_exception_params",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
