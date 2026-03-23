# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Test helpers for flext-core - service factories only.

This directory contains ONLY flext-core-specific service factories.
All generic test utilities come from flext_tests directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.helpers.factories import TestHelperFactories
    from tests.helpers.factories_impl import (
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        GenericModelFactory,
        GetUserService,
        GetUserServiceAuto,
        GetUserServiceAutoFactory,
        GetUserServiceFactory,
        ServiceFactoryRegistry,
        ServiceTestCase,
        ServiceTestCaseFactory,
        ServiceTestCases,
        TestDataGenerators,
        User,
        UserFactory,
        ValidatingService,
        ValidatingServiceAuto,
        ValidatingServiceAutoFactory,
        ValidatingServiceFactory,
        reset_all_factories,
    )
    from tests.helpers.scenarios import TestHelperScenarios

_LAZY_IMPORTS: Mapping[str, tuple[str, str]] = {
    "FailingService": ("tests.helpers.factories_impl", "FailingService"),
    "FailingServiceAuto": ("tests.helpers.factories_impl", "FailingServiceAuto"),
    "FailingServiceAutoFactory": (
        "tests.helpers.factories_impl",
        "FailingServiceAutoFactory",
    ),
    "FailingServiceFactory": ("tests.helpers.factories_impl", "FailingServiceFactory"),
    "GenericModelFactory": ("tests.helpers.factories_impl", "GenericModelFactory"),
    "GetUserService": ("tests.helpers.factories_impl", "GetUserService"),
    "GetUserServiceAuto": ("tests.helpers.factories_impl", "GetUserServiceAuto"),
    "GetUserServiceAutoFactory": (
        "tests.helpers.factories_impl",
        "GetUserServiceAutoFactory",
    ),
    "GetUserServiceFactory": ("tests.helpers.factories_impl", "GetUserServiceFactory"),
    "ServiceFactoryRegistry": (
        "tests.helpers.factories_impl",
        "ServiceFactoryRegistry",
    ),
    "ServiceTestCase": ("tests.helpers.factories_impl", "ServiceTestCase"),
    "ServiceTestCaseFactory": (
        "tests.helpers.factories_impl",
        "ServiceTestCaseFactory",
    ),
    "ServiceTestCases": ("tests.helpers.factories_impl", "ServiceTestCases"),
    "TestDataGenerators": ("tests.helpers.factories_impl", "TestDataGenerators"),
    "TestHelperFactories": ("tests.helpers.factories", "TestHelperFactories"),
    "TestHelperScenarios": ("tests.helpers.scenarios", "TestHelperScenarios"),
    "User": ("tests.helpers.factories_impl", "User"),
    "UserFactory": ("tests.helpers.factories_impl", "UserFactory"),
    "ValidatingService": ("tests.helpers.factories_impl", "ValidatingService"),
    "ValidatingServiceAuto": ("tests.helpers.factories_impl", "ValidatingServiceAuto"),
    "ValidatingServiceAutoFactory": (
        "tests.helpers.factories_impl",
        "ValidatingServiceAutoFactory",
    ),
    "ValidatingServiceFactory": (
        "tests.helpers.factories_impl",
        "ValidatingServiceFactory",
    ),
    "reset_all_factories": ("tests.helpers.factories_impl", "reset_all_factories"),
}

__all__ = [
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "TestDataGenerators",
    "TestHelperFactories",
    "TestHelperScenarios",
    "User",
    "UserFactory",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "reset_all_factories",
]


_LAZY_CACHE: MutableMapping[str, FlextTypes.ModuleExport] = {}


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562).

    A local cache ``_LAZY_CACHE`` persists resolved objects across repeated
    accesses during process lifetime.

    Args:
        name: Attribute name requested by dir()/import.

    Returns:
        Lazy-loaded module export type.

    Raises:
        AttributeError: If attribute not registered.

    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    value = lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> Sequence[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.

    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
