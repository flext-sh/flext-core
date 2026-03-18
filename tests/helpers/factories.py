from __future__ import annotations

from importlib import import_module
from typing import TypeAlias

_impl = import_module("tests.helpers._factories_impl")


class TestHelperFactories:
    User: TypeAlias = _impl.User
    ServiceTestCase: TypeAlias = _impl.ServiceTestCase
    GetUserService = _impl.GetUserService
    ValidatingService = _impl.ValidatingService
    FailingService = _impl.FailingService
    GetUserServiceAuto = _impl.GetUserServiceAuto
    ValidatingServiceAuto = _impl.ValidatingServiceAuto
    FailingServiceAuto = _impl.FailingServiceAuto
    UserFactory = _impl.UserFactory
    GetUserServiceFactory = _impl.GetUserServiceFactory
    ValidatingServiceFactory = _impl.ValidatingServiceFactory
    FailingServiceFactory = _impl.FailingServiceFactory
    GetUserServiceAutoFactory = _impl.GetUserServiceAutoFactory
    ValidatingServiceAutoFactory = _impl.ValidatingServiceAutoFactory
    FailingServiceAutoFactory = _impl.FailingServiceAutoFactory
    ServiceTestCaseFactory = _impl.ServiceTestCaseFactory
    ServiceFactoryRegistry = _impl.ServiceFactoryRegistry
    TestDataGenerators = _impl.TestDataGenerators
    ServiceTestCases = _impl.ServiceTestCases
    GenericModelFactory = _impl.GenericModelFactory

    @staticmethod
    def reset_all_factories() -> None:
        _impl.reset_all_factories()
