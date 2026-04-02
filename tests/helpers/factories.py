from __future__ import annotations

from tests import m

from .factories_impl import (
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
    ServiceTestCaseFactory,
    ServiceTestCases,
    TestDataGenerators,
    UserFactory,
    ValidatingService,
    ValidatingServiceAuto,
    ValidatingServiceAutoFactory,
    ValidatingServiceFactory,
    reset_all_factories,
)


class TestHelperFactories:
    User = m.Core.User
    ServiceTestCase = m.Core.ServiceTestCase
    GetUserService = GetUserService
    ValidatingService = ValidatingService
    FailingService = FailingService
    GetUserServiceAuto = GetUserServiceAuto
    ValidatingServiceAuto = ValidatingServiceAuto
    FailingServiceAuto = FailingServiceAuto
    UserFactory = UserFactory
    GetUserServiceFactory = GetUserServiceFactory
    ValidatingServiceFactory = ValidatingServiceFactory
    FailingServiceFactory = FailingServiceFactory
    GetUserServiceAutoFactory = GetUserServiceAutoFactory
    ValidatingServiceAutoFactory = ValidatingServiceAutoFactory
    FailingServiceAutoFactory = FailingServiceAutoFactory
    ServiceTestCaseFactory = ServiceTestCaseFactory
    ServiceFactoryRegistry = ServiceFactoryRegistry
    TestDataGenerators = TestDataGenerators
    ServiceTestCases = ServiceTestCases
    GenericModelFactory = GenericModelFactory

    @staticmethod
    def reset_all_factories() -> None:
        reset_all_factories()
