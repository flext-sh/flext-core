# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Example scripts for FLEXT core components.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from examples.00_single_import_demo import (
        demonstrate_exceptions,
        demonstrate_utilities,
        execute_demonstrations,
        execute_service_operations,
        execute_validation_chain,
        identity,
        identity_result,
        ignore_and_return_none,
        process_user_data,
        validate_transform_user,
    )
    from examples.01_basic_result import (
        DemonstrationResult,
        RailwayService,
        RunDemonstrationCommand,
    )
    from examples.02_dependency_injection import (
        CacheService,
        DependencyInjectionService,
        EmailService,
    )
    from examples.03_models_basics import (
        DomainModelService,
        Email,
        Money,
        OrderItem,
        demonstrate_advanced_pydantic_mixins,
        demonstrate_enhanced_generic_models,
    )
    from examples.04_config_basics import (
        AppConfig,
        ConfigManagementService,
        demonstrate_file_config,
        run_main,
    )
    from examples.05_utilities_advanced import (
        AdvancedUtilitiesService,
        StatusEnum,
        UserModel,
    )
    from examples.06_decorators_complete import DecoratorsService
    from examples.07_registry_dispatcher import (
        CreateUserHandler,
        GetUserHandler,
        RegistryDispatcherService,
        UserCreatedEvent,
    )
    from examples.08_integration_complete import IntegrationService, Order, User
    from examples.09_context_management import (
        ContextManagementService,
        demonstrate_context_features,
    )
    from examples.12_utilities_comprehensive import (
        TEST_DATA,
        UtilitiesService,
        demonstrate_utility_composition,
    )
    from examples.14_flext_handlers_complete import (
        CommandHandler,
        CreateUserCommand,
        GetUserQuery,
        HandlersService,
        QueryHandler,
        UserDTO,
        demonstrate_cqrs_architecture,
    )
    from examples.15_automation_showcase import (
        AutomationService,
        OrderService,
        PaymentService,
        UserService,
    )
    from examples._models.ex00 import Ex00UserInput, Ex00UserProfile
    from examples._models.ex01 import (
        Ex01DemonstrationResult,
        Ex01DemonstrationResult as r,
        Ex01RunDemonstrationCommand,
        Ex01User,
    )
    from examples._models.ex02 import (
        Ex02CacheService,
        Ex02DatabaseService,
        Ex02DatabaseService as s,
        Ex02EmailService,
        Ex02TestConfig,
    )
    from examples._models.ex03 import (
        Ex03Email,
        Ex03Money,
        Ex03Order,
        Ex03OrderItem,
        Ex03User,
    )
    from examples._models.ex05 import Ex05StatusEnum, Ex05UserModel
    from examples._models.ex07 import (
        Ex07CreateUserCommand,
        Ex07GetUserQuery,
        Ex07UserCreatedEvent,
    )
    from examples._models.ex08 import Ex08Order, Ex08User
    from examples._models.ex10 import (
        Ex10DerivedMessage,
        Ex10Entity,
        Ex10Message,
        Ex10ProcessorBad,
        Ex10ProcessorGood,
    )
    from examples._models.ex11 import Ex11HandlerLikeService
    from examples._models.ex12 import Ex12CommandA, Ex12CommandB
    from examples._models.ex14 import Ex14CreateUserCommand, Ex14GetUserQuery
    from examples._models.exconfig import ExConfigAppConfig
    from examples.ex_01_flext_result import Ex01FlextResult
    from examples.ex_02_flext_settings import Ex02FlextSettings
    from examples.ex_03_flext_logger import Ex03FlextLogger
    from examples.ex_04_flext_dispatcher import Ex04FlextDispatcher
    from examples.ex_05_flext_mixins import Ex05FlextMixins, Ex05FlextMixins as x
    from examples.ex_06_flext_context import Ex06FlextContext
    from examples.ex_07_flext_exceptions import (
        Ex07FlextExceptions,
        Ex07FlextExceptions as e,
    )
    from examples.ex_08_flext_container import Ex08FlextContainer
    from examples.ex_09_flext_decorators import (
        Ex09FlextDecorators,
        Ex09FlextDecorators as d,
    )
    from examples.ex_10_flext_handlers import Ex10FlextHandlers, Ex10FlextHandlers as h
    from examples.ex_11_flext_service import Ex11FlextService
    from examples.ex_12_flext_registry import Ex12FlextRegistry
    from examples.logging_config_once_pattern import (
        DatabaseService,
        MigrationService,
        main,
    )
    from examples.models import (
        FlextCoreExampleModels,
        FlextCoreExampleModels as m,
        UserInput,
        UserProfile,
        em,
    )
    from examples.shared import Examples

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AdvancedUtilitiesService": ("examples.05_utilities_advanced", "AdvancedUtilitiesService"),
    "AppConfig": ("examples.04_config_basics", "AppConfig"),
    "AutomationService": ("examples.15_automation_showcase", "AutomationService"),
    "CacheService": ("examples.02_dependency_injection", "CacheService"),
    "CommandHandler": ("examples.14_flext_handlers_complete", "CommandHandler"),
    "ConfigManagementService": ("examples.04_config_basics", "ConfigManagementService"),
    "ContextManagementService": ("examples.09_context_management", "ContextManagementService"),
    "CreateUserCommand": ("examples.14_flext_handlers_complete", "CreateUserCommand"),
    "CreateUserHandler": ("examples.07_registry_dispatcher", "CreateUserHandler"),
    "DatabaseService": ("examples.logging_config_once_pattern", "DatabaseService"),
    "DecoratorsService": ("examples.06_decorators_complete", "DecoratorsService"),
    "DemonstrationResult": ("examples.01_basic_result", "DemonstrationResult"),
    "DependencyInjectionService": ("examples.02_dependency_injection", "DependencyInjectionService"),
    "DomainModelService": ("examples.03_models_basics", "DomainModelService"),
    "Email": ("examples.03_models_basics", "Email"),
    "EmailService": ("examples.02_dependency_injection", "EmailService"),
    "Ex00UserInput": ("examples._models.ex00", "Ex00UserInput"),
    "Ex00UserProfile": ("examples._models.ex00", "Ex00UserProfile"),
    "Ex01DemonstrationResult": ("examples._models.ex01", "Ex01DemonstrationResult"),
    "Ex01FlextResult": ("examples.ex_01_flext_result", "Ex01FlextResult"),
    "Ex01RunDemonstrationCommand": ("examples._models.ex01", "Ex01RunDemonstrationCommand"),
    "Ex01User": ("examples._models.ex01", "Ex01User"),
    "Ex02CacheService": ("examples._models.ex02", "Ex02CacheService"),
    "Ex02DatabaseService": ("examples._models.ex02", "Ex02DatabaseService"),
    "Ex02EmailService": ("examples._models.ex02", "Ex02EmailService"),
    "Ex02FlextSettings": ("examples.ex_02_flext_settings", "Ex02FlextSettings"),
    "Ex02TestConfig": ("examples._models.ex02", "Ex02TestConfig"),
    "Ex03Email": ("examples._models.ex03", "Ex03Email"),
    "Ex03FlextLogger": ("examples.ex_03_flext_logger", "Ex03FlextLogger"),
    "Ex03Money": ("examples._models.ex03", "Ex03Money"),
    "Ex03Order": ("examples._models.ex03", "Ex03Order"),
    "Ex03OrderItem": ("examples._models.ex03", "Ex03OrderItem"),
    "Ex03User": ("examples._models.ex03", "Ex03User"),
    "Ex04FlextDispatcher": ("examples.ex_04_flext_dispatcher", "Ex04FlextDispatcher"),
    "Ex05FlextMixins": ("examples.ex_05_flext_mixins", "Ex05FlextMixins"),
    "Ex05StatusEnum": ("examples._models.ex05", "Ex05StatusEnum"),
    "Ex05UserModel": ("examples._models.ex05", "Ex05UserModel"),
    "Ex06FlextContext": ("examples.ex_06_flext_context", "Ex06FlextContext"),
    "Ex07CreateUserCommand": ("examples._models.ex07", "Ex07CreateUserCommand"),
    "Ex07FlextExceptions": ("examples.ex_07_flext_exceptions", "Ex07FlextExceptions"),
    "Ex07GetUserQuery": ("examples._models.ex07", "Ex07GetUserQuery"),
    "Ex07UserCreatedEvent": ("examples._models.ex07", "Ex07UserCreatedEvent"),
    "Ex08FlextContainer": ("examples.ex_08_flext_container", "Ex08FlextContainer"),
    "Ex08Order": ("examples._models.ex08", "Ex08Order"),
    "Ex08User": ("examples._models.ex08", "Ex08User"),
    "Ex09FlextDecorators": ("examples.ex_09_flext_decorators", "Ex09FlextDecorators"),
    "Ex10DerivedMessage": ("examples._models.ex10", "Ex10DerivedMessage"),
    "Ex10Entity": ("examples._models.ex10", "Ex10Entity"),
    "Ex10FlextHandlers": ("examples.ex_10_flext_handlers", "Ex10FlextHandlers"),
    "Ex10Message": ("examples._models.ex10", "Ex10Message"),
    "Ex10ProcessorBad": ("examples._models.ex10", "Ex10ProcessorBad"),
    "Ex10ProcessorGood": ("examples._models.ex10", "Ex10ProcessorGood"),
    "Ex11FlextService": ("examples.ex_11_flext_service", "Ex11FlextService"),
    "Ex11HandlerLikeService": ("examples._models.ex11", "Ex11HandlerLikeService"),
    "Ex12CommandA": ("examples._models.ex12", "Ex12CommandA"),
    "Ex12CommandB": ("examples._models.ex12", "Ex12CommandB"),
    "Ex12FlextRegistry": ("examples.ex_12_flext_registry", "Ex12FlextRegistry"),
    "Ex14CreateUserCommand": ("examples._models.ex14", "Ex14CreateUserCommand"),
    "Ex14GetUserQuery": ("examples._models.ex14", "Ex14GetUserQuery"),
    "ExConfigAppConfig": ("examples._models.exconfig", "ExConfigAppConfig"),
    "Examples": ("examples.shared", "Examples"),
    "FlextCoreExampleModels": ("examples.models", "FlextCoreExampleModels"),
    "GetUserHandler": ("examples.07_registry_dispatcher", "GetUserHandler"),
    "GetUserQuery": ("examples.14_flext_handlers_complete", "GetUserQuery"),
    "HandlersService": ("examples.14_flext_handlers_complete", "HandlersService"),
    "IntegrationService": ("examples.08_integration_complete", "IntegrationService"),
    "MigrationService": ("examples.logging_config_once_pattern", "MigrationService"),
    "Money": ("examples.03_models_basics", "Money"),
    "Order": ("examples.08_integration_complete", "Order"),
    "OrderItem": ("examples.03_models_basics", "OrderItem"),
    "OrderService": ("examples.15_automation_showcase", "OrderService"),
    "PaymentService": ("examples.15_automation_showcase", "PaymentService"),
    "QueryHandler": ("examples.14_flext_handlers_complete", "QueryHandler"),
    "RailwayService": ("examples.01_basic_result", "RailwayService"),
    "RegistryDispatcherService": ("examples.07_registry_dispatcher", "RegistryDispatcherService"),
    "RunDemonstrationCommand": ("examples.01_basic_result", "RunDemonstrationCommand"),
    "StatusEnum": ("examples.05_utilities_advanced", "StatusEnum"),
    "TEST_DATA": ("examples.12_utilities_comprehensive", "TEST_DATA"),
    "User": ("examples.08_integration_complete", "User"),
    "UserCreatedEvent": ("examples.07_registry_dispatcher", "UserCreatedEvent"),
    "UserDTO": ("examples.14_flext_handlers_complete", "UserDTO"),
    "UserInput": ("examples.models", "UserInput"),
    "UserModel": ("examples.05_utilities_advanced", "UserModel"),
    "UserProfile": ("examples.models", "UserProfile"),
    "UserService": ("examples.15_automation_showcase", "UserService"),
    "UtilitiesService": ("examples.12_utilities_comprehensive", "UtilitiesService"),
    "d": ("examples.ex_09_flext_decorators", "Ex09FlextDecorators"),
    "demonstrate_advanced_pydantic_mixins": ("examples.03_models_basics", "demonstrate_advanced_pydantic_mixins"),
    "demonstrate_context_features": ("examples.09_context_management", "demonstrate_context_features"),
    "demonstrate_cqrs_architecture": ("examples.14_flext_handlers_complete", "demonstrate_cqrs_architecture"),
    "demonstrate_enhanced_generic_models": ("examples.03_models_basics", "demonstrate_enhanced_generic_models"),
    "demonstrate_exceptions": ("examples.00_single_import_demo", "demonstrate_exceptions"),
    "demonstrate_file_config": ("examples.04_config_basics", "demonstrate_file_config"),
    "demonstrate_utilities": ("examples.00_single_import_demo", "demonstrate_utilities"),
    "demonstrate_utility_composition": ("examples.12_utilities_comprehensive", "demonstrate_utility_composition"),
    "e": ("examples.ex_07_flext_exceptions", "Ex07FlextExceptions"),
    "em": ("examples.models", "em"),
    "execute_demonstrations": ("examples.00_single_import_demo", "execute_demonstrations"),
    "execute_service_operations": ("examples.00_single_import_demo", "execute_service_operations"),
    "execute_validation_chain": ("examples.00_single_import_demo", "execute_validation_chain"),
    "h": ("examples.ex_10_flext_handlers", "Ex10FlextHandlers"),
    "identity": ("examples.00_single_import_demo", "identity"),
    "identity_result": ("examples.00_single_import_demo", "identity_result"),
    "ignore_and_return_none": ("examples.00_single_import_demo", "ignore_and_return_none"),
    "m": ("examples.models", "FlextCoreExampleModels"),
    "main": ("examples.logging_config_once_pattern", "main"),
    "process_user_data": ("examples.00_single_import_demo", "process_user_data"),
    "r": ("examples._models.ex01", "Ex01DemonstrationResult"),
    "run_main": ("examples.04_config_basics", "run_main"),
    "s": ("examples._models.ex02", "Ex02DatabaseService"),
    "validate_transform_user": ("examples.00_single_import_demo", "validate_transform_user"),
    "x": ("examples.ex_05_flext_mixins", "Ex05FlextMixins"),
}

__all__ = [
    "AdvancedUtilitiesService",
    "AppConfig",
    "AutomationService",
    "CacheService",
    "CommandHandler",
    "ConfigManagementService",
    "ContextManagementService",
    "CreateUserCommand",
    "CreateUserHandler",
    "DatabaseService",
    "DecoratorsService",
    "DemonstrationResult",
    "DependencyInjectionService",
    "DomainModelService",
    "Email",
    "EmailService",
    "Ex00UserInput",
    "Ex00UserProfile",
    "Ex01DemonstrationResult",
    "Ex01FlextResult",
    "Ex01RunDemonstrationCommand",
    "Ex01User",
    "Ex02CacheService",
    "Ex02DatabaseService",
    "Ex02EmailService",
    "Ex02FlextSettings",
    "Ex02TestConfig",
    "Ex03Email",
    "Ex03FlextLogger",
    "Ex03Money",
    "Ex03Order",
    "Ex03OrderItem",
    "Ex03User",
    "Ex04FlextDispatcher",
    "Ex05FlextMixins",
    "Ex05StatusEnum",
    "Ex05UserModel",
    "Ex06FlextContext",
    "Ex07CreateUserCommand",
    "Ex07FlextExceptions",
    "Ex07GetUserQuery",
    "Ex07UserCreatedEvent",
    "Ex08FlextContainer",
    "Ex08Order",
    "Ex08User",
    "Ex09FlextDecorators",
    "Ex10DerivedMessage",
    "Ex10Entity",
    "Ex10FlextHandlers",
    "Ex10Message",
    "Ex10ProcessorBad",
    "Ex10ProcessorGood",
    "Ex11FlextService",
    "Ex11HandlerLikeService",
    "Ex12CommandA",
    "Ex12CommandB",
    "Ex12FlextRegistry",
    "Ex14CreateUserCommand",
    "Ex14GetUserQuery",
    "ExConfigAppConfig",
    "Examples",
    "FlextCoreExampleModels",
    "GetUserHandler",
    "GetUserQuery",
    "HandlersService",
    "IntegrationService",
    "MigrationService",
    "Money",
    "Order",
    "OrderItem",
    "OrderService",
    "PaymentService",
    "QueryHandler",
    "RailwayService",
    "RegistryDispatcherService",
    "RunDemonstrationCommand",
    "StatusEnum",
    "TEST_DATA",
    "User",
    "UserCreatedEvent",
    "UserDTO",
    "UserInput",
    "UserModel",
    "UserProfile",
    "UserService",
    "UtilitiesService",
    "d",
    "demonstrate_advanced_pydantic_mixins",
    "demonstrate_context_features",
    "demonstrate_cqrs_architecture",
    "demonstrate_enhanced_generic_models",
    "demonstrate_exceptions",
    "demonstrate_file_config",
    "demonstrate_utilities",
    "demonstrate_utility_composition",
    "e",
    "em",
    "execute_demonstrations",
    "execute_service_operations",
    "execute_validation_chain",
    "h",
    "identity",
    "identity_result",
    "ignore_and_return_none",
    "m",
    "main",
    "process_user_data",
    "r",
    "run_main",
    "s",
    "validate_transform_user",
    "x",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
