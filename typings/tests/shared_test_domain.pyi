from .test_shared_domain import (
    TestComplexValueObject,
    TestDomainFactory as TestDomainFactory,
    TestMoney,
    TestUser,
    TestUserStatus as TestUserStatus,
)

__all__ = [
    "ComplexValueObject",
    "ConcreteFlextEntity",
    "ConcreteValueObject",
    "TestDomainFactory",
    "TestUserStatus",
    "create_complex_test_value_object_safe",
    "create_test_entity_safe",
    "create_test_value_object_safe",
]

ConcreteFlextEntity = TestUser
ConcreteValueObject = TestMoney
ComplexValueObject = TestComplexValueObject

def create_test_entity_safe(name: str, **kwargs: object) -> TestUser: ...
def create_test_value_object_safe(
    amount: str, currency: str = "USD", **kwargs: object
) -> object: ...
def create_complex_test_value_object_safe(
    name: str, tags: list[str], metadata: dict[str, object]
) -> object: ...
