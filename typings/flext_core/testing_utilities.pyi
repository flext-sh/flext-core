from collections.abc import Callable
from typing import ClassVar, Protocol, TypeVar
from unittest.mock import MagicMock, Mock

from flext_core.models import FlextModel
from flext_core.result import FlextResult

__all__ = [
    "FlextTestAssertion",
    "FlextTestConfig",
    "FlextTestFactory",
    "FlextTestMocker",
    "FlextTestModel",
    "FlextTestUtilities",
    "ITestAssertion",
    "ITestFactory",
    "ITestMocker",
    "create_api_test_response",
    "create_ldap_test_config",
    "create_oud_connection_config",
]

T = TypeVar("T")
TTestData = TypeVar("TTestData")
TTestConfig = TypeVar("TTestConfig")

class ITestFactory(Protocol[T]):
    def create(self, **kwargs: object) -> T: ...
    def create_many(self, count: int, **kwargs: object) -> list[T]: ...

class ITestAssertion(Protocol):
    def assert_equals(self, actual: object, expected: object) -> None: ...
    def assert_true(self, *, condition: bool) -> None: ...
    def assert_false(self, *, condition: bool) -> None: ...

class ITestMocker(Protocol):
    def mock(self, spec: type | None = None) -> Mock: ...
    def patch(self, target: str) -> object: ...

class FlextTestUtilities:
    @staticmethod
    def create_test_result(
        *, success: bool = True, data: object = None, error: str | None = None
    ) -> FlextResult[object]: ...
    @staticmethod
    def assert_result_success(result: FlextResult[T]) -> T: ...
    @staticmethod
    def assert_result_failure(result: FlextResult[T]) -> str: ...
    @staticmethod
    def create_test_data(
        *, size: int = 10, prefix: str = "test"
    ) -> list[dict[str, object]]: ...

class FlextTestFactory[T]:
    def __init__(self, model_class: type[T]) -> None: ...
    def set_defaults(self, **defaults: object) -> FlextTestFactory[T]: ...
    def create(self, **kwargs: object) -> T: ...
    def create_many(self, count: int, **kwargs: object) -> list[T]: ...
    def create_batch(self, specifications: list[dict[str, object]]) -> list[T]: ...

class FlextTestAssertion:
    @staticmethod
    def assert_equals(
        actual: object, expected: object, message: str | None = None
    ) -> None: ...
    @staticmethod
    def assert_true(*, condition: bool, message: str | None = None) -> None: ...
    @staticmethod
    def assert_false(*, condition: bool, message: str | None = None) -> None: ...
    @staticmethod
    def assert_in(
        item: object,
        container: list[object] | dict[object, object] | set[object] | str,
        message: str | None = None,
    ) -> None: ...
    @staticmethod
    def assert_not_in(
        item: object,
        container: list[object] | dict[object, object] | set[object] | str,
        message: str | None = None,
    ) -> None: ...
    @staticmethod
    def assert_raises(
        exception_class: type[Exception],
        callable_obj: Callable[[], object],
        message: str | None = None,
    ) -> None: ...

class FlextTestMocker:
    @staticmethod
    def mock(spec: type | None = None, **kwargs: object) -> Mock: ...
    @staticmethod
    def magic_mock(spec: type | None = None, **kwargs: object) -> MagicMock: ...
    @staticmethod
    def patch(target: str, **kwargs: object) -> object: ...
    @staticmethod
    def patch_object(target: object, attribute: str, **kwargs: object) -> object: ...
    @staticmethod
    def create_async_mock(
        return_value: object = None, side_effect: object = None, **kwargs: object
    ) -> MagicMock: ...

class FlextTestModel(FlextModel):
    name: str
    value: int
    active: bool
    tags: ClassVar[list[str]]
    metadata: ClassVar[dict[str, object]]
    def activate(self) -> FlextResult[None]: ...
    def deactivate(self) -> FlextResult[None]: ...

class FlextTestConfig(FlextModel):
    debug: bool
    timeout: int
    retries: int
    base_url: str
    headers: ClassVar[dict[str, str]]

def create_oud_connection_config() -> dict[str, str]: ...
def create_ldap_test_config() -> dict[str, object]: ...
def create_api_test_response(
    *, success: bool = True, data: object = None
) -> dict[str, object]: ...
