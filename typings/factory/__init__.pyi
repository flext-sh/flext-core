from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

class Factory[T]:
    @classmethod
    def create(cls, **kwargs: object) -> T: ...
    @classmethod
    def create_batch(cls, size: int, **kwargs: object) -> list[T]: ...

class Faker[T]:
    def __init__(self, provider: str, **kwargs: object) -> None: ...

class LazyAttribute[T]:
    def __init__(self, function: Callable[[object], T]) -> None: ...

class LazyFunction:
    def __init__(self, function: Callable[[], object]) -> None: ...

class Sequence[T]:
    def __init__(self, function: Callable[[int], T]) -> None: ...
