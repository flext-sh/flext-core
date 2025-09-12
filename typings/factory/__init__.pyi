from typing import TypeVar

T = TypeVar("T")

class Factory[T]:
    class Meta:
        model: type | None = None
        django_get_or_create: tuple[str, ...] = ()
        abstract: bool = False
        strategy: str = "create"

    @classmethod
    def create(cls, **kwargs: object) -> T: ...

    @classmethod
    def build(cls, **kwargs: object) -> T: ...

    @classmethod
    def create_batch(cls, size: int, **kwargs: object) -> list[T]: ...

    @classmethod
    def build_batch(cls, size: int, **kwargs: object) -> list[T]: ...
