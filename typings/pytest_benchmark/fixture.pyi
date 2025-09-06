from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

class BenchmarkFixture:
    group: str
    name: str | None
    extra_info: dict[str, object]

    def __call__(
        self, func: Callable[[], T], /, *args: object, **kwargs: object
    ) -> T: ...
    def pedantic(
        self, func: Callable[[], T], /, *args: object, **kwargs: object
    ) -> T: ...
    def timer(self, func: Callable[[], T], /, *args: object, **kwargs: object) -> T: ...
