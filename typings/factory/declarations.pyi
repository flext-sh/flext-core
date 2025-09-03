from collections.abc import Callable

class LazyAttribute:
    def __init__(self, func: Callable[[object], object]) -> None: ...
