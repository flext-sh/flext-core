from collections.abc import Callable

class LazyAttribute:
    def __init__(self, function: Callable[[object], object]) -> None: ...

class LazyFunction:
    def __init__(self, function: Callable[[], object]) -> None: ...

class Sequence:
    def __init__(self, function: Callable[[int], object]) -> None: ...
