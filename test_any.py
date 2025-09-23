from collections.abc import Callable


def test() -> Callable[..., object]:
    return lambda *args, **kwargs: None
