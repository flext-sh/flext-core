from collections.abc import Callable as Callable

def run_example_demonstration(
    title: str, examples: list[tuple[str, Callable[[], None]]]
) -> None: ...
