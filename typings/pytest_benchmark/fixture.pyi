class BenchmarkFixture:
    group: str
    timer: object
    disable_gc: bool
    min_rounds: int

    def __call__(self, func: object) -> object: ...
