from types import TracebackType
from typing import Self

from _typeshed import Incomplete

from flext_core import FlextLogger

MAX_PAYMENT_AMOUNT: int
MAX_STATEMENTS_THRESHOLD: int
MAX_VALUE_DISPLAY_LENGTH: int

def demonstrate_basic_logging() -> None: ...
def demonstrate_logger_factory() -> None: ...
def demonstrate_context_management() -> None: ...
def demonstrate_exception_logging() -> None: ...
def demonstrate_unified_api() -> None: ...
def demonstrate_enterprise_patterns() -> None: ...

class PerformanceMonitor:
    logger: Incomplete
    operation: Incomplete
    context: Incomplete
    start_time: float
    def __init__(
        self, logger: FlextLogger, operation: str, **context: object
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

def main() -> None: ...
