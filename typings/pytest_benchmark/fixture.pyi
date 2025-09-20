"""Pytest benchmark fixture type stubs - FLEXT performance testing fixtures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from collections.abc import Callable
from typing import TypeVar

from flext_core import FlextTypes

T = TypeVar("T")

class BenchmarkFixture:
    group: str
    name: str | None
    extra_info: FlextTypes.Core.Dict

    def __call__(
        self,
        func: Callable[[], T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T: ...
    def pedantic(
        self,
        func: Callable[[], T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T: ...
    def timer(self, func: Callable[[], T], /, *args: object, **kwargs: object) -> T: ...
