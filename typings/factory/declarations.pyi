"""Factory Boy declarations type stubs - FLEXT testing declarations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from collections.abc import Callable

class LazyAttribute:
    def __init__(self, function: Callable[[object], object]) -> None: ...

class LazyFunction:
    def __init__(self, function: Callable[[], object]) -> None: ...

class Sequence:
    def __init__(self, function: Callable[[int], object]) -> None: ...
