"""User service factory helpers for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from itertools import count
from typing import ClassVar, override

from tests._utilities import TestsFlextUtilitiesRailwayServicesMixin


class TestsFlextUtilitiesUserFactoriesMixin(TestsFlextUtilitiesRailwayServicesMixin):
    """User service factory helpers."""

    class GetUserFactoryBase[T]:
        """Shared counter-rotating state for GetUser-style factories.

        Subclasses provide their own typed ``build``;
        the base owns the per-subclass ``_counter``, ``build_batch``, and ``reset``.
        """

        _counter: ClassVar[count[int]] = count(1)

        @classmethod
        def _resolve_user_id(cls, user_id: str | None) -> str:
            """Return ``user_id`` or the next auto-generated identifier.

            Returns:
                ``user_id`` or the next auto-generated identifier.

            """
            if user_id is not None:
                return user_id
            return f"user_{next(cls._counter):03d}"

        @classmethod
        def build(cls, *, user_id: str | None = None) -> T:
            """Build a GetUser-style instance; subclasses override with typed return."""
            raise NotImplementedError

        @classmethod
        def build_batch(cls, size: int) -> list[T]:
            """Build multiple GetUser-style instances with auto-generated values.

            Returns:
                The resulting ``list[T]``.

            """
            return [cls.build() for _ in range(size)]

        @classmethod
        def reset(cls) -> None:
            """Reset per-subclass factory counter."""
            cls._counter = count(1)

    class GetUserServiceFactory(
        GetUserFactoryBase[TestsFlextUtilitiesRailwayServicesMixin.GetUserService],
    ):
        """Factory for `GetUserService`."""

        @classmethod
        @override
        def build(
            cls,
            *,
            user_id: str | None = None,
        ) -> TestsFlextUtilitiesUserFactoriesMixin.GetUserService:
            """Build a `GetUserService` instance.

            Returns:
                The resulting ``TestsFlextUtilitiesUserFactoriesMixin.GetUserService``.

            """
            return TestsFlextUtilitiesUserFactoriesMixin.GetUserService(
                user_id=cls._resolve_user_id(user_id),
            )


__all__: list[str] = ["TestsFlextUtilitiesUserFactoriesMixin"]
