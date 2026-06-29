"""User service factory helpers for flext-core tests."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import count
from typing import ClassVar, override

from flext_tests import m as tm

from tests import t
from tests._utilities.railway_services import TestsFlextUtilitiesRailwayServicesMixin


class TestsFlextUtilitiesUserFactoriesMixin(TestsFlextUtilitiesRailwayServicesMixin):
    """User service factory helpers."""

    class UserFactory:
        """Factory for `m.Tests.User` entities using native Python patterns."""

        _counter: ClassVar[count[int]] = count(1)
        _names: ClassVar[Sequence[str]] = [
            "Alice Johnson",
            "Bob Smith",
            "Carol Williams",
            "David Brown",
            "Eve Davis",
        ]
        _name_index: ClassVar[int] = 0

        @classmethod
        def _next_name(cls) -> str:
            """Get next name from rotation."""
            name = cls._names[cls._name_index % len(cls._names)]
            cls._name_index += 1
            return name

        @classmethod
        def build(
            cls,
            *,
            user_id: str | None = None,
            name: str | None = None,
            email: str | None = None,
            is_active: bool = True,
        ) -> tm.Tests.User:
            """Build a `tm.Tests.User` instance with optional overrides."""
            n = next(cls._counter)
            actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
            actual_name = name if name is not None else cls._next_name()
            actual_email = (
                email if email is not None else f"{actual_user_id}@example.com"
            )
            return tm.Tests.User(
                id=actual_user_id,
                unique_id=actual_user_id,
                name=actual_name,
                email=actual_email,
                active=is_active,
            )

        @classmethod
        def build_batch(cls, size: int) -> t.SequenceOf[tm.Tests.User]:
            """Build multiple `tm.Tests.User` instances with auto-generated values."""
            return [cls.build() for _ in range(size)]

        @classmethod
        def reset(cls) -> None:
            """Reset factory state for test isolation."""
            cls._counter = count(1)
            cls._name_index = 0

    class _GetUserFactoryBase[T]:
        """Shared counter-rotating state for GetUser-style factories.

        Subclasses provide their own typed ``build``;
        the base owns the per-subclass ``_counter``, ``build_batch``, and ``reset``.
        """

        _counter: ClassVar[count[int]] = count(1)

        @classmethod
        def _resolve_user_id(cls, user_id: str | None) -> str:
            """Return ``user_id`` or the next auto-generated identifier."""
            if user_id is not None:
                return user_id
            return f"user_{next(cls._counter):03d}"

        @classmethod
        def build(cls, *, user_id: str | None = None) -> T:
            """Build a GetUser-style instance; subclasses override with typed return."""
            raise NotImplementedError

        @classmethod
        def build_batch(cls, size: int) -> list[T]:
            """Build multiple GetUser-style instances with auto-generated values."""
            return [cls.build() for _ in range(size)]

        @classmethod
        def reset(cls) -> None:
            """Reset per-subclass factory counter."""
            cls._counter = count(1)

    class GetUserServiceFactory(
        _GetUserFactoryBase[TestsFlextUtilitiesRailwayServicesMixin.GetUserService]
    ):
        """Factory for `GetUserService`."""

        @classmethod
        @override
        def build(
            cls, *, user_id: str | None = None
        ) -> TestsFlextUtilitiesUserFactoriesMixin.GetUserService:
            """Build a `GetUserService` instance."""
            return TestsFlextUtilitiesUserFactoriesMixin.GetUserService(
                user_id=cls._resolve_user_id(user_id),
            )


__all__: list[str] = ["TestsFlextUtilitiesUserFactoriesMixin"]
